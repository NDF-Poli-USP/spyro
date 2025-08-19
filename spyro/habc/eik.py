import firedrake as fire
import numpy as np
from spyro.solvers.eikonal.eikonal_eq import Eikonal_Modeling

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Eikonal(Eikonal_Modeling):
    '''
    Class for the Nonlinear Eikonal used for HABC.

    Attributes
    ----------
    bnds : `tuple` of 'arrays'
        Mesh node indices on boundaries of the domain. Structure:
        - (left_boundary, right_boundary, bottom_boundary) for 2D
        - (left_boundary, right_boundary, bottom_boundary,
            left_bnd_y, right_bnd_y) for 3D
    bcs_eik : `list`
        Dirichlet BCs for eikonal
    c : `firedrake function`
        Velocity model without absorbing layer
    c_min : `float`
        Minimum velocity value in the model without absorbing layer
    comm : object
        An object representing the communication interface
        for parallel processing. Default is None
    diam_mesh : `ufl.geometry.CellDiameter`
        Mesh cell diameters
    funct_space_eik: `firedrake function space`
        Function space for the Eikonal modeling
    mesh: `firedrake mesh`
        Original mesh without absorbing layer
    node_positions : `tuple`
        Tuple containing the node positions in the mesh.
        - (z_data, x_data) for 2D
        - (z_data, x_data, y_data) for 3D
    path_save : `str`
        Path to save Eikonal results
    yp : `firedrake function`
        Eikonal field

    Methods
    -------
    define_bcs()
        Impose Dirichlet BCs for Eikonal equation
    ident_crit_eik()
        Identify the critical points at boundaries subject to reflections
    ident_eik_on_bnd()
        Identify Eikonal minimum values on boundary
    solve_eik()
        Solve the nonlinear Eikonal
    '''

    def __init__(self, Wave):
        '''
        Initialize the Eikonal class.

        Parameters
        ----------
        Wave : `wave`
            Wave object

        Returns
        -------
        None
        '''

        Eikonal_Modeling.__init__(self, Wave.dimension,
                                  Wave.sources.point_locations,
                                  ele_type=Wave.ele_type_eik,
                                  p_eik=Wave.p_eik, f_est=Wave.f_est)

        # Communicator MPI4py
        self.comm = Wave.comm

        # Setting the mesh
        self.mesh = Wave.mesh_original

        # Function space for the Eikonal modeling
        self.funct_space_eik = Wave.funct_space_eik

        # Mesh cell diameters
        self.diam_mesh = Wave.diam_mesh

        # Velocity profile model
        self.c = Wave.c

        # Minimum velocity value in the model
        self.c_min = Wave.c_min

        # Extract node positions
        self.node_positions = Wave.extract_node_positions(self.funct_space_eik)

        # Extract boundary node indices
        self.bnds = Wave.extract_bnd_node_indices(self.node_positions,
                                                  self.funct_space_eik)

        # Path to save data
        self.path_save = Wave.path_save + "preamble/"

        # Eikonal boundary conditions
        self.define_bcs()

    def define_bcs(self):
        '''
        Impose Dirichlet BCs for eikonal equation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nDefining Eikonal BCs")

        self.bcs_eik, sou_marker = self.eikonal_bcs(self.node_positions,
                                                    self.funct_space_eik)

        # Save source marker
        outfile = fire.VTKFile(self.path_save + "souEik.pvd")
        outfile.write(sou_marker)

    def solve_eik(self):
        '''
        Solve the nonlinear Eikonal.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Eikonal solution
        self.yp = self.eikonal_solver(self.c, self.c_min,
                                      self.funct_space_eik,
                                      self.diam_mesh)

        # Save Eikonal results
        eikonal_file = fire.VTKFile(self.path_save + "Eik.pvd")
        eikonal_file.write(self.yp)

    def ident_eik_on_bnd(self, boundary):
        '''
        Identify Eikonal minimum values on a boundary.

        Parameters
        ----------
        boundary : `array`
            Domain boundary subject to reflections

        Returns
        -------
        eikmin : `float`
            Minimum eikonal value
        idxmin : 'int'
            Array index corresponding to the minimum eikonal value
        '''

        boundary_eik = self.yp.dat.data_with_halos[boundary]
        eikmin = boundary_eik.min()
        idxbnd = np.where(boundary_eik == eikmin)[0][0]
        idxmin = boundary[0][idxbnd]  # Original index

        return eikmin, idxmin

    def ident_crit_eik(self):
        '''
        Identify the critical points at boundaries subject to reflections

        Parameters
        ----------
        None

        Returns
        -------
        eik_bnd: `list`
            Properties on boundaries according to minimum values of Eikonal
            Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
            - pt_cr : Critical point coordinates
            - c_bnd : Propagation speed at critical point
            - eikmin : Eikonal value in seconds
            - z_par : Inverse of minimum Eikonal (Equivalent to c_bound/lref)
            - lref : Distance to the closest source from critical point
            - sou_cr : Critical source coordinates
        '''

        # Node positions and boundary strings
        z_data, x_data = self.node_positions[:2]
        if self.dimension == 2:  # 2D
            bnds_str = ['Left Boundary', 'Right Boundary', 'Bottom Boundary']
        if self.dimension == 3:  # 3D
            y_data = self.node_positions[2]
            bnds_str = ['Xmin Boundary', 'Xmax Boundary', 'Bottom Boundary',
                        'Ymin Boundary', 'Ymax Boundary', ]

        print("\nIdentifying Critical Points on Boundaries")

        # Loop over boundaries
        eik_bnd = []
        eik_str = "Min Eikonal on {0:>16} (ms): {1:>7.3f} "
        for bnd, bnd_str in zip(self.bnds, bnds_str):

            # Identify Eikonal minimum
            eikmin, idxmin = self.ident_eik_on_bnd(bnd)

            # Identify critical point coordinates
            pt_cr = (z_data[idxmin], x_data[idxmin])
            pnt_str = "at (in km): ({2:3.3f}, {3:3.3f})"
            if self.dimension == 3:  # 3D
                pt_cr += (y_data[idxmin],)
                pnt_str = pnt_str[:-1] + ", {4:3.3f})"

            # Identifying propagation speed at critical point
            c_bnd = np.float64(self.c.at(pt_cr).item())

            # Print critical point coordinates
            print((eik_str + pnt_str).format(bnd_str, 1e3 * eikmin, *pt_cr))

            # Identify closest source
            lref_allsou = np.linalg.norm(
                np.asarray(pt_cr) - np.asarray(self.source_locations), axis=1)
            idxsou = np.argmin(lref_allsou)
            lref = lref_allsou[idxsou]
            sou_cr = self.source_locations[idxsou]
            z_par = 1 / eikmin

            # Grouping properties
            eik_bnd.append([pt_cr, c_bnd, eikmin, z_par, lref, sou_cr])

        # Sort the list by the minimum Eikonal and then by the maximum velocity
        return sorted(eik_bnd, key=lambda x: (x[2], -x[1]))
