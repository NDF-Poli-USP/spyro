import firedrake as fire
import numpy as np
from ..solvers.eikonal.eikonal_eq import Eikonal_Modeling
from spyro.tools.habc_tools import point_cloud_field

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
    bcs_eik : `list`
        Dirichlet BCs for eikonal
    boundaries : `tuple`
        Tuple containing the boundary boolean labels for applying absorbing BCs.
        - (absorb_top, absorb_bottom, absorb_right, absorb_left) for 2D
        - (absorb_top, absorb_bottom, absorb_right,
            absorb_left, absorb_front, absorb_back) for 3D
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
    lmin : `float`
        Minimum mesh size
    mesh : `Firedrake.Mesh`
        Original mesh without absorbing layer
    mesh_ops : `spyro.meshing.meshing_operations.MeshOps`
        Object with general mesh operations for domains w/o an absorbing layer
    node_positions : `array`
        Node positions of the mesh
        - array of shape (num_nodes, 2) and coordinates (z, x) for 2D
        - array of shape (num_nodes, 3) and coordinates (z, x, y) for 3D
    node_tol : `float`
        Tolerance for identifying minimum Eikonal values on boundaries
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

        Eikonal_Modeling.__init__(
            self, Wave.dimension, Wave.sources.point_locations,
            ele_type=Wave.ele_type_eik, p_eik=Wave.p_eik, f_est=Wave.f_est)

        # Communicator MPI
        self.comm = Wave.comm

        # Setting the mesh
        self.mesh = Wave.mesh_original

        # Function space for the Eikonal modeling
        self.funct_space_eik = Wave.funct_space_eik

        # Mesh cell diameters
        self.diam_mesh = Wave.mesh_parameters.diam_mesh

        # Minimum mesh size
        self.lmin = Wave.mesh_parameters.lmin

        # Velocity profile model
        self.c = Wave.c

        # Minimum velocity value in the model
        self.c_min = Wave.c_min

        # Absorbing boundaries
        self.boundaries = Wave.get_absorbing_boundaries()

        # Mesh operations
        self.mesh_ops = Wave.mesh_ops

        # Extract node positions
        self.node_positions = self.mesh_ops.extract_node_positions(self.mesh,
                                                                   self.funct_space_eik,
                                                                   output_type="array")

        # Tolerance for identifying minimum Eikonal values on boundaries
        self.node_tol = Wave.mesh_parameters.tol

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

        # Define Eikonal BCs and source marker
        self.bcs_eik, sou_marker = self.eikonal_bcs(self.node_positions,
                                                    self.funct_space_eik,
                                                    self.lmin)

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

    def ident_eik_on_bnd(self, bnd_ids):
        '''
        Identify Eikonal minimum values on a boundary.

        Parameters
        ----------
        bnd_ids : `array`
            IDs of the boundary subject to reflections

        Returns
        -------
        eikmin : `float`
            Minimum eikonal value
        pnt_crit : `array`
            Critical point coordinates
        '''

        # Create a point cloud to get the minimum eikonal value on the boundary
        ptos_bnd = self.node_positions[bnd_ids, :]
        eik_on_boundary = point_cloud_field(
            self.mesh, ptos_bnd, self.yp, self.node_tol).dat.data_with_halos[:]

        # Identify minimum Eikonal value on the boundary
        eikmin = eik_on_boundary.min()

        # Identify critical point coordinates
        pnt_crit = ptos_bnd[eik_on_boundary.argmin(), :]

        return eikmin, pnt_crit

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
            Structure sublist: [pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit]
            - pnt_crit : Critical point coordinates
            - c_bnd : Propagation speed at critical point
            - eikmin : Eikonal value in seconds
            - z_par : Inverse of minimum Eikonal (Equivalent to c_bound/lref)
            - lref : Distance to the closest source from critical point
            - sou_crit : Critical source coordinates
        '''

        # Build the boundary ID mapping
        self.boundary_nodes_ids = self.mesh_ops.mapping_boundary_ids(
            self.mesh, self.funct_space_eik, self.boundaries,
            box_domain=True, get_boundary_node_ids=True)[1]

        print("\nIdentifying Critical Points on Boundaries")

        # Loop over boundaries
        eik_bnd = []
        eik_str = "Min Eikonal on {0:>4} (ms): {1:>7.3f} "
        for bnd_str, (bnd_ids, status) in self.boundary_nodes_ids.items():
            if not status:
                continue

            # Identify minimum Eikonal and critical point on the boundary
            eikmin, pnt_crit = self.ident_eik_on_bnd(bnd_ids)

            # Identifying propagation speed at critical point
            c_bnd = np.float64(self.c.at(pnt_crit).item())

            # Print critical point coordinates
            pnt_str = "at (in km): ({2:3.3f}, {3:3.3f})"
            if self.dimension == 3:  # 3D
                pnt_str = pnt_str[:-1] + ", {4:3.3f})"
            print((eik_str + pnt_str).format(bnd_str, 1e3 * eikmin, *pnt_crit))

            # Identify closest source
            lref_allsou = np.linalg.norm(
                np.asarray(pnt_crit) - np.asarray(self.source_locations), axis=1)
            idxsou = lref_allsou.argmin()
            lref = lref_allsou[idxsou]
            sou_crit = self.source_locations[idxsou]
            z_par = 1. / eikmin

            # Grouping properties
            eik_bnd.append([pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit])

        # Sort the list by the minimum Eikonal and then by the maximum velocity
        return sorted(eik_bnd, key=lambda x: (x[2], -x[1]))
