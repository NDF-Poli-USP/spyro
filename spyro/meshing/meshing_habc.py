import firedrake as fire
import numpy as np
from spyro.utils.error_management import value_parameter_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Mesh():
    '''
    Class for HABC mesh generation

    Attributes
    ----------
    bnds : `list` of `arrays`
        Mesh point indices on boundaries of the domain. Structure:
        - [left_boundary, right_boundary, bottom_boundary] for 2D
        - [left_boundary, right_boundary, bottom_boundary,
            left_bnd_y, right_bnd_y] for 3D
    bnd_nodes : `list` of `arrays`
        Mesh point coordinates on boundaries of the domain. Structure:
        - [z_data[bnds], x_data[bnds]] for 2D
        - [z_data[bnds], x_data[bnds], y_data[bnds]] for 3D
    c : `firedrake function`
        Velocity model without absorbing layer
    c_min : `float`
        Minimum velocity value in the model without absorbing layer
    c_max : `float`
        Maximum velocity value in the model without absorbing layer
    diam_mesh : `ufl.geometry.CellDiameter`
        Mesh cell diameters
    ele_type_eik : `string`
        Finite element type for the Eikonal modeling. 'CG' or 'KMV'
    f_est : `float`
        Factor for the stabilizing term in Eikonal Eq. Default is 0.06
    funct_space_eik: `firedrake function space`
        Function space for the Eikonal modeling
    lmin : `float`
        Minimum mesh size
    lmax : `float`
        Maxmum mesh size
    mesh_original : `firedrake mesh`
        Original mesh without absorbing layer
    p_eik : `int`
        Finite element order for the Eikonal modeling
    tol : `float`
        Tolerance for searching nodes in the mesh

    Methods
    -------
    boundary_data()
        Generate the boundary data from the original domain mesh
    preamble_mesh_operations()
        Perform mesh operations previous to size an absorbing layer
    properties_eik_mesh()
        Set the properties for the mesh used to solve the Eikonal equation
    representative_mesh_dimensions()
        Get the representative mesh dimensions from original mesh.
    '''

    def __init__(self, f_est=0.06):
        '''
        Initialize the HABC_Mesh class

        Parameters
        ----------
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq. Default is 0.06

        Returns
        -------
        None
        '''

        # Factor for the stabilizing term in Eikonal equation
        self.f_est = f_est

    def representative_mesh_dimensions(self):
        '''
        Get the representative mesh dimensions from original mesh

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Mesh cell diameters
        self.diam_mesh = fire.CellDiameter(self.mesh)

        if self.dimension == 2:  # 2D
            fdim = 2**0.5

        if self.dimension == 3:  # 3D
            fdim = 3**0.5

        # Minimum and maximum mesh size for habc parameters
        diam = fire.Function(self.function_space).interpolate(self.diam_mesh)
        self.lmin = round(diam.dat.data_with_halos.min() / fdim, 6)
        self.lmax = round(diam.dat.data_with_halos.max() / fdim, 6)

        # Tolerance for searching nodes in the mesh
        self.tol = 10**(min(int(np.log10(self.lmin / 10)), -6))

    def properties_eik_mesh(self, p_usu=None, ele_type='CG'):
        '''
        Set the properties for the mesh used to solve the Eikonal equation

        Parameters
        ----------
        p_usu : `int`, optional
            Finite element order for the Eikonal equation. Default is None
        ele_type : `string`, optional
            Finite element type. 'CG' or 'KMV'. Default is 'CG'

        Returns
        -------
        None
        '''

        # Setting the properties of the mesh used to solve the Eikonal equation
        self.ele_type_eik = ele_type
        self.p_eik = self.degree if p_usu is None else p_usu
        self.funct_space_eik = fire.FunctionSpace(self.mesh,
                                                  self.ele_type_eik,
                                                  self.p_eik)

    def extract_node_positions(self, func_space):
        '''
        Extract node positions from the mesh

        Parameters
        ----------
        func_space : `firedrake function space`
            Function space to extract node positions

        Returns
        -------
        node_positions : `tuple`
            Tuple containing the node positions in the mesh.
            - (z_data, x_data) for 2D
            - (z_data, x_data, y_data) for 3D
        '''

        # Extract node positions
        z_f = fire.Function(func_space).interpolate(self.mesh_z)
        x_f = fire.Function(func_space).interpolate(self.mesh_x)
        z_data = z_f.dat.data_with_halos[:]
        x_data = x_f.dat.data_with_halos[:]
        node_positions = (z_data, x_data)

        if self.dimension == 3:  # 3D
            y_f = fire.Function(func_space).interpolate(self.mesh_y)
            y_data = y_f.dat.data_with_halos[:]
            node_positions += (y_data,)

        return node_positions

    def extract_bnd_node_positions(self, node_positions, func_space):
        '''
        Extract boundary node coordinates from node positions

        Parameters
        ----------
        node_positions : `tuple`
            Tuple containing the node positions in the mesh.
            - (z_data, x_data) for 2D
            - (z_data, x_data, y_data) for 3D

        Returns
        -------
        bnds : `tuple` of 'arrays'
            Mesh node indices on boundaries of the domain. Structure:
            - (left_boundary, right_boundary, bottom_boundary) for 2D
            - (left_boundary, right_boundary, bottom_boundary,
                left_bnd_y, right_bnd_y) for 3D
        '''

        # Extract node positions
        z_data, x_data = node_positions[0:2]

        # Boundary array
        left_boundary = np.where(x_data <= self.tol)
        right_boundary = np.where(x_data >= self.length_x - self.tol)
        bottom_boundary = np.where(z_data <= self.tol - self.length_z)
        bnds = (left_boundary, right_boundary, bottom_boundary)

        if self.dimension == 3:  # 3D
            y_data = node_positions[2]
            left_bnd_y = np.where(y_data <= self.tol)
            right_bnd_y = np.where(y_data >= self.length_y - self.tol)
            bnds += (left_bnd_y, right_bnd_y,)

        return bnds

    def boundary_data(self, typ_bnd='original'):
        '''
        Generate the boundary data from the original domain mesh.

        Parameters
        ----------
        typ_bnd : str, optional
            Type of boundary data to extract. Options: 'original' for original
            domain or 'eikonal' for Eikonal analysis. Default is 'original'

        Returns
        -------
        bnds : `list` of 'arrays'
            Mesh point indices on boundaries of the domain. Structure:
            - [left_boundary, right_boundary, bottom_boundary] for 2D
            - [left_boundary, right_boundary, bottom_boundary,
                left_bnd_y, right_bnd_y] for 3D
        node_positions : `list`, optional
            List of node positions for Eikonal analysis. Structure:
            - [z_data, x_data] for 2D
            - [z_data, x_data, y_data] for 3D
        '''

        if typ_bnd == 'original':
            func_space = self.function_space
        elif typ_bnd == 'eikonal':
            func_space = self.funct_space_eik
        else:
            value_parameter_error('typ_bnd', typ_bnd, ['original', 'eikonal'])

        # Extract node positions
        node_positions = self.extract_node_positions(func_space)

        # Extract boundary node coordinates
        bnds = self.extract_bnd_node_positions(node_positions, func_space)

        z_data, x_data = node_positions[0:2]
        if self.dimension == 3:  # 3D
            y_data = node_positions[2]
        if typ_bnd == 'original':
            # ToDo: Check usage of these atributes
            self.bnds = np.unique(np.concatenate([idxs for idx_list in bnds
                                                  for idxs in idx_list]))
            self.bnd_nodes = [z_data[self.bnds], x_data[self.bnds]]
            if self.dimension == 3:  # 3D
                self.bnd_nodes.append(y_data[self.bnds])

    def preamble_mesh_operations(self):
        '''
        Perform mesh operations previous to size an absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nCreating Mesh and Initial Velocity Model")

        # Get mesh parameters from original mesh
        self.representative_mesh_dimensions()

        # Save a copy of the original mesh
        self.mesh_original = self.mesh
        mesh_orig = fire.VTKFile(self.path_save + "preamble/mesh_orig.pvd")
        mesh_orig.write(self.mesh_original)

        # Velocity profile model
        self.c = fire.Function(self.function_space, name='c_orig [km/s])')
        self.c.interpolate(self.initial_velocity_model)

        # Get extreme values of the velocity model
        self.c_min = self.initial_velocity_model.dat.data_with_halos.min()
        self.c_max = self.initial_velocity_model.dat.data_with_halos.max()

        # Save initial velocity model
        vel_c = fire.VTKFile(self.path_save + "preamble/c_vel.pvd")
        vel_c.write(self.c)

        # Mesh properties for Eikonal
        self.properties_eik_mesh(p_usu=self.abc_deg_eikonal)

        # Generating boundary data from the original domain mesh
        # self.boundary_data()
