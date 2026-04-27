import firedrake as fire
from firedrake.__future__ import interpolate
import numpy as np
fire.interpolate = interpolate
from spyro.utils.error_management import value_parameter_error
from spyro.utils.eval_functions_to_ufl import generate_ufl_functions


class MeshOps():
    """Class for general mesh operations for domains w/o an absorbing layer.

    Attributes
    ----------
    comm : `object`
        An object representing the communication interface
        for parallel processing. Default is None
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D
    domain_dim : `tuple`
        Domain dimensions: (length_z, length_x) for 2D
        or (length_z, length_x, length_y) for 3D
    func_space_type, `str`
        Type of function space for the state variable.
        Options: 'scalar' or 'vector'. Default is None
    quadrilateral : bool
        Flag to indicate whether to use quadrilateral/hexahedral elements

    bnds : 'array'
        Mesh node indices on boundaries of the original domain
    bnd_nodes : `tuple`
        Mesh node coordinates on boundaries of the origianl domain.
        - (z_data[bnds], x_data[bnds]) for 2D
        - (z_data[bnds], x_data[bnds], y_data[bnds]) for 3D
    c : `firedrake function`
        Velocity model without absorbing layer
    c_bnd_min : `float`
        Minimum velocity value on the boundary of the original domain
    c_bnd_max : `float`
        Maximum velocity value on the boundary of the original domain
    c_min : `float`
        Minimum velocity value in the model without absorbing layer
    c_max : `float`
        Maximum velocity value in the model without absorbing layer
    domain_dim : `tuple`
        Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
    ele_type_c0 : `string`
        Finite element type for the velocity model without absorbing layer
    ele_type_eik : `string`
        Finite element type for the Eikonal modeling. 'CG' or 'KMV'
    f_est : `float`
        Factor for the stabilizing term in Eikonal Eq. Default is 0.03
    funct_space_eik: `firedrake function space`
        Function space for the Eikonal modeling
    mesh_original : `firedrake mesh`
        Original mesh without absorbing layer
    p_c0 : `int`
        Finite element order for the velocity model without absorbing layer
    p_eik : `int`
        Finite element order for the Eikonal modeling

    Methods
    -------
    _set_spatial_coordinates()
        Set the coordinates of a mesh.
    extract_extreme_coordinates()
        Extract the minimum and maximum coordinates from a mesh.
    extract_node_positions()
        Extract the node positions from the mesh and return as a tuple of arrays.
    representative_mesh_dimensions()
        Get the representative mesh dimensions from a mesh.



    clipping_coordinates_lay_field()
        Generate a field with clipping coordinates to the original boundary
    extend_velocity_profile()
        Extend the velocity profile inside the absorbing layer
    extract_bnd_node_indices()
        Extract boundary node indices on boundaries of the domain
        excluding the free surface at the top boundary
    layer_boundary_data()
        Generate the boundary data from the domain with the absorbing layer
    layer_mask_field()
        Generate a mask for the absorbing layer
    original_boundary_data()
        Generate the boundary data from the original domain mesh
    point_cloud_field()
        Create a field on a point cloud from a parent mesh and field
    rectangular_mesh_habc()
        Generate a rectangular mesh with an absorbing layer
    """

    def __init__(self, domain_dim, dimension=2, quadrilateral=False,
                 func_space_type=None, comm=None):
        """Initialize the MeshOps class.

        Parameters
        ----------
        domain_dim : `tuple`
            Domain dimensions: (length_z, length_x) for 2D
            or (length_z, length_x, length_y) for 3D
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D
        quadrilateral : bool, optional
            Flag to indicate whether to use quadrilateral/hexahedral elements
        func_space_type, `str`, optional
            Type of function space for the state variable.
            Options: 'scalar' or 'vector'. Default is None
        comm : `object`, optional
            An object representing the communication interface
            for parallel processing. Default is None

        Returns
        -------
        None
        """

        # Original domain dimensions
        self.domain_dim = domain_dim

        # Model dimension
        self.dimension = dimension

        # Quadrilateral/hexahedral elements
        self.quadrilateral = quadrilateral

        # Type of function space
        self.func_space_type = func_space_type

        # Communicator MPI
        self.comm = comm

    def _set_spatial_coordinates(self, mesh):
        """Set the coordinates of a mesh.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh

        Returns
        -------
        mesh_z : `ufl.geometry.SpatialCoordinate`
            Symbolic coordinate z of the mesh object
        mesh_x: `ufl.geometry.SpatialCoordinate`
            Symbolic coordinate x of the mesh object
        mesh_y: `ufl.geometry.SpatialCoordinate`
            Symbolic coordinate y of the mesh object
        """
        if self.dimension == 2:
            mesh_z, mesh_x = fire.SpatialCoordinate(mesh)
            return mesh_z, mesh_x

        elif self.dimension == 3:
            mesh_z, mesh_x, mesh_y = fire.SpatialCoordinate(mesh)
            return mesh_z, mesh_x, mesh_y

    def representative_mesh_dimensions(self, mesh, function_space):
        """Get the representative mesh dimensions from a mesh.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh
        function_space : `FiredrakeFunctionSpace`
            Function space for the projection of the mesh cell diameters

        Returns
        -------
        alpha : `float`
            Ratio between the representative mesh dimensions
        diam_mesh : `ufl.geometry.CellDiameter`
            Mesh cell diameters
        lmin : `float`
            Minimum mesh size
        lmax : `float`
            Maxmum mesh size
        tol : `float`
            Tolerance for searching nodes in the mesh
        """

        # Mesh cell diameters
        diam_mesh = fire.CellDiameter(mesh)

        if self.dimension == 2:  # 2D
            fdim = 2**0.5

        if self.dimension == 3:  # 3D
            fdim = 3**0.5

        # Minimum and maximum mesh size for habc parameters
        diam = fire.assemble(fire.interpolate(diam_mesh,
                                              function_space))
        lmin = round(diam.dat.data_with_halos.min() / fdim, 6)
        lmax = round(diam.dat.data_with_halos.max() / fdim, 6)

        # Ratio between the representative mesh dimensions
        alpha = lmax / lmin

        # Tolerance for searching nodes in the mesh
        tol = 10**(min(int(np.log10(lmin / 10)), -6))

        return (diam, lmin, lmax, alpha, tol)

    @staticmethod
    def extract_extreme_coordinates(mesh):
        """Extract the minimum and maximum coordinates from a mesh.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh

        Returns
        -------
        min_coordinates : `array`
            Array containing the minimum coordinates in each dimension (z, x, y)
        max_coordinates : `array`
            Array containing the maximum coordinates in each dimension (z, x, y)
        """

        coords = mesh.coordinates.dat.data_with_halos
        min_coordinates = np.min(coords, axis=0)
        max_coordinates = np.max(coords, axis=0)

        return min_coordinates, max_coordinates

    def extract_node_positions(self, mesh, function_space, output_type="tuple"):
        """Extract the node positions from the mesh.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh
        function_space : `FiredrakeFunctionSpace`
            Function space to extract node positions
        output_type : `str`, optional
            Output type for node positions. Options are "tuple" or "array".
            Default is "tuple".

        Returns
        -------
        node_positions : `tuple` or `array` 
            Node positions of the mesh
            If output_type is "tuple":
                - (z_data, x_data) for 2D
                - (z_data, x_data, y_data) for 3D
            If output_type is "array":
                - array of shape (num_nodes, 2) and coordinates (z, x) for 2D
                - array of shape (num_nodes, 3) and coordinates (z, x, y) for 3D
        """

        # Interpolate the coordinates according to the function space
        coords = []
        coord_expression = ['z', 'x', 'y']
        for i in range(self.dimension):
            ufl_input = generate_ufl_functions(mesh, coord_expression[i],
                                               self.dimension)

            if self.func_space_type == 'scalar':
                V = function_space

            if self.func_space_type == 'vector':
                V = function_space.sub(i)

            coords.append(fire.assemble(fire.interpolate(ufl_input, V)))

        # Get the node positions
        if output_type == "tuple":
            z_data = coords[0].dat.data_with_halos[:]
            x_data = coords[1].dat.data_with_halos[:]
            node_positions = (z_data, x_data)
            if self.dimension == 3:  # 3D
                y_data = coords[2].dat.data_with_halos[:]
                node_positions += (y_data,)

        elif output_type == "array":
            node_positions = np.column_stack([comp.dat.data_with_halos[:]
                                              for comp in coords])

        else:
            value_parameter_error('output_type', output_type, ["tuple", "array"])
        del coords

        # TODO: either put wave's similar method here or just remove the one from wave
        # to decouple it from wave altogether.
        return node_positions

    def mapping_boundary_ids(self, mesh, function_space, boundaries,
                             box_domain=True, get_boundary_node_ids=True):
        """Map the boundaries of the a mesh.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh
        function_space : `Firedrake.FunctionSpace`
            Function space for the state variable
        boundaries : `tuple`
            Tuple containing the boundary boolean labels for applying absorbing BCs.
            - (absorb_top, absorb_bottom, absorb_right, absorb_left) for 2D
            - (absorb_top, absorb_bottom, absorb_right,
                absorb_left, absorb_front, absorb_back) for 3D
        box_domain : `bool`, optional
            Flag to indicate whether the domain is a box (Rectangle or Parallelepiped).
            Default is True
        get_boundary_node_ids : `bool`, optional
            if True, return the boundary node ids according to the boundary map.
            Default is True

        Returns
        -------
        boundary_ids_map: `dict`
            Mapping of boundary IDs for applying absorbing boundary conditions
        boundary_nodes_ids: `dict`
            IDs of the boundary nodes according to the function space provide and their
            status to apply absorbing boundary conditions. Structure is a `tuple`:
            boundary_nodes_ids[key_bnd] = (bnd_node_ids, status) where key_bnd is 'Imin'
            or 'Imax' with I = Z, X or Y, and status is a boolean from boundary_ids_map. 
        """

        if box_domain:

            # Simulating a Dirichlet BC at each boundary
            if self.func_space_type == "scalar":
                bc_val = 0.
            else:
                bc_val = fire.as_vector((0.,) * self.dimension)

            num_boundaries = 4
            min_coordinates, max_coordinates = self.extract_extreme_coordinates(mesh)
            min_z, min_x = min_coordinates[:2]
            max_z, max_x = max_coordinates[:2]
            absorb_top, absorb_bottom, absorb_right, absorb_left = boundaries[:4]
            if self.dimension == 3:
                num_boundaries += 2
                min_y = min_coordinates[2]
                max_y = max_coordinates[2]
                absorb_front, absorb_back = boundaries[4:]

            # Node coordinates
            node_positions = self.extract_node_positions(mesh, function_space)

            # Verify numerical ids
            exterior_markers = set(mesh.exterior_facets.unique_markers)
            # print("Available boundary markers:", exterior_markers)

            # Boundary nodes indices
            if len(exterior_markers) == 0:
                boundary_ids_map = {idx_bdn: None for idx_bdn
                                    in range(1, num_boundaries + 1)}
                return boundary_ids_map

            boundary_ids_map = {}
            boundary_nodes_ids = {}
            for idx_bdn in range(1, num_boundaries + 1):

                if self.dimension == 3 and self.quadrilateral:
                    idx_bdn = "bottom" if idx_bdn == 5 else idx_bdn
                    idx_bdn = "top" if idx_bdn == 6 else idx_bdn

                # Applying a dummy Dirichlet BC
                bnd_node_ids = fire.DirichletBC(function_space, bc_val, idx_bdn).nodes

                if len(bnd_node_ids) == 0:
                    # For DG spaces, DirichletBC doesn't work
                    boundary_ids_map[idx_bdn] = None
                    continue

                idx_test = np.linspace(0, len(bnd_node_ids) - 1, 10, dtype=int)
                sample_nodes = bnd_node_ids[idx_test]

                # Data for checking
                z_data = node_positions[0][sample_nodes]
                x_data = node_positions[1][sample_nodes]
                if np.allclose(z_data, min_z):
                    boundary_ids_map[idx_bdn] = absorb_bottom  # Bottom boundary
                    key_bnd = "Zmin"
                elif np.allclose(z_data, max_z):
                    boundary_ids_map[idx_bdn] = absorb_top  # Top boundary
                    key_bnd = "Zmax"
                elif np.allclose(x_data, min_x):
                    boundary_ids_map[idx_bdn] = absorb_left  # Left boundary
                    key_bnd = "Xmin"
                elif np.allclose(x_data, max_x):
                    boundary_ids_map[idx_bdn] = absorb_right  # Right boundary
                    key_bnd = "Xmax"

                if self.dimension == 3:
                    y_data = node_positions[2][sample_nodes]
                    if np.allclose(y_data, min_y):
                        boundary_ids_map[idx_bdn] = absorb_front  # Front boundary
                        key_bnd = "Ymin"
                    elif np.allclose(y_data, max_y):
                        boundary_ids_map[idx_bdn] = absorb_back  # Back boundary
                        key_bnd = "Ymax"

                # Boundary node coordinates
                if get_boundary_node_ids:
                    boundary_nodes_ids[key_bnd] = (bnd_node_ids,
                                                   boundary_ids_map[idx_bdn])
                del bnd_node_ids

            if get_boundary_node_ids:
                return boundary_ids_map, boundary_nodes_ids

            else:
                return boundary_ids_map
        else:
            raise NotImplementedError("Mapping of boundary ids for other type of "
                                      "domains is not implemented yet. Only box "
                                      "domains are supported. The numbering the "
                                      "future boundary ids must start at 7.")

    def extract_bnd_node_indices(self, mesh, function_space, mesh_parameters):
        """Extract boundary node indices excluding the free surface at the top.

        Parameters
        ----------
        node_positions : `tuple`
            Tuple containing the node positions in the mesh.
            - (z_data, x_data) for 2D
            - (z_data, x_data, y_data) for 3D
        function_space : `FiredrakeFunctionSpace`
            Function space to extract node positions.
        mesh_parameters : `MeshParameters`
            Mesh parameters for the problem.
            - length_z : float
                Mesh length in the z-direction.
            - length_x : float
                Mesh length in the x-direction.
            - length_y : float
                Mesh length in the y-direction (for 3D meshes).
            - tol : `float`
                Tolerance for searching nodes in the mesh.

        Returns
        -------
        bnds : `tuple` of 'arrays'
            Mesh node indices on boundaries of the domain.
            - (left_boundary, right_boundary, bottom_boundary) for 2D
            - (left_boundary, right_boundary, bottom_boundary,
                left_bnd_y, right_bnd_y) for 3D
        """

        node_positions = self.extract_node_positions(mesh, function_space)

        # Extract node positions
        z_data, x_data = node_positions[0:2]

        # Boundary array
        left_boundary = np.where(x_data <= mesh_parameters.tol)
        right_boundary = np.where(x_data >= mesh_parameters.length_x
                                  - mesh_parameters.tol)
        bottom_boundary = np.where(z_data <= mesh_parameters.tol
                                   - mesh_parameters.length_z)
        bnds = (left_boundary, right_boundary, bottom_boundary)

        if self.dimension == 3:  # 3D
            y_data = node_positions[2]
            left_bnd_y = np.where(y_data <= mesh_parameters.tol)
            right_bnd_y = np.where(y_data >= mesh_parameters.length_y
                                   - mesh_parameters.tol)
            bnds += (left_bnd_y, right_bnd_y,)

        return bnds


    # def layer_boundary_data(self, V):
    #     """
    #     Generate the boundary data from the domain with the absorbing layer

    #     Parameters
    #     ----------
    #     V : `firedrake function space`
    #         Function space for the boundary of the domain with absorbing layer

    #     Returns
    #     -------
    #     bnd_nfs : 'array'
    #         Mesh node indices on non-free surfaces
    #     bnd_nodes_nfs : `tuple`
    #         Mesh node coordinates on non-free surfaces.
    #         - (z_data[nfs_idx], x_data[nfs_idx]) for 2D
    #         - (z_data[nfs_idx], x_data[nfs_idx], y_data[nfs_idx]) for 3D
    #     """

    #     # Boundary nodes indices
    #     bnd_nod = fire.DirichletBC(V, 0., "on_boundary").nodes

    #     # Extract node positions
    #     node_positions = self.extract_node_positions(V)

    #     # Boundary node coordinates
    #     z_f, x_f = node_positions[:2]
    #     bnd_z = z_f[bnd_nod]
    #     bnd_x = x_f[bnd_nod]

    #     # Identify non-free surfaces (remain unchanged)
    #     no_free_surf = ~(abs(bnd_z) <= self.tol)

    #     bnd_nodes_nfs = (bnd_z[no_free_surf], bnd_x[no_free_surf])
    #     if self.dimension == 3:  # 3D
    #         y_f = node_positions[2]
    #         bnd_y = y_f[bnd_nod]
    #         bnd_nodes_nfs += (bnd_y[no_free_surf],)

    #     # Boundary node indices on non-free surfaces
    #     bnd_nfs = bnd_nod[no_free_surf]

    #     return bnd_nfs, bnd_nodes_nfs
