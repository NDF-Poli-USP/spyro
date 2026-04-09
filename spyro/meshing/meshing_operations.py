import firedrake as fire
from firedrake.__future__ import interpolate
import numpy as np
from spyro.utils.error_management import value_parameter_error
fire.interpolate = interpolate


class MeshOps():
    """
    Class for general mesh operations for domains w/o an absorbing layer.

    Attributes
    ----------
    alpha : `float`
        Ratio between the representative mesh dimensions
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
    comm : object
        An object representing the communication interface
        for parallel processing. Default is None
    diam_mesh : `ufl.geometry.CellDiameter`
        Mesh cell diameters
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D
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
    lmin : `float`
        Minimum mesh size
    lmax : `float`
        Maxmum mesh size
    mesh_original : `firedrake mesh`
        Original mesh without absorbing layer
    p_c0 : `int`
        Finite element order for the velocity model without absorbing layer
    p_eik : `int`
        Finite element order for the Eikonal modeling
    quadrilateral : bool
        Flag to indicate whether to use quadrilateral/hexahedral elements
    tol : `float`
        Tolerance for searching nodes in the mesh

    Methods
    -------
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
    representative_mesh_dimensions()
        Get the representative mesh dimensions from original mesh
    """

    def __init__(self, domain_dim, dimension=2, quadrilateral=False, comm=None):
        """
        Initialize the HABC_Mesh class.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D
        quadrilateral : bool, optional
            Flag to indicate whether to use quadrilateral/hexahedral elements
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

        # Communicator MPI
        self.comm = comm

    def _set_spatial_coordinates(self, mesh):
        """
        Set the coordinates of a mesh.

        Parameters
        ----------
        mesh : `FiredrakeMesh`
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
            z, x = fire.SpatialCoordinate(mesh)
            mesh_z = z
            mesh_x = x
            return mesh_z, mesh_x

        elif self.dimension == 3:
            z, x, y = fire.SpatialCoordinate(mesh)
            mesh_z = z
            mesh_x = x
            mesh_y = y
            return mesh_z, mesh_x, mesh_y

    def representative_mesh_dimensions(self, mesh, function_space):
        """
        Get the representative mesh dimensions from a mesh.

        Parameters
        ----------
        mesh : `FiredrakeMesh`
            Current mesh
        func_space : `FiredrakeFunctionSpace`
            Function space to extract node positions

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

    def extract_node_positions(self, func_space):
        """
        Extract node positions from a mesh

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
        """

        # Extract node positions
        z_f = fire.assemble(fire.interpolate(self.mesh_z, func_space))
        x_f = fire.assemble(fire.interpolate(self.mesh_x, func_space))
        z_data = z_f.dat.data_with_halos[:]
        x_data = x_f.dat.data_with_halos[:]
        node_positions = (z_data, x_data)

        if self.dimension == 3:  # 3D
            y_f = fire.assemble(fire.interpolate(self.mesh_y, func_space))
            y_data = y_f.dat.data_with_halos[:]
            node_positions += (y_data,)

        return node_positions

    def extract_bnd_node_indices(self, node_positions, func_space):
        """
        Extract boundary node indices on boundaries of the domain
        excluding the free surface at the top boundary

        Parameters
        ----------
        node_positions : `tuple`
            Tuple containing the node positions in the mesh.
            - (z_data, x_data) for 2D
            - (z_data, x_data, y_data) for 3D
        func_space : `firedrake function space`
            Function space to extract node positions

        Returns
        -------
        bnds : `tuple` of 'arrays'
            Mesh node indices on boundaries of the domain.
            - (left_boundary, right_boundary, bottom_boundary) for 2D
            - (left_boundary, right_boundary, bottom_boundary,
                left_bnd_y, right_bnd_y) for 3D
        """

        # Extract node positions
        z_data, x_data = node_positions[0:2]

        # Boundary array
        left_boundary = np.where(x_data <= self.tol)
        right_boundary = np.where(x_data >= self.mesh_parameters.length_x
                                  - self.tol)
        bottom_boundary = np.where(z_data <= self.tol
                                   - self.mesh_parameters.length_z)
        bnds = (left_boundary, right_boundary, bottom_boundary)

        if self.dimension == 3:  # 3D
            y_data = node_positions[2]
            left_bnd_y = np.where(y_data <= self.tol)
            right_bnd_y = np.where(y_data >= self.mesh_parameters.length_y
                                   - self.tol)
            bnds += (left_bnd_y, right_bnd_y,)

        return bnds

    def original_boundary_data(self):
        """
        Generate the boundary data from the original domain mesh

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Extract node positions
        node_positions = self.extract_node_positions(self.function_space)

        # Extract boundary node indices
        bnds = self.extract_bnd_node_indices(node_positions,
                                             self.function_space)
        self.bnds = np.unique(np.concatenate([idxs for idx_list in bnds
                                              for idxs in idx_list]))

        # Extract boundary node positions
        z_data, x_data = node_positions[0:2]
        self.bnd_nodes = (z_data[self.bnds], x_data[self.bnds])
        if self.dimension == 3:  # 3D
            y_data = node_positions[2]
            self.bnd_nodes += (y_data[self.bnds],)

        # Get extreme values of the velocity model on the boundary
        mask_boundary = np.isin(
            np.asarray(self.bnd_nodes).T,
            self.mesh_original.coordinates.dat.data_with_halos).all(axis=1)
        vel_on_boundary = self.point_cloud_field(
            self.mesh_original, np.asarray(self.bnd_nodes).T[mask_boundary],
            self.initial_velocity_model).dat.data_with_halos[:]
        self.c_bnd_min = vel_on_boundary[vel_on_boundary > 0.].min()
        self.c_bnd_max = vel_on_boundary[vel_on_boundary > 0.].max()

        # Print on screen
        cbnd_str = "Boundary Velocity Range (km/s): {:.3f} - {:.3f}"
        print(cbnd_str.format(self.c_bnd_min, self.c_bnd_max), flush=True)

    def rectangular_mesh_habc(self, dom_lay, pad_len):
        """
        Generate a rectangular mesh with an absorbing layer

        Parameters
        ----------
        dom_lay : `tuple`
            Domain dimensions with layer including truncation by free surface.
            - 2D : (Lx + 2 * pad_len, Lz + pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + pad_len, Ly + 2 * pad_len)
        pad_len : `float`
            Size of the absorbing layer

        Returns
        -------
        mesh_habc : `firedrake mesh`
            Rectangular mesh with an absorbing layer.
        """

        # Domain dimensions
        Lx, Lz = self.domain_dim[:2]

        # Number of elements
        n_pad = round(pad_len / self.lmin)  # Elements in the layer
        nz = int(round(Lz / self.lmin)) + int(n_pad)
        nx = int(round(Lx / self.lmin)) + int(2 * n_pad)

        # New geometry with layer
        Lx_habc, Lz_habc = dom_lay[:2]

        # Creating the rectangular mesh with layer
        q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
        if self.dimension == 2:  # 2D
            mesh_habc = fire.RectangleMesh(nz, nx, Lz_habc, Lx_habc,
                                           distribution_parameters=q,
                                           quadrilateral=self.quadrilateral,
                                           comm=self.comm.comm)
            typ_ele_str = "Area Elements"

        if self.dimension == 3:  # 3D

            # Number of elements
            Ly = self.domain_dim[2]
            ny = int(round(Ly / self.lmin)) + int(2 * n_pad)

            # New geometry with layer
            Ly_habc = dom_lay[2]

            # Mesh
            if self.quadrilateral:
                quad_habc = fire.RectangleMesh(
                    nz, nx, Lz_habc, Lx_habc, distribution_parameters=q,
                    quadrilateral=self.quadrilateral, comm=self.comm.comm)
                # fire.VTKFile("output/quad_habc.pvd").write(quad_habc)

                mesh_habc = fire.ExtrudedMesh(quad_habc, ny,
                                              layer_height=Ly_habc / ny)
                # fire.VTKFile("output/extr_habc.pvd").write(mesh_habc)
            else:
                mesh_habc = fire.BoxMesh(
                    nz, nx, ny, Lz_habc, Lx_habc, Ly_habc,
                    distribution_parameters=q, comm=self.comm.comm)
            typ_ele_str = "Volume Elements"

            # Adjusting coordinates
            mesh_habc.coordinates.dat.data_with_halos[:, 2] -= pad_len
            min_y = mesh_habc.coordinates.dat.data_with_halos[:, 2].min()
            if abs(min_y / pad_len) != 1.:  # Forcing node at (0,0,0)
                err_y = (1. - abs(min_y / pad_len)) * pad_len
                err_y *= -np.sign(err_y)
                mesh_habc.coordinates.dat.data_with_halos[:, 2] += err_y

        # Adjusting coordinates
        mesh_habc.coordinates.dat.data_with_halos[:, 0] *= -1.0
        mesh_habc.coordinates.dat.data_with_halos[:, 1] -= pad_len
        min_x = mesh_habc.coordinates.dat.data_with_halos[:, 1].min()
        if abs(min_x / pad_len) != 1.:  # Forcing node at (0,0)
            err_x = (1. - abs(min_x / pad_len)) * pad_len
            err_x *= -np.sign(err_x)
            mesh_habc.coordinates.dat.data_with_halos[:, 1] += err_x

        # Mesh data
        print(f"Mesh Created with {mesh_habc.num_vertices()} Nodes "
              f"and {mesh_habc.num_cells()} " + typ_ele_str, flush=True)

        print("Extended Rectangular Mesh Generated Successfully", flush=True)

        return mesh_habc

    def layer_mask_field(self, coords, V, damp_par=None,
                         type_marker='damping', name_mask=None):
        """
        Generate a mask for the absorbing layer. The mask is defined
        for conditional expressions to identify the domain of the layer
        (option: 'mask') or the reference to the original boundary
        (option: 'damping') used to compute the damping profile.

        Parameters
        ----------
        coords : 'ufl.geometry.SpatialCoordinate'
            Domain Coordinates including the absorbing layer
        V : `firedrake function space`
            Function space for the mask field
        damp_par : `tuple`, optional
            Damping parameters for the absorbing layer.
            Structure: (pad_len, eta_crt, aq, bq)
            - pad_len : `float`
                Size of the absorbing layer
            - eta_crt : `float`
                Critical damping coefficient (1/s)
            - aq : `float`
                Coefficient for quadratic term in the damping function
            - bq : `float`
                Coefficient bq for linear term in the damping function
        type_marker : `string`, optional
            Type of marker. Default is 'mask'.
            - 'damping' : Get the reference distance to the original boundary
            - 'mask' : Define a mask to filter the layer boundary domain
        name_mask : `string`, optional
            Name for the mask field. Default is None

        Returns
        -------
        layer_mask : `firedrake function`
            Mask for the absorbing layer
            - 'damping' : `ufl.conditional.Conditional`
                Reference distance to the original boundary
            - 'mask' : `ufl.algebra.Division`
                Conditional expression to identify the layer domain
        """

        # Domain dimensions
        Lx, Lz = self.domain_dim[:2]

        # Domain coordinates
        z, x = coords[0], coords[1]

        # Conditional value
        val_condz = (z + Lz)**2 if type_marker == 'damping' else 1.
        val_condx1 = x**2 if type_marker == 'damping' else 1.
        val_condx2 = (x - Lx)**2 if type_marker == 'damping' else 1.

        # Conditional expressions for the mask
        z_pd = fire.conditional(z < -Lz, val_condz, 0.)
        x_pd = fire.conditional(x < 0., val_condx1, 0.) + \
            fire.conditional(x > Lx, val_condx2, 0.)
        ref = z_pd + x_pd

        if self.dimension == 3:  # 3D

            # 3D dimension
            Ly = self.domain_dim[2]
            y = coords[2]

            # Conditional value
            val_condy1 = y**2 if type_marker == 'damping' else 1.
            val_condy2 = (y - Ly)**2 if type_marker == 'damping' else 1.

            # Conditional expressions for the mask
            y_pd = fire.conditional(y < 0., val_condy1, 0.) + \
                fire.conditional(y > Ly, val_condy2, 0.)
            ref += y_pd

        # Final value for the mask
        if type_marker == 'damping':

            if damp_par is None:
                raise ValueError("Damping parameters must be provided "
                                 "when 'type_marker' is 'damping'.")

            # Damping parameters
            pad_len, eta_crt, aq, bq = damp_par

            if pad_len <= 0:
                raise ValueError(f"Invalid value for 'pad_len': {pad_len}. "
                                 "'pad_len' must be greater than zero "
                                 "when 'type_marker' is 'damping'.")
            if eta_crt <= 0:
                raise ValueError(f"Invalid value for 'eta_crt': {eta_crt}. "
                                 "'eta_crt' must be greater than zero "
                                 "when 'type_marker' is 'damping'.")

            # Reference distance to the original boundary
            ref = fire.sqrt(ref) / fire.Constant(pad_len)

            # Quadratic damping profile
            if bq == 0.:
                ref = fire.Constant(eta_crt) * fire.Constant(aq) * ref**2
            else:
                ref = fire.Constant(eta_crt) * (fire.Constant(aq) * ref**2
                                                + fire.Constant(bq) * ref)

        elif type_marker == 'mask':
            # Mask filter for layer boundary domain
            ref = fire.conditional(ref > 0, 1., 0.)

        else:
            value_parameter_error('type_marker', type_marker,
                                  ['damping', 'mask'])

        layer_mask = fire.Function(V, name=name_mask)
        layer_mask.assign(fire.assemble(fire.interpolate(ref, V)))

        return layer_mask

    def clipping_coordinates_lay_field(self, V):
        """
        Generate a field with clipping coordinates to the original boundary

        Parameters
        ----------
        V : `firedrake function space`
            Function space for the mask field

        Returns
        -------
        lay_field : `firedrake function`
            Field with clipped coordinates only in the absorbing layer
        layer_mask : `firedrake function`
            Mask for the absorbing layer
        """

        print("Clipping Coordinates Inside Layer", flush=True)

        # Domain dimensions
        Lx, Lz = self.domain_dim[:2]

        # Vectorial space for auxiliar field of clipped coordinates
        if self.quadrilateral:
            base_mesh = self.mesh._base_mesh
            base_cell = base_mesh.ufl_cell()
            element_zx = fire.FiniteElement("DQ", base_cell, 0,
                                            variant="spectral")
            element_y = fire.FiniteElement("DG", fire.interval, 0,
                                           variant="spectral")
            tensor_element = fire.TensorProductElement(element_zx, element_y)
            W_sp = fire.VectorFunctionSpace(self.mesh, tensor_element)
        else:
            W_sp = fire.VectorFunctionSpace(self.mesh,
                                            self.ele_type_c0,
                                            self.p_c0)

        # Mesh coordinates
        coords = fire.SpatialCoordinate(self.mesh)

        # Clipping coordinates
        lay_field = fire.Function(W_sp)
        lay_field.assign(fire.assemble(fire.interpolate(coords, W_sp)))
        lay_arr = lay_field.dat.data_with_halos[:]
        lay_arr[:, 0] = np.clip(lay_arr[:, 0], -Lz, 0.)
        lay_arr[:, 1] = np.clip(lay_arr[:, 1], 0., Lx)

        if self.dimension == 3:  # 3D

            # 3D dimension
            Ly = self.domain_dim[2]

            # Clipping coordinates
            lay_arr[:, 2] = np.clip(lay_arr[:, 2], 0., Ly)

        # Mask function to identify the absorbing layer domain
        layer_mask = self.layer_mask_field(coords, V, type_marker='mask')

        # Field with clipped coordinates only in the absorbing layer
        lay_field.assign(fire.assemble(fire.interpolate(
            lay_field * layer_mask, W_sp)))

        return lay_field, layer_mask

    def point_cloud_field(self, parent_mesh, pts_cloud, parent_field):
        """
        Create a field on a point cloud from a parent mesh and field

        Parameters
        ----------
        parent_mesh : `firedrake mesh`
            Parent mesh containing the original field
        pts_cloud : `array`
            Array of shape (num_pts, dim) containing the coordinates
            of the point cloud
        parent_field : `firedrake function`
            Parent field defined on the parent mesh

        Returns
        -------
        cloud_field : `firedrake function`
            Field defined on the point cloud
        """

        # Creating a point cloud field from the parent mesh
        pts_mesh = fire.VertexOnlyMesh(
            parent_mesh, pts_cloud, reorder=True, tolerance=self.tol,
            missing_points_behaviour='error', redundant=False)
        del pts_cloud

        # Cloud field
        V0 = fire.FunctionSpace(pts_mesh, "DG", 0)
        f_pts = fire.assemble(fire.interpolate(parent_field, V0))

        # Ensuring correct assemble
        V1 = fire.FunctionSpace(pts_mesh.input_ordering, "DG", 0)
        del pts_mesh
        cloud_field = fire.Function(V1)
        cloud_field.assign(fire.assemble(fire.interpolate(f_pts, V1)))
        del f_pts

        return cloud_field

    def extend_velocity_profile(self, lay_field, layer_mask,
                                method='point_cloud'):
        """
        Extend the velocity profile inside the absorbing layer

        Parameters
        ----------
        lay_field : `firedrake function`
            Field with clipped coordinates only in the absorbing layer
        layer_mask : `firedrake function`
            Mask for the absorbing layer
        method : `str`, optional
            Method to extend the velocity profile. Options:
            'point_cloud' or 'nearest_point'. Default is 'point_cloud'

        Returns
        -------
        None
        """

        print("Extending Profile Inside Layer", flush=True)

        # Extracting the nodes from the layer field
        lay_nodes = lay_field.dat.data_with_halos[:]

        # Nodes to extend the velocity model
        ind_nodes = np.where(layer_mask.dat.data_with_halos)[0]
        pts_to_extend = lay_nodes[ind_nodes]

        if method == 'point_cloud':

            print("Using Cloud Points Method to Extend Velocity Profile",
                  flush=True)

            # Set the velocity of the nearest point on the original boundary
            vel_to_extend = self.point_cloud_field(
                self.mesh_original, pts_to_extend,
                self.initial_velocity_model).dat.data_with_halos[:]

        elif method == 'nearest_point':

            print("Using Nearest Point Method to Extend Velocity Profile",
                  flush=True)

            # Set the velocity of the nearest point on the original boundary
            vel_to_extend = self.initial_velocity_model.at(pts_to_extend,
                                                           dont_raise=True)
            del pts_to_extend

        else:
            value_parameter_error('method', method,
                                  ['point_cloud', 'nearest_point'])

        # Velocity profile inside the layer
        lay_field.dat.data_with_halos[ind_nodes, 0] = vel_to_extend
        del vel_to_extend, ind_nodes

    def layer_boundary_data(self, V):
        """
        Generate the boundary data from the domain with the absorbing layer

        Parameters
        ----------
        V : `firedrake function space`
            Function space for the boundary of the domain with absorbing layer

        Returns
        -------
        bnd_nfs : 'array'
            Mesh node indices on non-free surfaces
        bnd_nodes_nfs : `tuple`
            Mesh node coordinates on non-free surfaces.
            - (z_data[nfs_idx], x_data[nfs_idx]) for 2D
            - (z_data[nfs_idx], x_data[nfs_idx], y_data[nfs_idx]) for 3D
        """

        # Boundary nodes indices
        bnd_nod = fire.DirichletBC(V, 0., "on_boundary").nodes

        # Extract node positions
        node_positions = self.extract_node_positions(V)

        # Boundary node coordinates
        z_f, x_f = node_positions[:2]
        bnd_z = z_f[bnd_nod]
        bnd_x = x_f[bnd_nod]

        # Identify non-free surfaces (remain unchanged)
        no_free_surf = ~(abs(bnd_z) <= self.tol)

        bnd_nodes_nfs = (bnd_z[no_free_surf], bnd_x[no_free_surf])
        if self.dimension == 3:  # 3D
            y_f = node_positions[2]
            bnd_y = y_f[bnd_nod]
            bnd_nodes_nfs += (bnd_y[no_free_surf],)

        # Boundary node indices on non-free surfaces
        bnd_nfs = bnd_nod[no_free_surf]

        return bnd_nfs, bnd_nodes_nfs
