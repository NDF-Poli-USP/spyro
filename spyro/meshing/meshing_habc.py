import firedrake as fire
import numpy as np
from netgen.geom2d import SplineGeometry
from netgen.meshing import Element2D, \
    Element3D, FaceDescriptor, Mesh, MeshPoint
from scipy.spatial import Delaunay, cKDTree
from SeismicMesh import generation, geometry
from spyro.utils.error_management import value_parameter_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender and Romildo Soares Jr


class Sign_Distance_Hyp3D():
    '''
    Class for the signed distance function of a truncated
    hyperellipsoid with a truncation plane at x = x_cut

    Attributes
    ----------
    semi_axes : `tuple`
        Semi-axes of the hyperellipsoid (a, b, c)
    centroid : `tuple`
        Centroid of the hyperellipsoid (xc, yc, zc)
     n : `float`
        Degree of the hyperellipsoid
    x_cut : `float`, optional
        x-coordinate of the truncation plane. Default is 0

    Methods
    -------
    hyperellipsoid_sdf()
        Compute the signed distance function for the hyperellipsoid
    sign_dist_hyptrunc
            Compute the combined signed distance function to
            create the boundary of a truncated hypererellipsoid
    top_cut_sdf()
        Compute the signed distance function for the top cut
    '''

    def __init__(self, semi_axes, centroid, n, x_cut=0.):
        '''
        Initialize the Sign_Distance_Hyp3D class.
        Create a signed distance function for a truncated
        hyperellipsoid with a truncation plane at x = x_cut

        Parameters
        ----------
        semi_axes : `tuple`
            Semi-axes of the hyperellipsoid (a, b, c)
        centroid : `tuple`
            Centroid of the hyperellipsoid (xc, yc, zc)
        n : `float`
            Degree of the hyperellipsoid
        x_cut : `float`, optional
            x-coordinate of the truncation plane. Default is 0

        Returns
        -------
        None
        '''

        # Validate input parameters
        if len(semi_axes) != 3 or any(sa <= 0 for sa in semi_axes):
            raise ValueError("semi_axes must be a tuple of "
                             "three positive values (a, b, c).")

        if len(centroid) != 3:
            raise ValueError("centroid must be a tuple of "
                             "three values (xc, yc, zc).")

        if n <= 0:
            raise ValueError("n must be a positive value.")

        if not isinstance(x_cut, (int, float)):
            raise ValueError("x_cut must be a numeric value.")

        # Assign attributes
        self.semi_axes = semi_axes
        self.centroid = centroid
        self.n = n
        self.x_cut = x_cut

    def hyperellipsoid_sdf(self, pts):
        '''
        Compute the signed distance function for the hyperellipsoid

        Parameters
        ----------
        pts : `array`
            Array containing the coordinates of the points
            where the signed distance function is evaluated

        Returns
        -------
        sdf : `array`
            Array containing the signed distance function values
        '''

        # Semi-axes and centroid
        a, b, c = self.semi_axes
        xc, yc, zc = self.centroid

        # Relative coordinates to center at (xc, yc, zc)
        x_h = abs((pts[:, 0] - xc) / a)**self.n
        y_h = abs((pts[:, 1] - yc) / b)**self.n
        z_h = abs((pts[:, 2] - zc) / c)**self.n

        # Compute radius in hyperellipsoid space
        r = (x_h + y_h + z_h)**(1. / self.n)

        # Signed distance
        sign_dist = r - 1.

        # Rescale back to original space
        scale = min(a, b, c)

        return sign_dist * scale

    def top_cut_sdf(self, pts):
        '''
        Compute the signed distance function for the top cut

        Parameters
        ----------
        pts : `array`
            Array containing the coordinates of the points
            where the signed distance function is evaluated

        Returns
        -------
        sdf : `array`
            Array containing the signed distance function values
        '''
        return (pts[:, 0] - self.x_cut)

    def sign_dist_hyptrunc(self, pts):
        '''
        Compute the combined signed distance function to
        create the boundary of a truncated hypererellipsoid

        Parameters
        ----------
        pts : `array`
            Array containing the coordinates of the points
            where the signed distance function is evaluated

        Returns
        -------
        sdf : `array`
            Array containing the signed distance function values
        '''
        return np.maximum(self.hyperellipsoid_sdf(pts), self.top_cut_sdf(pts))


class HABC_Mesh():
    '''
    Class for HABC mesh generation

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
    dom_dim : `tuple`
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
    tol : `float`
        Tolerance for searching nodes in the mesh

    Methods
    -------
    bnd_pnts_hyp_2D()
        Generate points on the boundary of a hyperellipse
    clipping_coordinates_lay_field()
        Generate a field with clipping coordinates to the original boundary
    create_bnd_mesh_2D()
        Generate the boundary segment curves for the hyperellipse boundary mesh
    create_hyp_trunc_mesh_2D()
        Generate the mesh for the hyperelliptical absorbing layer
    extend_velocity_profile()
        Extend the velocity profile inside the absorbing layer
    extract_bnd_node_indices()
        Extract boundary node indices on boundaries of the domain
        excluding the free surface at the top boundary
    get_clean_mesh_pnts3D()
        Remove points outside the original mesh and too close to the box
    hypershape_mesh_habc()
        Generate a mesh with a hypershape absorbing layer
    layer_boundary_data()
        Generate the boundary data from the domain with the absorbing layer
    layer_mask_field()
        Generate a mask for the absorbing layer
    merge_mesh_2D()
        Merge the rectangular and the hyperelliptical meshes
    merge_mesh_3D()
        Build a merged mesh from a box mesh and a hyperellipsoidal mesh
    mesh_data_3D()
        Generate mesh data for the hyperellipsoidal absorbing layer
    original_boundary_data()
        Generate the boundary data from the original domain mesh
    point_cloud_field()
        Create a field on a point cloud from a parent mesh and field
    preamble_mesh_operations()
        Perform mesh operations previous to size an absorbing layer
    properties_eik_mesh()
        Set the properties for the mesh used to solve the Eikonal equation
    rectangular_mesh_habc()
        Generate a rectangular mesh with an absorbing layer
    remove_elements_outside_hyp3D()
        Remove elements outside the hyperellipsoid.
    representative_mesh_dimensions()
        Get the representative mesh dimensions from original mesh
    trunc_hyp_bndpts_2D()
        Generate the boundary points for a truncated hyperellipse
    '''

    def __init__(self, dom_dim, dimension=2, comm=None):
        '''
        Initialize the HABC_Mesh class

        Parameters
        ----------
        dom_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D
        comm : `object`, optional
            An object representing the communication interface
            for parallel processing. Default is None

        Returns
        -------
        None
        '''

        # Original domain dimensions
        self.dom_dim = dom_dim

        # Model dimension
        self.dimension = dimension

        # Communicator MPI
        self.comm = comm

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

        # Ratio between the representative mesh dimensions
        self.alpha = self.lmax / self.lmin

        # Tolerance for searching nodes in the mesh
        self.tol = 10**(min(int(np.log10(self.lmin / 10)), -6))

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

    def extract_bnd_node_indices(self, node_positions, func_space):
        '''
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

    def original_boundary_data(self):
        '''
        Generate the boundary data from the original domain mesh

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

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

        # Get extreme values of the velocity model on the boundarty
        vel_on_boundary = self.point_cloud_field(
            self.mesh_original, np.asarray(self.bnd_nodes).T,
            self.initial_velocity_model).dat.data_with_halos[:]
        self.c_bnd_min = vel_on_boundary.min()
        self.c_bnd_max = vel_on_boundary.max()

        # Print on screen
        cbnd_str = "Boundary Velocity Range (km/s): {:.3f} - {:.3f}"
        print(cbnd_str.format(self.c_bnd_min, self.c_bnd_max))

    def properties_eik_mesh(self, p_usu=None, ele_type='CG', f_est=0.03):
        '''
        Set the properties for the mesh used to solve the Eikonal equation

        Parameters
        ----------
        p_usu : `int`, optional
            Finite element order for the Eikonal equation. Default is None
        ele_type : `string`, optional
            Finite element type. 'CG' or 'KMV'. Default is 'CG'
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03

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

        # Factor for the stabilizing term in Eikonal equation
        self.f_est = f_est

    def preamble_mesh_operations(self, f_est=0.03):
        '''
        Perform mesh operations previous to size an absorbing layer

        Parameters
        ----------
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03

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

        # Get finite element data from the velocity model
        self.ele_type_c0 = self.initial_velocity_model.ufl_element().family()
        self.p_c0 = \
            self.initial_velocity_model.function_space().ufl_element().degree()

        # Get extreme values of the velocity model
        self.c_min = self.initial_velocity_model.dat.data_with_halos.min()
        self.c_max = self.initial_velocity_model.dat.data_with_halos.max()

        # Print on screen
        cdom_str = "Domain Velocity Range (km/s): {:.3f} - {:.3f}"
        print(cdom_str.format(self.c_min, self.c_max))

        # Save initial velocity model
        vel_c = fire.VTKFile(self.path_save + "preamble/c_vel.pvd")
        vel_c.write(self.c)

        # Generating boundary data from the original domain mesh
        print("Getting Boundary Mesh Data from Original Domain")
        self.original_boundary_data()

        # Mesh properties for Eikonal
        print("Setting Mesh Properties for Eikonal Analysis")
        self.properties_eik_mesh(p_usu=self.abc_deg_eikonal, f_est=f_est)

    def rectangular_mesh_habc(self, dom_lay, pad_len):
        '''
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
        '''

        # Domain dimensions
        Lx, Lz = self.dom_dim[:2]

        # Number of elements
        n_pad = pad_len / self.lmin  # Elements in the layer
        nz = int(Lz / self.lmin) + int(n_pad)
        nx = int(Lx / self.lmin) + int(2 * n_pad)
        nx += nx % 2

        # New geometry with layer
        Lx_habc, Lz_habc = dom_lay[:2]

        # Creating the rectangular mesh with layer
        if self.dimension == 2:  # 2D
            mesh_habc = fire.RectangleMesh(
                nz, nx, Lz_habc, Lx_habc, comm=self.comm.comm)

        if self.dimension == 3:  # 3D

            # Number of elements
            Ly = self.dom_dim[2]
            ny = int(Ly / self.lmin) + int(2 * n_pad)
            ny += ny % 2

            # New geometry with layer
            Ly_habc = dom_lay[2]

            # Mesh
            mesh_habc = fire.BoxMesh(
                nz, nx, ny, Lz_habc, Lx_habc, Ly_habc, comm=self.comm.comm)

            # Adjusting coordinates
            mesh_habc.coordinates.dat.data_with_halos[:, 2] -= pad_len

        # Adjusting coordinates
        mesh_habc.coordinates.dat.data_with_halos[:, 0] *= -1.0
        mesh_habc.coordinates.dat.data_with_halos[:, 1] -= pad_len

        print("Extended Rectangular Mesh Generated Successfully")

        return mesh_habc

    @staticmethod
    def bnd_pnts_hyp_2D(a, b, n, num_pts):
        '''
        Generate points on the boundary of a hyperellipse.

        'Parameters
        ----------
        a : `float`
            Hyperellipse semi-axis in direction 1
        b : `float`
            Hyperellipse semi-axis in direction 2
        n : `int`
            Degree of the hyperellipse
        num_pts : `int`
            Number of points to generate on the hyperellipse boundary

        Returns
        -------
        bnd_pnts : `array`
            Array of shape (num_pts, 2) containing the coordinates
            of the hyperellipse boundary points
        '''

        # Generate angle values for the parametric equations
        theta = np.linspace(0, 2 * np.pi, num_pts)

        # Especial angle values
        rc_zero = [np.pi / 2., 3 * np.pi / 2.]
        rs_zero = [0., np.pi, 2 * np.pi]

        # Trigonometric function evaluation
        cr = np.cos(theta)
        sr = np.sin(theta)
        cr = np.where(np.isin(theta, rc_zero), 0, cr)
        sr = np.where(np.isin(theta, rs_zero), 0, sr)

        # Parametric equations for the hyperellipse
        x = a * np.sign(cr) * np.abs(cr)**(2 / n)
        y = b * np.sign(sr) * np.abs(sr)**(2 / n)

        bnd_pnts = np.column_stack((x, y))

        return bnd_pnts

    def trunc_hyp_bndpts_2D(self, hyp_par, xdom, z0):
        '''
        Generate the boundary points for a truncated hyperellipse

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipse parameters.
            Structure: (n_hyp, perimeter, a_hyp, b_hyp)
            - n_hyp : `float`
                Degree of the hyperellipse
            - perimeter : `float`
                Perimeter of the full hyperellipse (2D)
            - a_hyp : `float`
                Hyperellipse semi-axis in direction x
            - b_hyp : `float`
                Hyperellipse semi-axis in direction z
        xdom : `float`
            Maximum coordinate in normal to the truncation plane
        z0 : `float`
            Truncation plane

        Returns
        -------
        filt_bnd_pts : `array`
            Array of shape (num_bnd_pts, 2) containing the coordinates
            of the truncated hyperellipse boundary points
        trunc_feat : `tuple`
            Truncation features.
            Structure: [ini_trunc, end_trunc, num_filt_bnd_pts, ltrunc]
            - ini_trunc : `int`
                Index of the first point after the truncation plane
            - end_trunc : `int`
                Index of the last point before the truncation plane
            - num_filt_bnd_pts : `int`
                Number of filtered boundary points
            - ltrunc : `float`
                Mesh size for arc length due to the truncation operation
        '''

        # Hyperellipse parameters
        n_hyp, perimeter, a_hyp, b_hyp = hyp_par

        # Boundary points: Use 16 or 24 as a minimum
        lmin = self.lmin
        num_bnd_pts = int(max(np.ceil(perimeter / lmin), 16)) - 1

        # Generate the hyperellipse boundary points
        pnt_bef_trunc = 0
        pnt_aft_trunc = 0
        pnt_str = "Number of Boundary Points for"
        while pnt_bef_trunc % 2 == 0 or pnt_aft_trunc % 2 == 0 \
                or pnt_bef_trunc < 3 or pnt_aft_trunc < 3:

            num_bnd_pts += 1
            print(pnt_str, f"Complete Hyperellipse: {num_bnd_pts}")
            bnd_pts = self.bnd_pnts_hyp_2D(a_hyp, b_hyp, n_hyp, num_bnd_pts)

            # Filter hyperellipse points based on the truncation plane z0
            filt_bnd_pts = np.array([point for point in bnd_pts
                                     if point[1] <= z0])
            print(pnt_str, f"Truncated Hyperellipse: {len(filt_bnd_pts)}")

            # Identify truncation index
            ini_trunc = max(np.where(bnd_pts[:, 1] > z0)[0][0] - 1, 0)

            # Modify points to match with the truncation plane
            r0 = np.asin((z0 / b_hyp)**(n_hyp / 2))
            x0 = a_hyp * np.cos(r0)**(2 / n_hyp)
            filt_bnd_pts[ini_trunc] = np.array([x0, z0])
            filt_bnd_pts[ini_trunc + 1] = np.array([-x0, z0])

            # Insert new points to create a rectangular trunc
            new_pnts = np.array([[xdom, z0], [xdom, -z0],
                                 [-xdom, -z0], [-xdom, z0]])
            filt_bnd_pts = np.insert(filt_bnd_pts, ini_trunc + 1,
                                     new_pnts, axis=0)
            end_trunc = ini_trunc + 5
            # Points before and after truncation
            pnt_bef_trunc = len(filt_bnd_pts[:ini_trunc + 1])
            pnt_aft_trunc = len(filt_bnd_pts[end_trunc:])

        # Total number of the boundary points including the trunc
        num_filt_bnd_pts = len(filt_bnd_pts)

        # Mesh size for arc length due to the truncation operation
        ltrunc = perimeter / num_bnd_pts

        # Truncation features
        trunc_feat = (ini_trunc, end_trunc, num_filt_bnd_pts, ltrunc)

        return filt_bnd_pts, trunc_feat

    def create_bnd_mesh_2D(self, geo, bnd_pts, trunc_feat, spln):
        '''
        Generate the boundary segments for the hyperellipse boundary mesh

        Parameters
        ----------
        geo : `SplineGeometry`
            Geometry object with the data to generate the mesh
        bnd_pts : `array`
            Array of shape (num_bnd_pts, 2) containing the coordinates
            of the hyperellipse boundary points
        trunc_feat : `tuple`
            Truncation features.
            Structure: [ini_trunc, end_trunc, num_filt_bnd_pts, ltrunc]
            - ini_trunc : `int`
                Index of the first point after the truncation plane
            - end_trunc : `int`
                Index of the last point before the truncation plane
            - num_filt_bnd_pts : `int`
                Number of filtered boundary points
            - ltrunc : `float`
                Mesh size for arc length due to the truncation operation
        spln : `bool`
            Flag to indicate whether to use splines (True) or lines (False)

        Returns
        -------
        curves : `list`
            List of curves to be added to the geometry object. Each curve is
            represented as a list containing the curve type and its points.
        '''

        ini_trunc, end_trunc, num_bnd_pts, ltrunc = trunc_feat

        curves = []
        if spln:
            # Mesh with spline segments
            for idp in range(0, ini_trunc, 2):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                p3 = geo.PointData()[2][idp + 2]
                curves.append(["spline3", p1, p2, p3, ltrunc])
                # print(p1, p2, p3)

            for idp in range(ini_trunc, end_trunc):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                # print(p1, p2)

                if idp == ini_trunc or idp == end_trunc - 1:
                    curves.append(["line", p1, p2, ltrunc])
                else:
                    curves.append(["line", p1, p2, self.lmin])

            for idp in range(end_trunc, num_bnd_pts - 1, 2):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                p3 = geo.PointData()[2][idp + 2]
                curves.append(["spline3", p1, p2, p3, ltrunc])
                # print(p1, p2, p3)

        else:
            # Mesh with line segments
            for idp in range(0, num_bnd_pts - 1, 2):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                # print(p1, p2)

                if ini_trunc + 1 <= idp <= end_trunc - 2:
                    curves.append(["line", p1, p2, self.lmin])
                else:
                    curves.append(["line", p1, p2, ltrunc])

        return curves

    def create_hyp_trunc_mesh_2D(self, hyp_par, spln=True, fmesh=1.):
        '''
        Generate the mesh for the hyperelliptical absorbing layer

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipse parameters.
            Structure: (n_hyp, perimeter, a_hyp, b_hyp)
            - n_hyp : `float`
                Degree of the hyperellipse
            - perimeter : `float`
                Perimeter of the full hyperellipse (2D)
            - a_hyp : `float`
                Hyperellipse semi-axis in direction x
            - b_hyp : `float`
                Hyperellipse semi-axis in direction z
        spln : `bool`, optional
            Flag to indicate whether to use splines (True) or lines (False)
        fmesh : `float`, optional
            Mesh size factor for the hyperelliptical layer with respect to mesh
            size of the original domain. Default is 1.0.

        Returns
        -------
        hyp_mesh : `netgen mesh`
            Generated mesh for the hyperelliptical layer
        '''

        # Domain dimensions
        Lx, Lz = self.dom_dim[:2]

        # Generate the hyperellipse boundary points
        bnd_pts, trunc_feat = self.trunc_hyp_bndpts_2D(hyp_par, Lx / 2, Lz / 2)

        # Initialize geometry
        geo = SplineGeometry()

        # Append points to the geometry
        [geo.AppendPoint(*pnt) for pnt in bnd_pts]

        #  Mesh size for arc length due to the truncation operation
        ltrunc = trunc_feat[-1]
        fmin = ltrunc / self.lmin

        # Mesh size factor for the hyperelliptical layer
        if fmesh != 1.:
            fmesh = max(fmesh, fmin)
        fm_str = "Mesh Factor Size Inside Layer (Min): {:.2f} ({:.2f})"
        print(fm_str.format(fmesh, fmin))

        while True:
            try:
                # Generate the boundary segment curves
                curves = self.create_bnd_mesh_2D(geo, bnd_pts,
                                                 trunc_feat, spln)
                [geo.Append(c[:-1], bc="outer", maxh=c[-1], leftdomain=1,
                            rightdomain=0) for c in curves]

                # Generate the mesh using netgen library
                hyp_mesh = geo.GenerateMesh(maxh=fmesh*self.lmin,
                                            quad_dominated=False)
                print("Hyperelliptical Mesh Generated Successfully")
                break

            except Exception as e:
                if spln:
                    print(f"Error Meshing with Splines: {e}")
                    print("Now meshing with Lines")
                    spln = False
                else:
                    UserWarning(f"Error Meshing with Lines: {e}. Exiting.")
                    break

        # Mesh is transformed into a firedrake mesh
        return fire.Mesh(hyp_mesh, comm=self.comm.comm)

    def merge_mesh_2D(self, rec_mesh, hyp_mesh):
        '''
        Merge the rectangular and the hyperelliptical meshes

        Parameters
        ----------
        rec_mesh : `firedrake mesh`
            Rectangular mesh representing the original domain
        hyp_mesh : `firedrake mesh`
            Hyperelliptical annular mesh representing the absorbing layer

        Returns
        -------
        final_mesh : `firedrake mesh`
            Merged final mesh
        '''

        # Create the final mesh that will contain both
        final_mesh = Mesh()
        final_mesh.dim = self.dimension

        # Get coordinates of the rectangular mesh
        coord_rec = rec_mesh.coordinates.dat.data_with_halos

        # Create KDTree for efficient nearest neighbor search
        boundary_coords = np.column_stack((self.bnd_nodes[0],
                                           self.bnd_nodes[1]))
        boundary_tree = cKDTree(boundary_coords)

        # Add all vertices from rectangular mesh and create mapping
        rec_map = {}
        boundary_coords = []
        boundary_points = []
        for i, coord in enumerate(coord_rec):
            z, x = coord
            rec_map[i] = final_mesh.Add(MeshPoint((z, x, 0.)))  # y = 0 for 2D

            # Check if the point is on the original boundary
            if boundary_tree.query(
                coord, distance_upper_bound=self.tol,
                    workers=-1)[0] <= self.tol:
                boundary_coords.append(coord)
                boundary_points.append(rec_map[i])

        # Create KDTree for efficient nearest neighbor search
        boundary_tree = cKDTree(np.asarray(boundary_coords))

        # Face descriptor for the rectangular mesh
        fd_rec = final_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=2))

        # Get mesh cells from rectangular mesh
        rec_cells = rec_mesh.coordinates.cell_node_map().values_with_halo

        # Add all elements from rectangular mesh to the netgen mesh
        final_mesh.SetMaterial(1, "rec")
        for cell in rec_cells:
            netgen_points = [rec_map[cell[i]] for i in range(len(cell))]
            final_mesh.Add(Element2D(fd_rec, netgen_points))

        # Add all vertices from hyperelliptical mesh and create mapping
        hyp_map = {}
        coord_hyp = hyp_mesh.coordinates.dat.data_with_halos
        for i, coord in enumerate(coord_hyp):
            z, x = coord

            # Check if the point is on the original boundary
            dist, idx = boundary_tree.query(
                coord, distance_upper_bound=self.tol, workers=-1)

            if dist <= self.tol:
                # Reuse the existing point
                hyp_map[i] = boundary_points[idx]
            else:
                # Creating a new point (y = 0 for 2D)
                hyp_map[i] = final_mesh.Add(MeshPoint((z, x, 0.)))

        # Face descriptor for the hyperelliptical mesh
        fd_hyp = final_mesh.Add(FaceDescriptor(bc=2, domin=2, domout=0))

        # Get mesh cells from hyperelliptical mesh
        hyp_cells = hyp_mesh.coordinates.cell_node_map().values_with_halo

        # Add all elements from hyperelliptical mesh to the netgen mesh
        final_mesh.SetMaterial(2, "hyp")
        for cell in hyp_cells:
            netgen_points = [hyp_map[cell[i]] for i in range(len(cell))]
            final_mesh.Add(Element2D(fd_hyp, netgen_points))

        try:
            # Mesh data
            final_mesh.Compress()
            print(f"Mesh created with {len(final_mesh.Points())} points "
                  f"and {len(final_mesh.Elements2D())} elements")

            # Mesh is transformed into a firedrake mesh
            final_mesh = fire.Mesh(final_mesh, comm=self.comm.comm)
            print("Merged Mesh Generated Successfully")

        except Exception as e:
            UserWarning(f"Error Generating Merged Mesh: {e}. Exiting.")

        return final_mesh

    def mesh_data_3D(self, hyp_par):
        '''
        Generate mesh data for the hyperellipsoidal absorbing layer

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipsoid parameters.
            Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hyperellipsoid
            - surface : `float`
                Surface area of the full hyperellipsoid (3D)
            - a_hyp : `float`
                Hyperellipsoid semi-axis in direction x
            - b_hyp : `float`
                Hyperellipsoid semi-axis in direction z
            - c_hyp : `float`
                Hyperellipsoid semi-axis in direction y

        Returns
        -------
        rec_box : `tuple`
            Box defined by original domain dimensions
            Structure: (zmin, zmax, xmin, xmax, ymin, ymax)
        hyp_box : `tuple`
            Box that envelopes the hyperellipsoid
            Structure: (zmin, zmax, xmin, xmax, ymin, ymax)
        fixed_pnts : `array`
            Array containing the coordinates of the original mesh points
        centroid : `tuple`
            Hyperellipsoid centroid according to reference system (z, x, y)
            Structure: (zc, xc, yc)
        semi_axes : `tuple`
            Hyperellipsoid semi-axes according to reference system (z, x, y)
            Structure: (b_hyp, a_hyp, c_hyp)
        surf_ele_expt : `int`
            Expected elements in a full hyperellipsoid surface
        '''

        # Domain dimensions
        Lx, Lz, Ly = self.dom_dim

        # Hyperellipsoid parameters
        n_hyp, surface, a_hyp, b_hyp, c_hyp = hyp_par

        # Box defined by original domain dimensions
        rec_box = (-Lz, 0., 0., Lx, 0., Ly)

        # Hyperellipsoid centroid
        zc = -Lz / 2.
        xc = Lx / 2.
        yc = Ly / 2.

        # Box that envelopes the hyperellipsoid
        hyp_zmin = zc - b_hyp
        hyp_zmax = zc + b_hyp
        hyp_xmin = xc - a_hyp
        hyp_xmax = xc + a_hyp
        hyp_ymin = yc - c_hyp
        hyp_ymax = yc + c_hyp
        hyp_box = (hyp_zmin, hyp_zmax, hyp_xmin, hyp_xmax, hyp_ymin, hyp_ymax)

        # Extract node positions
        fixed_pnts = self.mesh_original.coordinates.dat.data_with_halos

        # Clip fixed_pnts points to ensure they are within the bounding box
        fixed_pnts[:, 0] = np.clip(fixed_pnts[:, 0], hyp_zmin, hyp_zmax)
        fixed_pnts[:, 1] = np.clip(fixed_pnts[:, 1], hyp_xmin, hyp_xmax)
        fixed_pnts[:, 2] = np.clip(fixed_pnts[:, 2], hyp_ymin, hyp_ymax)

        # Hyperellipsoid centroid according to reference system (z, x, y)
        centroid = (zc, xc, yc)

        # Hyperellipsoid semi-axes according to reference system (z, x, y)
        semi_axes = (b_hyp, a_hyp, c_hyp)

        # Expected elements in a full hyperellipsoid surface
        surf_ele_expt = np.ceil(surface / self.lmin**2) + 1

        return rec_box, hyp_box, fixed_pnts, centroid, semi_axes, surf_ele_expt

    def get_clean_mesh_pnts3D(self, all_pnts, dom_pnts, box=None):
        '''
        Remove points outside the original mesh and too close to the box

        Parameters
        ----------
        all_pnts : `array`
            Array containing the coordinates of all the mesh points
        dom_pnts : `array`
            Array containing the coordinates of the original mesh points
        box : `tuple`, optional
            Original box bounds. Default is None
            Structure: (xmin, xmax, ymin, ymax, zmin, zmax).

        Returns
        -------
        clean_pnts : `array`
            Array containing the coordinates of the filtered mesh points
        '''

        if len(all_pnts) == 0:
            return all_pnts.copy()

        # Convert to numpy arrays
        all_pnts = np.asarray(all_pnts)
        dom_pnts = np.asarray(dom_pnts)

        # Determine box bounds
        if box is None:
            if len(dom_pnts) == 0:
                return all_pnts.copy()

            # If not provided
            xmin = np.min(dom_pnts[:, 0])
            xmax = np.max(dom_pnts[:, 0])
            ymin = np.min(dom_pnts[:, 1])
            ymax = np.max(dom_pnts[:, 1])
            zmin = np.min(dom_pnts[:, 2])
            zmax = np.max(dom_pnts[:, 2])
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = box

        # Vectorized check
        x, y, z = all_pnts[:, 0], all_pnts[:, 1], all_pnts[:, 2]
        cond_out = ((x < xmin - self.tol) | (x > xmax + self.tol)
                    | (y < ymin - self.tol) | (y > ymax + self.tol)
                    | (z < zmin - self.tol) | (z > zmax + self.tol))

        # Get inside points only
        in_pnts = all_pnts[~cond_out]

        if len(in_pnts):
            # Build k-d tree for fast nearest neighbor queries
            dom_tree = cKDTree(dom_pnts)

            # Batch query all inside points at once
            distances = dom_tree.query(
                in_pnts, distance_upper_bound=self.tol, workers=-1)[0]

            # Get the points are close to domain points
            clean_in_pnts = in_pnts[distances <= self.tol]
        else:
            clean_in_pnts = np.array([])

        # Get outside points only
        out_pnts = all_pnts[cond_out]

        if len(out_pnts):

            # Vectorized minimum distance to box calculation
            x, y, z = out_pnts[:, 0], out_pnts[:, 1], out_pnts[:, 2]

            # Calculate distance components
            dx = np.maximum(0, np.maximum(xmin - x, x - xmax))
            dy = np.maximum(0, np.maximum(ymin - y, y - ymax))
            dz = np.maximum(0, np.maximum(zmin - z, z - zmax))

            # Euclidean distance
            distances = np.sqrt(dx**2 + dy**2 + dz**2)

            # Keep points that are far enough from structured box mesh
            clean_out_pnts = out_pnts[distances > self.lmin / 1.5]
        else:
            clean_out_pnts = np.array([])

        # Combine outside points and inside clean points
        clean_pnts = np.vstack([clean_out_pnts, clean_in_pnts])

        return clean_pnts if len(clean_pnts) else np.array([]).reshape(0, 3)

    @staticmethod
    def remove_elements_outside_hyp3D(pts, connect, geps, hyp3D_sdf):
        '''
        Remove elements outside the hyperellipsoid.
        Based on a post-processing function used by SeismicMesh.

        Parameters
        ----------
        pts : `array`
            Array containing the coordinates of the mesh points
        connect : `array`
            Array containing the connectivity of the mesh points
        geps : `float`
            Geometric tolerance for the hyperellipsoid surface
        hyp3D_sdf : `function`
            Function to evaluate the signed distance to the hyperellipsoid

        Returns
        -------
        connect : `array`
            Filtered array of shape (num_tets_filtered, 4) containing the
            indices of the mesh points that form each tetrahedral element
            3D array
        '''

        dim = pts.shape[1]
        pmid = pts[connect].sum(1) / (dim + 1)  # Compute centroids

        return connect[hyp3D_sdf(pmid) < -geps]  # Keep interior elements

    def merge_mesh_3D(self, hyp_par):
        '''
        Build a merged mesh from a box mesh and a hyperellipsoidal mesh

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipsoid parameters.
            Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hyperellipsoid
            - surface : `float`
                Surface area of the full hyperellipsoid (3D)
            - a_hyp : `float`
                Hyperellipsoid semi-axis in direction x
            - b_hyp : `float`
                Hyperellipsoid semi-axis in direction z
            - c_hyp : `float`
                Hyperellipsoid semi-axis in direction y

        Returns
        -------
        final_mesh : `firedrake mesh`
            Merged final mesh
        '''

        # Hyperellipse parameters
        n_hyp, surface, a_hyp, b_hyp, c_hyp = hyp_par

        rec_box, hyp_box, pfix, centr, s_axs, \
            surf_ele_expt = self.mesh_data_3D(hyp_par)

        sdfunct = Sign_Distance_Hyp3D(s_axs, centr, n_hyp)

        # Generate a initial 3D mesh
        print("Generating Initial Mesh with Fixed Points in Original Box")
        orig_dom = geometry.Cube(rec_box)
        points, cells = generation.generate_mesh(
            domain=sdfunct.sign_dist_hyptrunc, edge_length=self.lmin,
            bbox=hyp_box, max_iter=50, pfix=pfix, subdomains=[orig_dom])

        # Get clean points to generate final mesh
        print("Getting Clean Points to Generate Final Mesh")
        points = self.get_clean_mesh_pnts3D(points, pfix, rec_box)

        # Create new Delaunay with structured mesh part
        fixed_triangulation = Delaunay(points)
        cells = fixed_triangulation.simplices

        # SeismicMesh filter for boundary elements
        print("Removing Elements Outside the Hypershape")
        geps = (surface / (1e3 * surf_ele_expt))**0.5
        cells = self.remove_elements_outside_hyp3D(points, cells, geps,
                                                   sdfunct.sign_dist_hyptrunc)

        # Create the final mesh
        final_mesh = Mesh()
        final_mesh.dim = self.dimension
        final_mesh.SetMaterial(1, "habc")

        # Add points
        point_map = {}
        for i, coord in enumerate(points):
            z, x, y = coord
            point_map[i] = final_mesh.Add(MeshPoint((z, x, y)))

        # Add elements
        for cell in cells:
            netgen_points = [point_map[cell[i]] for i in range(len(cell))]
            final_mesh.Add(Element3D(1, netgen_points))  # 1: Volume marker

        try:
            # Mesh data
            final_mesh.Compress()
            print(f"Mesh created with {len(final_mesh.Points())} points "
                  f"and {len(final_mesh.Elements3D())} elements")

            # Mesh is transformed into a firedrake mesh
            final_mesh = fire.Mesh(final_mesh, comm=self.comm.comm)
            print("Merged Mesh Generated Successfully")

        except Exception as e:
            UserWarning(f"Error Generating Merged Mesh: {e}. Exiting.")

        return final_mesh

    def hypershape_mesh_habc(self, hyp_par, spln=True, fmesh=1.):
        '''
        Generate a mesh with a hypershape absorbing layer

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipshape parameters.
            Structure 2D: (n_hyp, perimeter, a_hyp, b_hyp)
            Structure 3D: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hypershape
            - perimeter : `float`
                Perimeter of the full hyperellipse (2D)
            - surface : `float`
                Surface area of the full hyperellipsoid (3D)
            - a_hyp : `float`
                Hypershape semi-axis in direction x
            - b_hyp : `float`
                Hypershape semi-axis in direction z
            - c_hyp : `float`
                Hypershape semi-axis in direction y (3D only)
        spln : `bool`, optional
            Flag to indicate whether to use splines (True) or lines (False)
            in hypershape layer generation. Default is True
        fmesh : `float`, optional
            Mesh size factor for the hyperelliptical layer with respect to mesh
            size of the original domain. Default is 1.0.

        Returns
        -------
        mesh_habc : `firedrake mesh`
            Mesh with a hypershape absorbing layer
        '''

        if self.dimension == 2:  # 2D

            # Domain dimensions
            Lx, Lz = self.dom_dim[:2]

            # Creating the hyperellipse layer mesh
            hyp_mesh = self.create_hyp_trunc_mesh_2D(hyp_par[:4],
                                                     spln=spln,
                                                     fmesh=fmesh)

            # Adjusting coordinates: Swap (x,z) -> (z,x) and apply offsets
            hyp_mesh.coordinates.dat.data_with_halos[:, [0, 1]] = \
                hyp_mesh.coordinates.dat.data_with_halos[:, [1, 0]]
            hyp_mesh.coordinates.dat.data_with_halos[:, 0] -= Lz / 2
            hyp_mesh.coordinates.dat.data_with_halos[:, 1] += Lx / 2
            fire.VTKFile("output/trunc_hyp_test.pvd").write(hyp_mesh)

            # Merging the original mesh with the hyperellipse layer mesh
            mesh_habc = self.merge_mesh_2D(self.mesh_original, hyp_mesh)
            # fire.VTKFile("output/trunc_merged_test.pvd").write(mesh_habc)

        if self.dimension == 3:  # 3D

            # Merging the original mesh with a hyperellipsoid layer mesh
            mesh_habc = self.merge_mesh_3D(hyp_par)
            # fire.VTKFile("output/trunc_merged_test3d.pvd").write(mesh_habc)

        return mesh_habc

    def layer_mask_field(self, coords, V, damp_par=None,
                         type_marker='damping', name_mask=None):
        '''
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
        '''

        # Domain dimensions
        Lx, Lz = self.dom_dim[:2]

        # Domain coordinates
        z, x = coords[0], coords[1]

        # Conditional value
        val_condz = (z + Lz)**2 if type_marker == 'damping' else 1.0
        val_condx1 = x**2 if type_marker == 'damping' else 1.0
        val_condx2 = (x - Lx)**2 if type_marker == 'damping' else 1.0

        # Conditional expressions for the mask
        z_pd = fire.conditional(z + Lz < 0., val_condz, 0.)
        x_pd = fire.conditional(x < 0., val_condx1, 0.) + \
            fire.conditional(x - Lx > 0., val_condx2, 0.)
        ref = z_pd + x_pd

        if self.dimension == 3:  # 3D

            # 3D dimension
            Ly = self.dom_dim[2]
            y = coords[2]

            # Conditional value
            val_condy1 = y**2 if type_marker == 'damping' else 1.0
            val_condy2 = (y - Ly)**2 if type_marker == 'damping' else 1.0

            # Conditional expressions for the mask
            y_pd = fire.conditional(y < 0., val_condy1, 0.) + \
                fire.conditional(y - Ly > 0., val_condy2, 0.)
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
            ref = fire.Constant(eta_crt) * (fire.Constant(aq) * ref**2
                                            + fire.Constant(bq) * ref)

        elif type_marker == 'mask':
            # Mask filter for layer boundary domain
            ref = fire.conditional(ref > 0, 1., 0.)

        else:
            value_parameter_error('type_marker', type_marker,
                                  ['damping', 'mask'])

        layer_mask = fire.Function(V, name=name_mask)
        layer_mask.interpolate(ref)

        return layer_mask

    def clipping_coordinates_lay_field(self, V):
        '''
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
        '''

        print("Clipping Coordinates Inside Layer")

        # Domain dimensions
        Lx, Lz = self.dom_dim[:2]

        # Vectorial space for auxiliar field of clipped coordinates
        W_sp = fire.VectorFunctionSpace(self.mesh, self.ele_type_c0, self.p_c0)

        # Mesh coordinates
        coords = fire.SpatialCoordinate(self.mesh)

        # Clipping coordinates
        lay_field = fire.Function(W_sp).interpolate(coords)
        lay_arr = lay_field.dat.data_with_halos[:]
        lay_arr[:, 0] = np.clip(lay_arr[:, 0], -Lz, 0.)
        lay_arr[:, 1] = np.clip(lay_arr[:, 1], 0., Lx)

        if self.dimension == 3:  # 3D

            # 3D dimension
            Ly = self.dom_dim[2]

            # Clipping coordinates
            lay_arr[:, 2] = np.clip(lay_arr[:, 2], 0., Ly)

        # Mask function to identify the absorbing layer domain
        layer_mask = self.layer_mask_field(coords, V, type_marker='mask')

        # Field with clipped coordinates only in the absorbing layer
        lay_field = fire.Function(W_sp).interpolate(lay_field * layer_mask)

        return lay_field, layer_mask

    def point_cloud_field(self, parent_mesh, pts_cloud, parent_field):
        '''
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
        '''

        # Creating a point cloud field from the parent mesh
        pts_mesh = fire.VertexOnlyMesh(parent_mesh, pts_cloud, redundant=True,
                                       missing_points_behaviour='warn')
        del pts_cloud

        # Cloud field
        V0 = fire.FunctionSpace(pts_mesh, "DG", 0)
        f_int = fire.Interpolator(parent_field, V0, allow_missing_dofs=True)
        f_pts = fire.assemble(f_int.interpolate())
        del f_int

        # Ensuring correct assemble
        V1 = fire.FunctionSpace(pts_mesh.input_ordering, "DG", 0)
        del pts_mesh
        cloud_field = fire.Function(V1).interpolate(f_pts)
        del f_pts

        return cloud_field

    def extend_velocity_profile(self, lay_field, method='point_cloud'):
        '''
        Extend the velocity profile inside the absorbing layer

        Parameters
        ----------
        lay_field : `firedrake function`
            Field with clipped coordinates only in the absorbing layer
        method : `str`, optional
            Method to extend the velocity profile. Options:
            'point_cloud' or 'nearest_point'. Default is 'point_cloud'

        Returns
        -------
        None
        '''

        print("Extending Profile Inside Layer")

        # Extracting the nodes from the layer field
        lay_nodes = lay_field.dat.data_with_halos[:]

        # Nodes to extend the velocity model
        if self.dimension == 2:  # 2D
            ind_nodes = np.where(~((abs(lay_nodes[:, 0]) == 0.)
                                   & (abs(lay_nodes[:, 1]) == 0.)))

        if self.dimension == 3:  # 3D
            ind_nodes = np.where(~((abs(lay_nodes[:, 0]) == 0.)
                                   & (abs(lay_nodes[:, 1]) == 0.)
                                   & (abs(lay_nodes[:, 2]) == 0.)))

        pts_to_extend = lay_nodes[ind_nodes]

        if method == 'point_cloud':

            print("Using Cloud Points Method to Extend Velocity Profile")

            # Set the velocity of the nearest point on the original boundary
            vel_to_extend = self.point_cloud_field(
                self.mesh_original, pts_to_extend,
                self.initial_velocity_model).dat.data_with_halos[:]

        elif method == 'nearest_point':

            print("Using Nearest Point Method to Extend Velocity Profile")

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
        '''
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
        '''

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
