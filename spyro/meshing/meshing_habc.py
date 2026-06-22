import firedrake as fire
import numpy as np
from netgen.geom2d import SplineGeometry
from netgen.meshing import Element2D, Element3D, FaceDescriptor, Mesh, MeshPoint
from scipy.spatial import cKDTree
from spyro.domains.space import create_function_space
from spyro.meshing.meshing_functions import AutomaticMesh
from spyro.meshing.meshing_operations import MeshOps
from spyro.tools.habc_tools import point_cloud_field
from spyro.utils.error_management import value_parameter_error
from ..tools.version_control import is_firedrake_new


if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate
    fire.interpolate = interpolate


# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender and Romildo Soares Jr


class HABCMesh(MeshOps):
    """Class for HABC mesh generation.

    Attributes
    ----------
    comm : object
        An object representing the communication interface for parallel processing.
        Default is `None`.
    coord_bnd_nodes : `array`
        Mesh node coordinates on boundaries of the original domain.
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D.
    domain_dim : `tuple`
        Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D.
    func_space_type, `str`
        Type of function space for the state variable.
        Options: 'scalar' or 'vector'. Default is `None`.
    lmin : `float`
        Minimum mesh size
    quadrilateral : bool
        Flag to indicate whether to use quadrilateral/hexahedral elements

    Methods
    -------
    bnd_pnts_hyp_2D()
        Generate points on the boundary of a hyperellipse.
    build_hyp_mesh_3D()
        Build a hyperellipsoidal mesh from a box mesh by snapping the boundary.
    create_bnd_mesh_2D()
        Generate the boundary segment curves for the hyperellipse boundary mesh.
    create_hyp_trunc_mesh_2D()
        Generate the mesh for the hyperelliptical absorbing layer.
    get_spatial_coordinates_habc()
        Get the ufl coordinates of the mesh with absorbing layer.
    hypershape_mesh_habc()
        Generate a mesh with a hypershape absorbing layer.
    inside_hyp_3D()
        Check if a point is inside a hyperellipsoid.
    layer_boundary_data()
        Generate the boundary data from the domain with the absorbing layer.
    merge_mesh_2D()
        Merge the rectangular and the hyperelliptical meshes.
    original_boundary_data()
        Generate the boundary data from the original domain mesh.
    preamble_mesh_operations()
        Perform mesh operations previous to size an absorbing layer.
    properties_eik_mesh()
        Set the properties for the mesh used to solve the Eikonal equation.
    radial_project_on_hyp_3D()
        Project a point radially onto the hyperellipsoid surface.
    sharp_mesh_3D()
        Generate a sharp mesh by cutting the rectangular mesh.
        with the hyperellipsoid surface
    snap_nodes_to_hyp()
        Snap boundary nodes of a sharp mesh to the hyperellipsoid surface.
    trunc_hyp_bndpts_2D()
        Generate the boundary points for a truncated hyperellipse.
    """

    def __init__(self, domain_dim, dimension=2, quadrilateral=False,
                 func_space_type=None, comm=None):
        """Initialize the HABCMesh class.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (length_z, length_x) for 2D
            or (length_z, length_x, length_y) for 3D.
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        quadrilateral : bool, optional
            Flag to indicate whether to use quadrilateral/hexahedral elements.
        func_space_type, `str`, optional
            Type of function space for the state variable.
            Options: 'scalar' or 'vector'. Default is `None`
        comm : `object`, optional
            An object representing the communication interface
            for parallel processing. Default is `None`

        Returns
        -------
        None
        """

        MeshOps.__init__(self, domain_dim, dimension=dimension,
                         quadrilateral=quadrilateral,
                         func_space_type=func_space_type, comm=comm)

    def original_boundary_data(self, mesh, function_space, mesh_parameters,
                               initial_velocity_model):
        """Generate the boundary data from the original domain mesh.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh.
        function_space : `Firedrake.FunctionSpace`
            Function space for the current mesh operations.
        mesh_parameters : `meshing_parameters.MeshingParameters`
            Contains mesh parameters
        initial_velocity_model : `Firedrake.Function`
            Initial velocity model.

        Returns
        -------
        c_bnd_min : `float`
            Minimum velocity value on the boundary of the original domain.
        c_bnd_max : `float`
            Maximum velocity value on the boundary of the original domain.
        coord_bnd_nodes : `array`
            Mesh node coordinates on boundaries of the original domain.
        """

        print("Getting Boundary Mesh Data from Original Domain", flush=True)

        # Extract node positions
        node_positions = self.extract_node_positions(mesh, function_space,
                                                     output_type="array")

        # Extract boundary node positions
        all_bnd_nodes = []
        for (bnd_ids, status) in mesh_parameters.boundary_nodes_ids.values():
            if status:
                all_bnd_nodes.append(bnd_ids)
        all_bnd_nodes = np.unique(np.concatenate(all_bnd_nodes))
        coord_msh = mesh.coordinates.dat.data_with_halos
        coord_bnd_nodes = node_positions[all_bnd_nodes, :]

        # Identify the boundary nodes
        tree = cKDTree(coord_msh)
        indices = tree.query(coord_bnd_nodes, k=1,
                             distance_upper_bound=mesh_parameters.tol)[1]
        mask_boundary = indices[indices < len(coord_msh)]

        # Create a point cloud to get the extreme velocity values on the boundary
        ptos_bnd = mesh.coordinates.dat.data_with_halos[mask_boundary, :]
        vel_on_boundary = point_cloud_field(mesh, ptos_bnd, initial_velocity_model,
                                            mesh_parameters.tol).dat.data_with_halos[:]

        # Get extreme values of the velocity on the boundary excluding free surfaces
        decimal = int(abs(np.log10(mesh_parameters.tol)))
        c_bnd_min = round(vel_on_boundary[vel_on_boundary > 0.].min(), decimal)
        c_bnd_max = round(vel_on_boundary[vel_on_boundary > 0.].max(), decimal)

        # Print on screen
        cbnd_str = "Boundary Velocity Range (km/s): {:.3f} - {:.3f}"
        print(cbnd_str.format(c_bnd_min, c_bnd_max), flush=True)

        return c_bnd_min, c_bnd_max, coord_bnd_nodes

    def creating_velocity_profile(self, function_space,
                                  initial_velocity_model, path_save):
        """Create the velocity profile for the original domain.

        Parameters
        ----------
        function_space : `Firedrake.FunctionSpace`
            Function space for the current mesh operations.
        initial_velocity_model : `Firedrake.Function`
            Initial velocity model.
        path_save : `str`
            Path to save the velocity model.

        Returns
        -------
        c : `Firedrake.Function`
            Velocity profile for the original domain.
        c_min : `float`
            Minimum velocity value in the model without absorbing layer.
        c_max : `float`
            Maximum velocity value in the model without absorbing layer.
        """

        # Velocity profile model
        c = fire.Function(function_space, name='c_orig [km/s])')
        c.assign(fire.assemble(fire.interpolate(initial_velocity_model,
                                                function_space)))

        # Get extreme values of the velocity model
        c_min = initial_velocity_model.dat.data_with_halos.min()
        c_max = initial_velocity_model.dat.data_with_halos.max()

        # Print on screen
        cdom_str = "Domain Velocity Range (km/s): {:.3f} - {:.3f}"
        print(cdom_str.format(c_min, c_max), flush=True)

        # Save initial velocity model
        vel_c = fire.VTKFile(path_save + "preamble/c_vel.pvd")
        vel_c.write(c)

        return c, c_min, c_max

    def create_function_space_eik(self, mesh, degree_eik, ele_type_eik='consistent'):
        """Create the function space for the Eikonal equation modeling.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh.
        degree_eik : `int`
            Finite element order for the Eikonal modeling.
        ele_type_eik : `string`, optional
            Finite element type. 'consistent' or 'underintegrated'.
            Default is 'consistent'

        Returns
        -------
        funct_space_eik: `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        """

        print("Setting Mesh Properties for Eikonal Analysis", flush=True)

        allowed_ele_types = ['consistent', 'underintegrated']
        if ele_type_eik not in allowed_ele_types:
            value_parameter_error('ele_type_eik', ele_type_eik, allowed_ele_types)

        # Function space for the Eikonal modeling
        if ele_type_eik == 'consistent':
            funct_space_eik = create_function_space(mesh, 'CG', degree_eik)

        if ele_type_eik == 'underintegrated':
            method = 'spectral_quadrilateral' if self.quadrilateral \
                else 'mass_lumped_triangle'
            degree = min(degree_eik, 4 if self.dimension == 2 else 3)
            funct_space_eik = create_function_space(mesh, method, degree)

        return funct_space_eik

    def preamble_mesh_operations(self, Wave, ele_type_eik='consistent', f_est=0.03):
        """Perform mesh operations previous to size an absorbing layer.

        Parameters
        ----------
        Wave : `wave.Wave`
            An instance of the :class:`~spyro.solvers.wave.Wave` with attributes:
            abc_deg_eikonal : `int`
                Finite element order for the Eikonal analysis.
            function_space : `Firedrake.FunctionSpace`
                Function space for the current mesh operations.
            initial_velocity_model: `Firedrake.Function`
                Initial velocity model.
            mesh : `Firedrake.Mesh`
                Current mesh.
            mesh_parameters : `meshing_parameters.MeshingParameters`
                Contains mesh parameters.
        ele_type : `string`, optional
            Finite element type. 'consistent' or 'underintegrated'.
            Default is 'consistent'.
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03.

        Returns
        -------
        c_bnd_min : `float`
            Minimum velocity value on the boundary of the original domain.
        c_bnd_max : `float`
            Maximum velocity value on the boundary of the original domain.

        Notes
        -----
        New attributes added to the wave object in mesh_parameters:
        mesh_original : `Firedrake.Mesh`
            Original mesh without absorbing layer.
        mesh_parameters.degree_eik : `int`
            Finite element order for the Eikonal modeling.
        mesh_parameters.ele_type_eik : `string`
            Finite element type for the Eikonal modeling. 'CG' or 'KMV'.
        mesh_parameters.f_est : `float`
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03.
        mesh_parameters.funct_space_eik: `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        """

        print("\nCreating Mesh and Initial Velocity Model", flush=True)

        # Mesh data
        print(f"Original Mesh with {Wave.mesh.num_vertices()} Nodes "
              f"and {Wave.mesh.num_cells()} Volume Elements", flush=True)

        # Save a copy of the original mesh
        Wave.mesh_original = Wave.mesh
        mesh_orig = fire.VTKFile(Wave.path_save + "preamble/mesh_orig.pvd")
        mesh_orig.write(Wave.mesh_original)

        # Velocity profile model
        Wave.c, Wave.c_min, Wave.c_max = self.creating_velocity_profile(
            Wave.function_space, Wave.initial_velocity_model, Wave.path_save)

        # Generating boundary data from the original domain mesh
        Wave.c_bnd_min, Wave.c_bnd_max, \
            self.coord_bnd_nodes = self.original_boundary_data(
                Wave.mesh, Wave.function_space,
                Wave.mesh_parameters, Wave.initial_velocity_model)

        # Setting the properties of the mesh used to solve the Eikonal equation
        Wave.mesh_parameters.degree_eik = Wave.degree if not hasattr(
            Wave, 'abc_deg_eikonal') else Wave.abc_deg_eikonal
        Wave.mesh_parameters.ele_type_eik = ele_type_eik

        # Factor for the stabilizing term in Eikonal equation
        Wave.mesh_parameters.f_est = f_est

        # Function space for Eikonal modeling
        Wave.mesh_parameters.funct_space_eik = self.create_function_space_eik(
            Wave.mesh, Wave.mesh_parameters.degree_eik, ele_type_eik=ele_type_eik)

    @staticmethod
    def bnd_pnts_hyp_2D(a, b, n, num_pts):
        """Generate points on the boundary of a hyperellipse.

        Parameters
        ----------
        a : `float`
            Hyperellipse semi-axis in direction 1.
        b : `float`
            Hyperellipse semi-axis in direction 2.
        n : `int`
            Degree of the hyperellipse.
        num_pts : `int`
            Number of points to generate on the hyperellipse boundary.

        Returns
        -------
        bnd_pnts : `array`
            Array of shape (num_pts, 2) containing the coordinates
            of the hyperellipse boundary points.
        """

        # Generate angle values for the parametric equations
        theta = np.linspace(0., 2. * np.pi, num_pts)

        # Especial angle values
        rc_zero = [np.pi / 2., 3. * np.pi / 2.]
        rs_zero = [0., np.pi, 2. * np.pi]

        # Trigonometric function evaluation
        cr = np.cos(theta)
        sr = np.sin(theta)
        cr = np.where(np.isin(theta, rc_zero), 0., cr)
        sr = np.where(np.isin(theta, rs_zero), 0., sr)

        # Parametric equations for the hyperellipse
        x = a * np.sign(cr) * np.abs(cr)**(2. / n)
        y = b * np.sign(sr) * np.abs(sr)**(2. / n)

        bnd_pnts = np.column_stack((x, y))

        return bnd_pnts

    def trunc_hyp_bndpts_2D(self, hyp_par, xdom, z0):
        """Generate the boundary points for a truncated hyperellipse.

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipse parameters.
            Structure: (n_hyp, perimeter, a_hyp, b_hyp)
            - n_hyp : `float`
                Degree of the hyperellipse.
            - perimeter : `float`
                Perimeter of the full hyperellipse (2D).
            - a_hyp : `float`
                Hyperellipse semi-axis in direction x.
            - b_hyp : `float`
                Hyperellipse semi-axis in direction z.
        xdom : `float`
            Maximum coordinate in normal to the truncation plane.
        z0 : `float`
            Truncation plane.

        Returns
        -------
        filt_bnd_pts : `array`
            Array of shape (num_bnd_pts, 2) containing the coordinates
            of the truncated hyperellipse boundary points.
        trunc_feat : `tuple`
            Truncation features.
            Structure: [ini_trunc, end_trunc, num_filt_bnd_pts, ltrunc]
            - ini_trunc : `int`
                Index of the first point after the truncation plane.
            - end_trunc : `int`
                Index of the last point before the truncation plane.
            - num_filt_bnd_pts : `int`
                Number of filtered boundary points.
            - ltrunc : `float`
                Mesh size for arc length due to the truncation operation.
        """

        # Hyperellipse parameters
        n_hyp, perimeter, a_hyp, b_hyp = hyp_par

        # Boundary points: Use 16 or 24 as a minimum
        num_bnd_pts = int(max(np.ceil(perimeter / self.lmin), 16)) - 1

        # Generate the hyperellipse boundary points
        pnt_bef_trunc = 0
        pnt_aft_trunc = 0
        pnt_str = "Number of Boundary Points for"
        while pnt_bef_trunc % 2 == 0 or pnt_aft_trunc % 2 == 0 \
                or pnt_bef_trunc < 3 or pnt_aft_trunc < 3:

            num_bnd_pts += 1
            print(pnt_str, f"Complete Hyperellipse: {num_bnd_pts}", flush=True)
            bnd_pts = self.bnd_pnts_hyp_2D(a_hyp, b_hyp, n_hyp, num_bnd_pts)

            # Filter hyperellipse points based on the truncation plane z0
            filt_bnd_pts = np.array([point for point in bnd_pts
                                     if point[1] <= z0])
            print(pnt_str, f"Truncated Hyperellipse: {len(filt_bnd_pts)}",
                  flush=True)

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
        """Generate the boundary segments for the hyperellipse boundary mesh.

        Parameters
        ----------
        geo : `SplineGeometry`
            Geometry object with the data to generate the mesh.
        bnd_pts : `array`
            Array of shape (num_bnd_pts, 2) containing the coordinates
            of the hyperellipse boundary points.
        trunc_feat : `tuple`
            Truncation features.
            Structure: [ini_trunc, end_trunc, num_filt_bnd_pts, ltrunc]
            - ini_trunc : `int`
                Index of the first point after the truncation plane.
            - end_trunc : `int`
                Index of the last point before the truncation plane.
            - num_filt_bnd_pts : `int`
                Number of filtered boundary points.
            - ltrunc : `float`
                Mesh size for arc length due to the truncation operation.
        spln : `bool`
            Flag to indicate whether to use splines (True) or lines (False)

        Returns
        -------
        curves : `list`
            List of curves to be added to the geometry object. Each curve is
            represented as a list containing the curve type and its points.
        """

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

    def create_hyp_trunc_mesh_2D(self, hyp_par, spln=True):
        """
        Generate the mesh for the hyperelliptical absorbing layer

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipse parameters.
            Structure: (n_hyp, perimeter, a_hyp, b_hyp)
            - n_hyp : `float`
                Degree of the hyperellipse.
            - perimeter : `float`
                Perimeter of the full hyperellipse (2D).
            - a_hyp : `float`
                Hyperellipse semi-axis in direction x.
            - b_hyp : `float`
                Hyperellipse semi-axis in direction z.
        spln : `bool`, optional
            Flag to indicate whether to use splines (`True`) or lines (`False`).
            Default is `True`.

        Returns
        -------
        hyp_mesh : `netgen mesh`
            Generated mesh for the hyperelliptical layer.
        """

        # Domain dimensions
        Lx, Lz = self.domain_dim[:2]

        # Generate the hyperellipse boundary points
        bnd_pts, trunc_feat = self.trunc_hyp_bndpts_2D(hyp_par, Lx / 2, Lz / 2)

        # Initialize geometry
        geo = SplineGeometry()

        # Append points to the geometry
        [geo.AppendPoint(*pnt) for pnt in bnd_pts]

        while True:
            try:
                # Generate the boundary segment curves
                curves = self.create_bnd_mesh_2D(geo, bnd_pts,
                                                 trunc_feat, spln)
                [geo.Append(c[:-1], bc="outer", maxh=c[-1], leftdomain=1,
                            rightdomain=0) for c in curves]

                # Generate the mesh using netgen library
                hyp_mesh = geo.GenerateMesh(maxh=self.lmin,
                                            quad_dominated=self.quadrilateral,
                                            optsteps2d=10,  # Optimize mesh
                                            )
                hyp_mesh.Compress()
                print("Hyperelliptical Mesh Generated Successfully",
                      flush=True)
                break

            except Exception as e:

                # Retry with lines if splines fail
                if spln:
                    print(f"Error Meshing with Splines: {e}", flush=True)
                    print("Now meshing with Lines", flush=True)
                    spln = False

                else:
                    print(f"Error Meshing with Lines: {e}. Exiting.",
                          flush=True)
                    break

        # Mesh is transformed into a firedrake mesh
        q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
        hyp_mesh = fire.Mesh(
            hyp_mesh, distribution_parameters=q, comm=self.comm.comm)

        # Adjusting coordinates: Swap (x,z) -> (z,x) and apply offsets
        hyp_mesh.coordinates.dat.data_with_halos[:, [0, 1]] = \
            hyp_mesh.coordinates.dat.data_with_halos[:, [1, 0]]
        hyp_mesh.coordinates.dat.data_with_halos[:, 0] -= Lz / 2
        hyp_mesh.coordinates.dat.data_with_halos[:, 1] += Lx / 2

        # Forcing node at (0,0)
        err_m = np.sqrt(hyp_mesh.coordinates.dat.data_with_halos[:, 0]**2
                        + hyp_mesh.coordinates.dat.data_with_halos[:, 1]**2)
        if err_m.min() > 0.:
            id_min = err_m.argmin()
            err_z = -hyp_mesh.coordinates.dat.data_with_halos[:, 0][id_min]
            err_x = -hyp_mesh.coordinates.dat.data_with_halos[:, 1][id_min]
            hyp_mesh.coordinates.dat.data_with_halos[:, 0] += err_z
            hyp_mesh.coordinates.dat.data_with_halos[:, 1] += err_x

        return hyp_mesh

    def merge_mesh_2D(self, rec_mesh, hyp_mesh):
        """Merge the rectangular and the hyperelliptical meshes.

        Parameters
        ----------
        rec_mesh : `Firedrake.Mesh`
            Rectangular mesh representing the original domain.
        hyp_mesh : `Firedrake.Mesh`
            Hyperelliptical annular mesh representing the absorbing layer.

        Returns
        -------
        final_mesh : `Firedrake.Mesh`
            Merged final mesh.
        """

        # Create the final mesh that will contain both
        final_mesh = Mesh()
        final_mesh.dim = self.dimension

        # Get coordinates of the rectangular mesh
        coord_rec = rec_mesh.coordinates.dat.data_with_halos

        # Create KDTree for efficient nearest neighbor search
        boundary_tree = cKDTree(self.coord_bnd_nodes)

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
                  f"and {len(final_mesh.Elements2D())} elements", flush=True)

            # Mesh is transformed into a firedrake mesh
            q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
            final_mesh = fire.Mesh(
                final_mesh, distribution_parameters=q, comm=self.comm.comm)
            print("Merged Mesh Generated Successfully", flush=True)

        except Exception as e:
            print(f"Error Generating Merged Mesh: {e}. Exiting.", flush=True)

        return final_mesh

    @staticmethod
    def inside_hyp_3D(pnt, a, b, c, n):
        """Check if a point is inside a hyperellipsoid.

        Parameters
        ----------
        pnt : `array`
            Point to be checked (x, y, z).
        a : `float`
            Hyperellipsoid semi-axis in direction 1.
        b : `float`
            Hyperellipsoid semi-axis in direction 2.
        c : `float`
            Hyperellipsoid semi-axis in direction 3.
        n : `float`
            Degree of the hyperellipsoid.

        Returns
        -------
        in_hyp : `bool`
            `True` if the point is inside the hyperellipsoid, `False` otherwise
        """

        # Evaluate hyperellipsoid equation
        x, y, z = pnt
        in_hyp = abs(x / a) ** n + abs(y / b) ** n + abs(z / c) ** n <= 1.1

        return in_hyp

    def sharp_mesh_3D(self, rec_mesh, hyp_par, centroid):
        """Generate a sharp mesh by cutting with a hyperellipsoid surface.

        The rectangular mesh is cut with the hyperellipsoid surface.

        Parameters
        ----------
        rec_mesh : `Firedrake.Mesh`
            Rectangular mesh with an absorbing layer.
        hyp_par : `tuple`
            Hyperellipsoid parameters.
            Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hyperellipsoid.
            - surface : `float`
                Surface area of the full hyperellipsoid (3D).
            - a_hyp : `float`
                Hyperellipsoid semi-axis in direction x.
            - b_hyp : `float`
                Hyperellipsoid semi-axis in direction z.
            - c_hyp : `float`
                Hyperellipsoid semi-axis in direction y.
        centroid : `array`
            Centroid of the full hyperellipsoid (z0, x0, y0).

        Returns
        -------
        sharp_mesh : `Firedrake.Mesh`
            Generated sharp mesh.
        """

        # Hyperellipsoid parameters
        n_hyp, _, a_hyp, b_hyp, c_hyp = hyp_par

        # Create the final mesh that will contain both
        sharp_mesh = Mesh()
        sharp_mesh.dim = 3

        # Get coordinates of the rectangular mesh
        coord_rec = rec_mesh.coordinates.dat.data_with_halos[:]
        # fire.VTKFile("output/base_mesh.pvd").write(rec_mesh)

        # Add all vertices from rectangular mesh and create mapping
        rec_map = {}
        for i, coord in enumerate(coord_rec):
            z, x, y = coord
            if self.inside_hyp_3D(coord - centroid,
                                  b_hyp, a_hyp, c_hyp, n_hyp):
                rec_map[i] = sharp_mesh.Add(MeshPoint((z, x, y)))

        # Face descriptor for the rectangular mesh
        fd_rec = sharp_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=0))

        # Get mesh cells from rectangular mesh
        rec_cells = rec_mesh.coordinates.cell_node_map().values_with_halo

        # Add all elements from rectangular mesh to the netgen mesh
        sharp_mesh.SetMaterial(1, "rec")
        for cell in rec_cells:
            netgen_points = [rec_map.get(cell[i]) for i in range(len(cell))]
            if not any(p is None for p in netgen_points):
                sharp_mesh.Add(Element3D(fd_rec, netgen_points))

        try:
            # Mesh data
            sharp_mesh.Compress()
            print(f"Mesh created with {len(sharp_mesh.Points())} points "
                  f"and {len(sharp_mesh.Elements3D())} elements", flush=True)

            # Mesh is transformed into a firedrake mesh
            q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
            sharp_mesh = fire.Mesh(
                sharp_mesh, distribution_parameters=q, comm=self.comm.comm)
            print("Sharp Mesh Generated Successfully", flush=True)
            # fire.VTKFile("output/sharp_mesh.pvd").write(sharp_mesh)

        except Exception as e:
            print(f"Error Generating Merged Mesh: {e}. Exiting.", flush=True)

        return sharp_mesh

    @staticmethod
    def radial_project_on_hyp_3D(p_to_snap, centroid, a, b, c, n):
        """Project a point radially onto the hyperellipsoid surface.

        Parameters
        ----------
        p_to_snap : `array`
            Point to be projected (x, y, z).
        centroid : `array`
            Centroid of the full hyperellipsoid (x0, y0, z0).
        a : `float`
            Hyperellipsoid semi-axis in direction 1.
        b : `float`
            Hyperellipsoid semi-axis in direction 2.
        c : `float`
            Hyperellipsoid semi-axis in direction 3.
        n : `float`
            Degree of the hyperellipsoid.

        Returns
        -------
        q_snapped : `array`
            Projected point (x', y', z').
        """

        # Vector from centroid to the point
        d = p_to_snap - centroid
        if np.allclose(d, 0.):
            return p_to_snap.copy()

        # Compute scaling factor
        val = (abs(d[0]/a)**n + abs(d[1]/b)**n + abs(d[2]/c)**n)
        if val <= 0.:
            return p_to_snap.copy()
        s = (1. / val)**(1. / n)

        # Compute snapped point
        q_snapped = centroid + d * s

        return q_snapped

    def snap_nodes_to_hyp(self, mesh, hyp_par, centroid, plane_tol=1e-5):
        """Snap boundary nodes of a sharp mesh to the hyperellipsoid surface.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Mesh to be modified.
        hyp_par : `tuple`
            Hyperellipsoid parameters.
            Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hyperellipsoid
            - surface : `float`
                Surface area of the full hyperellipsoid (3D).
            - a_hyp : `float`
                Hyperellipsoid semi-axis in direction x.
            - b_hyp : `float`
                Hyperellipsoid semi-axis in direction z.
            - c_hyp : `float`
                Hyperellipsoid semi-axis in direction y.
        centroid : `array`
            Centroid of the full hyperellipsoid (z0, x0, y0).
        plane_tol : `float`, optional
            Tolerance to identify boundary planes. Default is 1e-5.

        Returns
        -------
        mesh : `Firedrake.Mesh`
            Modified mesh with snapped boundary nodes.
        """

        # Hyperellipsoid parameters
        n_hyp, _, a_hyp, b_hyp, c_hyp = hyp_par

        # Auto-detect bounds
        coords = mesh.coordinates.dat.data_with_halos
        min_z, min_x, min_y = np.min(coords, axis=0)
        max_z, max_x, max_y = np.max(coords, axis=0)
        print("Mesh Bounds Detected:", flush=True)
        print(f"       X: [{min_x:.4f}, {max_x:.4f}]", flush=True)
        print(f"       Y: [{min_y:.4f}, {max_y:.4f}]", flush=True)
        print(f"       Z: [{min_z:.4f}, {max_z:.4f}]", flush=True)

        # Select nodes to snap
        mask_min_z = np.isclose(coords[:, 0], min_z, atol=plane_tol)
        mask_min_x = np.isclose(coords[:, 1], min_x, atol=plane_tol)
        mask_max_x = np.isclose(coords[:, 1], max_x, atol=plane_tol)
        mask_min_y = np.isclose(coords[:, 2], min_y, atol=plane_tol)
        mask_max_y = np.isclose(coords[:, 2], max_y, atol=plane_tol)

        # Combine masks
        mask = mask_min_x | mask_max_x | mask_min_y | mask_max_y | mask_min_z

        # Snap boundary nodes
        pnts_to_snap = np.where(mask)[0]
        for pnt in pnts_to_snap:
            coords[pnt, :] = self.radial_project_on_hyp_3D(
                coords[pnt, :], centroid, b_hyp, a_hyp, c_hyp, n_hyp)
            coords[pnt, 0] = np.clip(coords[pnt, 0], -np.inf, max_z)

        print(f"Boundary Nodes Snapped: {len(pnts_to_snap)}", flush=True)
        print("Snapped Mesh Generated Successfully", flush=True)

        return mesh

    def build_hyp_mesh_3D(self, rec_mesh, hyp_par, plane_tol=1e-5):
        """Build a hyperellipsoidal mesh from a box mesh by snapping the boundary.

        Parameters
        ----------
        rec_mesh : `Firedrake.Mesh`
            Rectangular mesh with an absorbing layer.
        hyp_par : `tuple`
            Hyperellipsoid parameters.
            Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hyperellipsoid
            - surface : `float`
                Surface area of the full hyperellipsoid (3D).
            - a_hyp : `float`
                Hyperellipsoid semi-axis in direction x.
            - b_hyp : `float`
                Hyperellipsoid semi-axis in direction z.
            - c_hyp : `float`
                Hyperellipsoid semi-axis in direction y.
        plane_tol : `float`, optional
            Tolerance to identify boundary planes. Default is 1e-5.

        Returns
        -------
        final_mesh : `Firedrake.Mesh`
            Merged final mesh
        """

        # Original domain dimensions
        Lx, Lz, Ly = self.domain_dim

        # Centroid of the hyperellipsoid
        centroid = np.array([-Lz / 2, Lx / 2., Ly / 2.])

        # Generate sharp mesh
        sharp_mesh = self.sharp_mesh_3D(rec_mesh, hyp_par, centroid)

        # Snap boundary nodes to the hyperellipsoid
        final_mesh = self.snap_nodes_to_hyp(sharp_mesh, hyp_par, centroid)
        del sharp_mesh

        return final_mesh

    def hypershape_mesh_habc(self, hyp_par, mesh_original, mesh_parameters, spln=True):
        """Generate a mesh with a hypershape absorbing layer.

        Parameters
        ----------
        hyp_par : `tuple`
            Hyperellipshape parameters.
            Structure 2D: (n_hyp, perimeter, a_hyp, b_hyp).
            Structure 3D: (n_hyp, surface, a_hyp, b_hyp, c_hyp).
            - n_hyp : `float`
                Degree of the hypershape.
            - perimeter : `float`
                Perimeter of the full hyperellipse (2D).
            - surface : `float`
                Surface area of the full hyperellipsoid (3D).
            - a_hyp : `float`
                Hypershape semi-axis in direction x.
            - b_hyp : `float`
                Hypershape semi-axis in direction z.
            - c_hyp : `float`
                Hypershape semi-axis in direction y (3D only).
        mesh_original : `Firedrake.Mesh`
            Original mesh without absorbing layer.
        mesh_parameters : `meshing_parameters.MeshingParameters`
            Contains mesh parameters:
            lmin : `float`
                Minimum mesh size.
            tol : `float`
                Tolerance for searching nodes in the mesh.
        spln : `bool`, optional
            Flag to indicate whether to use splines (`True`) or lines (`False`)
            in hypershape layer generation. Default is `True`.

        Returns
        -------
        mesh_habc : `Firedrake.Mesh`
            Mesh with a hypershape absorbing layer.
        """

        # Get the mesh parameters for use in hypershape mesh generation
        self.lmin = mesh_parameters.lmin
        self.tol = mesh_parameters.tol

        if self.dimension == 2:  # 2D

            # Creating the hyperellipse layer mesh
            hyp_mesh = self.create_hyp_trunc_mesh_2D(hyp_par[:4], spln=spln)
            # fire.VTKFile("output/trunc_hyp_test.pvd").write(hyp_mesh)

            # Merging the original mesh with the hyperellipse layer mesh
            mesh_habc = self.merge_mesh_2D(mesh_original, hyp_mesh)
            # fire.VTKFile("output/trunc_merged_test.pvd").write(mesh_habc)

        if self.dimension == 3:  # 3D

            # Base rectangular mesh
            rec_mesh = AutomaticMesh(
                mesh_parameters=mesh_parameters).create_firedrake_mesh()
            # fire.VTKFile("output/rectang_test.pvd").write(rec_mesh)

            # Merging the original mesh with a hyperellipsoid layer mesh
            mesh_habc = self.build_hyp_mesh_3D(rec_mesh, hyp_par)
            # fire.VTKFile("output/snapped_test3d.pvd").write(mesh_habc)

        return mesh_habc

    def layer_boundary_data(self, V):
        """Generate the boundary data from the domain with the absorbing layer.

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space for the boundary of the domain with absorbing layer.

        Returns
        -------
        bnd_nfs : 'array'
            Mesh node indices on non-free surfaces.
        bnd_nodes_nfs : `tuple`
            Mesh node coordinates on non-free surfaces.
            - (z_data[nfs_idx], x_data[nfs_idx]) for 2D.
            - (z_data[nfs_idx], x_data[nfs_idx], y_data[nfs_idx]) for 3D.
        """

        # Boundary nodes indices
        bnd_nod = fire.DirichletBC(V, 0., "on_boundary").nodes

        # Extract node positions
        node_positions = self.extract_node_positions(self.mesh, V)

        # Boundary node coordinates
        z_f, x_f = node_positions[:2]
        bnd_z = z_f[bnd_nod]
        bnd_x = x_f[bnd_nod]

        # Identify non-free surfaces (remain unchanged)
        no_free_surf = ~(abs(bnd_z) <= self.mesh_parameters.tol)

        bnd_nodes_nfs = (bnd_z[no_free_surf], bnd_x[no_free_surf])
        if self.dimension == 3:  # 3D
            y_f = node_positions[2]
            bnd_y = y_f[bnd_nod]
            bnd_nodes_nfs += (bnd_y[no_free_surf],)

        # Boundary node indices on non-free surfaces
        bnd_nfs = bnd_nod[no_free_surf]

        return bnd_nfs, bnd_nodes_nfs

    def get_spatial_coordinates_abc(self, mesh, domain_layer):
        """Get the ufl coordinates of the mesh with absorbing layer.

        Parameters
        ----------
        mesh : `Firedrake.Mesh`
            Current mesh.
        domain_layer : `tuple`
            Domain dimensions with layer. For rectangular layers, truncation
            due to the free surface is included (n = 1). For hypershape layers,
            truncation by free surface is not included (n = 2) if 'full_hyp' is
            True; otherwise, it is included (n = 1). Dimensions are defined as:
            2D: (length_z + n * pad_len, length_x + 2 * pad_len).
            3D: (length_x + 2 * pad_len, length_z + n * pad_len, length_y + 2 * pad_len).

        Returns
        -------
        ufl_coordinates_abc : `ufl.geometry.SpatialCoordinate`
            Domain Coordinates including the absorbing layer.
        """

        min_coordinates, max_coordinates = self.extract_extreme_coordinates(mesh)
        domain_abc = np.asarray(domain_layer)
        domain_to_check = abs(max_coordinates - min_coordinates)

        assert np.allclose(domain_abc, domain_to_check, atol=0.01), \
            "Mesh dimensions do not match with expected dimensions of " \
            f"domain with absorbing layer. Expected: {np.round(domain_abc, 3)}, " \
            f"Got: {np.round(domain_to_check, 3)}."

        ufl_coordinates_abc = fire.SpatialCoordinate(mesh)

        return ufl_coordinates_abc
