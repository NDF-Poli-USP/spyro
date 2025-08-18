import firedrake as fire
import numpy as np
from scipy.spatial import Delaunay
import meshio


try:
    
    import SeismicMesh
    from SeismicMesh.geometry import Rectangle
    from SeismicMesh.generation import generate_mesh
    from SeismicMesh.sizing import get_sizing_function_from_segy
except ImportError:
    SeismicMesh = None


class AutomaticMesh:
    """
    Class for automatic meshing.

    Attributes
    ----------
    dimension : int
        Spatial dimension of the mesh.
    length_z : float
        Length of the domain in the z direction.
    length_x : float
        Length of the domain in the x direction.
    length_y : float
        Length of the domain in the y direction.
    dx : float
        Mesh size.
    quadrilateral : bool
        If True, the mesh is quadrilateral.
    periodic : bool
        If True, the mesh is periodic.
    comm : MPI communicator
        MPI communicator.
    mesh_type : str
        Type of the mesh.
    abc_pad : float
        Padding to be added to the domain.

    Methods
    -------
    set_mesh_size(length_z=None, length_x=None, length_y=None)
        Sets the mesh size.
    set_meshing_parameters(dx=None, cell_type=None, mesh_type=None)
        Sets the meshing parameters.
    set_seismicmesh_parameters(cpw=None, velocity_model=None, edge_length=None)
        Sets the SeismicMesh parameters.
    make_periodic()
        Sets the mesh boundaries periodic. Only works for firedrake_mesh.
    create_mesh()
        Creates the mesh.
    create_firedrake_mesh()
        Creates a mesh based on Firedrake meshing utilities.
    create_firedrake_2D_mesh()
        Creates a 2D mesh based on Firedrake meshing utilities.
    create_firedrake_3D_mesh()
        Creates a 3D mesh based on Firedrake meshing utilities.
    create_seismicmesh_mesh()
        Creates a mesh based on SeismicMesh meshing utilities.
    create_seimicmesh_2d_mesh()
        Creates a 2D mesh based on SeismicMesh meshing utilities.
    create_seismicmesh_2D_mesh_homogeneous()
        Creates a 2D mesh homogeneous velocity mesh based on SeismicMesh meshing utilities.
    """

    def __init__(
        self, mesh_parameters=None
    ):
        """
        Initialize the MeshingFunctions class.

        Parameters
        ----------
        comm : MPI communicator, optional
            MPI communicator. The default is None.
        mesh_parameters : dict, optional
            Dictionary containing the mesh parameters. The default is None.

        Raises
        ------
        ValueError
            If `abc_pad_length` is negative.

        Notes
        -----
        The `mesh_parameters` dictionary should contain the following keys:
        - 'dimension': int, optional. Dimension of the mesh. The default is 2.
        - 'length_z': float, optional. Length of the mesh in the z-direction.
        - 'length_x': float, optional. Length of the mesh in the x-direction.
        - 'length_y': float, optional. Length of the mesh in the y-direction.
        - 'cell_type': str, optional. Type of the mesh cells.
        - 'mesh_type': str, optional. Type of the mesh.

        For mesh with absorbing layer only:
        - 'abc_pad_length': float, optional. Length of the absorbing boundary condition padding.

        For Firedrake mesh only:
        - 'dx': float, optional. Mesh element size.
        - 'periodic': bool, optional. Whether the mesh is periodic.
        - 'edge_length': float, optional. Length of the mesh edges.

        For SeismicMesh only:
        - 'cells_per_wavelength': float, optional. Number of cells per wavelength.
        - 'source_frequency': float, optional. Frequency of the source.
        - 'minimum_velocity': float, optional. Minimum velocity.
        - 'velocity_model_file': str, optional. File containing the velocity model.
        - 'edge_length': float, optional. Length of the mesh edges.
        """
        self.dimension = mesh_parameters.dimension
        self.length_z = mesh_parameters.length_z
        self.length_x = mesh_parameters.length_x
        self.length_y = mesh_parameters.length_y
        self.quadrilateral = mesh_parameters.quadrilateral
        self.comm = mesh_parameters.comm
        self.mesh_type = mesh_parameters.mesh_type
        self.edge_length = mesh_parameters.edge_length
        self.abc_pad = mesh_parameters.abc_pad_length

        # Firedrake mesh only parameters

        self.periodic = mesh_parameters.periodic

        # SeismicMesh only parameters
        self.cpw = mesh_parameters.cells_per_wavelength
        self.source_frequency = mesh_parameters.source_frequency
        self.minimum_velocity = mesh_parameters.minimum_velocity
        self.lbda = None
        self.velocity_model = mesh_parameters.velocity_model
        self.padding_type = mesh_parameters.padding_type
        self.ellipse_n = mesh_parameters.ellipse_n
        self.extrapolate = mesh_parameters.extrapolate
        self.hpad = mesh_parameters.hpad
        self.transition_width = mesh_parameters.transition_width
        self.grade = mesh_parameters.grade
        self.output_file_name = mesh_parameters.output_filename

    def create_mesh(self):
        """
        Creates the mesh.

        Returns
        -------
        mesh : Mesh
            Mesh
        """
        print(f"Creating {self.mesh_type} type mesh.", flush=True)
        if self.mesh_type == "firedrake_mesh":
            return self.create_firedrake_mesh()
        elif self.mesh_type == "SeismicMesh":
            if SeismicMesh is None:
                raise ImportError("SeismicMesh is not available. Please install it to use this function.")
            return self.create_seismicmesh_mesh()
        else:
            raise ValueError("mesh_type is not supported")

    def create_firedrake_mesh(self):
        """
        Creates a mesh based on Firedrake meshing utilities.
        """
        if self.dimension == 2:
            return self.create_firedrake_2D_mesh()
        elif self.dimension == 3:
            return self.create_firedrake_3D_mesh()
        else:
            raise ValueError("dimension is not supported")

    def create_firedrake_2D_mesh(self):
        """
        Creates a 2D mesh based on Firedrake meshing utilities.
        """
        if self.edge_length is None and self.cpw is not None:
            self.edge_length = calculate_edge_length(self.cpw, self.minimum_velocity, self.source_frequency)
        if self.abc_pad:
            nx = int((self.length_x + 2*self.abc_pad) / self.edge_length)
            nz = int((self.length_z + self.abc_pad) / self.edge_length)
        else:
            nx = int(self.length_x / self.edge_length)
            nz = int(self.length_z / self.edge_length)

        comm = self.comm

        if self.periodic:
            return PeriodicRectangleMesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=self.quadrilateral,
                comm=comm.comm,
                pad=self.abc_pad,
            )
        else:
            return RectangleMesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=self.quadrilateral,
                comm=comm.comm,
                pad=self.abc_pad,
            )

    def create_firedrake_3D_mesh(self):
        """
        Creates a 3D mesh based on Firedrake meshing utilities.
        """
        dx = self.edge_length
        nx = int(self.length_x / dx)
        nz = int(self.length_z / dx)
        ny = int(self.length_y / dx)

        return BoxMesh(
            nz,
            nx,
            ny,
            self.length_z,
            self.length_x,
            self.length_y,
            quadrilateral=self.quadrilateral,
        )

    def create_seismicmesh_mesh(self):
        """
        Creates a mesh based on SeismicMesh meshing utilities.

        Returns
        -------
        mesh : Mesh
            Mesh
        """
        if self.dimension == 2:
            return self.create_seimicmesh_2d_mesh()
        elif self.dimension == 3:
            raise NotImplementedError("Not implemented yet")
            # return self.create_seismicmesh_3D_mesh()
        else:
            raise ValueError("dimension is not supported")

    def create_seimicmesh_2d_mesh(self):
        """
        Creates a 2D mesh based on SeismicMesh meshing utilities.
        """
        print(f"velocity_model{self.velocity_model}", flush=True)
        if self.velocity_model is None:
            return self.create_seismicmesh_2D_mesh_homogeneous()
        else:
            return self.create_seismicmesh_2D_mesh_with_velocity_model()

    def create_seismicmesh_2D_mesh_with_velocity_model(self):
        if self.comm.ensemble_comm.rank == 0:
            v_min = self.minimum_velocity
            frequency = self.source_frequency
            padding_type = self.padding_type
            C = self.cpw
            Lz = self.length_z
            Lx = self.length_x
            domain_pad = self.abc_pad
            lbda_min = v_min/frequency
            bbox = (-Lz, 0.0, 0.0, Lx)
            domain = SeismicMesh.geometry.Rectangle(bbox)
            hmin = lbda_min/C
            
            if padding_type == "rectangular":
            
                self.comm.comm.barrier()
    
                ef = SeismicMesh.sizing.get_sizing_function_from_segy(
                    self.velocity_model,
                    bbox,
                    hmin=hmin,
                    wl=C,
                    freq=frequency,
                    grade=self.grade,
                    domain_pad=domain_pad,
                    pad_style="edge",
                    units='km/s',
                    comm=self.comm.comm,
                )
                self.comm.comm.barrier()
    
                # Creating rectangular mesh
                points, cells = SeismicMesh.generation.generate_mesh(
                    domain=domain,
                    edge_length=ef,
                    verbose=0,
                    mesh_improvement=False,
                    comm=self.comm.comm,
                )
                self.comm.comm.barrier()
    
                print('entering spatial rank 0 after mesh generation')
                if self.comm.comm.rank == 0:
                    meshio.write_points_cells(
                        "automatic_mesh.msh",
                        points,
                        [("triangle", cells)],
                        file_format="gmsh22",
                        binary=False
                    )
    
                    meshio.write_points_cells(
                        "automatic_mesh.vtk",
                        points,
                        [("triangle", cells)],
                        file_format="vtk"
                    )

            if padding_type == None:
            
                self.comm.comm.barrier()
    
                ef = SeismicMesh.sizing.get_sizing_function_from_segy(
                    self.velocity_model,
                    bbox,
                    hmin=hmin,
                    wl=C,
                    freq=frequency,
                    grade=self.grade,
                    units='km/s',
                    comm=self.comm.comm,
                )
                self.comm.comm.barrier()
    
                # Creating rectangular mesh
                points, cells = SeismicMesh.generation.generate_mesh(
                    domain=domain,
                    edge_length=ef,
                    verbose=0,
                    mesh_improvement=False,
                    comm=self.comm.comm,
                )
                self.comm.comm.barrier()
    
                print('entering spatial rank 0 after mesh generation')
                if self.comm.comm.rank == 0:
                    meshio.write_points_cells(
                        "automatic_mesh.msh",
                        points,
                        [("triangle", cells)],
                        file_format="gmsh22",
                        binary=False
                    )
    
                    meshio.write_points_cells(
                        "automatic_mesh.vtk",
                        points,
                        [("triangle", cells)],
                        file_format="vtk"
                    )            
            if padding_type == "elliptical":   
                
                # Ellipse exponent
                ellipse_n = self.ellipse_n
                # Element size for the rest of the domain
                h_default = self.hpad
                # Segy function extrapolation beyond rectangular domain 
                extrapolate = self.extrapolate
                # Rectangle center 
                zc = -Lz / 2.0     
                xc = Lx / 2.0     
                # Ellipse semi-axes 
                ellipse_a = (-Lz / 2.0) - domain_pad  
                ellipse_b = (Lx / 2.0) + domain_pad  
                # Domain bounding box 
                domain_zmin = zc + ellipse_a
                domain_zmax =  0.0
                domain_xmin = xc - ellipse_b 
                domain_xmax = xc + ellipse_b 
                d_bbox = (domain_zmin, domain_zmax, domain_xmin, domain_xmax)
                
                
                ef_segy = SeismicMesh.sizing.get_sizing_function_from_segy(
                    self.velocity_model,
                    bbox,
                    hmin=hmin,
                    wl=C,
                    freq=frequency,
                    grade=self.grade,
                )
                
                # Finding minimum value of segy for mesh normalization
                approx_resolution = 2000 # Resolution for approximation
                x_vals = np.linspace(0.0, Lx, approx_resolution)
                z_vals = np.linspace(-Lz, 0.0, approx_resolution)
                X, Z = np.meshgrid(x_vals, z_vals)
                points = np.column_stack([Z.ravel(), X.ravel()])
                values = ef_segy.eval(points)
                min_value = np.min(values)
                
                # Fixed Points around interior rectangular subdomain
                fixed_points = []
                fixed_points = place_horizontal_boundary(fixed_points, 0.0, Lx, -Lz, ef_segy, approx_resolution,hmin/min_value)
                fixed_points = place_vertical_boundary(fixed_points, -Lz, 0.0, 0.0, ef_segy, approx_resolution, hmin/min_value)
                fixed_points = place_vertical_boundary(fixed_points, -Lz, 0.0, Lx, ef_segy, approx_resolution, hmin/min_value)
                
                rectangle_subdomain = SeismicMesh.geometry.Rectangle((-Lz, 0.0, 0.0, Lx))

                if extrapolate: 
                    #if extrapolate, create an extended function for the ellipse domain
                    ef_segy_extended = create_padded_segy_function(
                    ef_segy, 
                    -Lz, 0.0, 0.0, Lx,
                    domain_zmin, domain_zmax, domain_xmin, domain_xmax,
                    nz_original=approx_resolution, nx_original=approx_resolution, 
                    pad_mode='edge' ,
                    )
                
                else: 
                    ef_segy_extended = create_padded_segy_function(
                    ef_segy, 
                    -Lz, 0.0, 0.0, Lx,
                    domain_zmin, domain_zmax, domain_xmin, domain_xmax,
                    nz_original=approx_resolution, nx_original=approx_resolution,  # Resolution for aproximation
                    pad_mode='constant' ,
                    pad_size=self.hpad,
                    transition_width=self.transition_width,    
                    )
                    
                def unified_sizing_function_extrapolated(p):
                    #Points outside the rectangle, use the extended function
                    sizes = ef_segy_extended(p)
                    inside_rect = ((p[:, 0] >= -Lz) & (p[:, 0] <= 0.0) & 
                               (p[:, 1] >= 0.0) & (p[:, 1] <= Lx))
                    
                    if np.any(inside_rect):
                        # For points in rectangle, use original SEGY
                        rect_points = p[inside_rect]
                        segy_sizes = ef_segy.eval(rect_points)
                        sizes[inside_rect] = segy_sizes
                            
                    return sizes
                
                # Generate mesh
                points, cells = SeismicMesh.generation.generate_mesh(
                    domain=lambda p: domain_sdf(p, a=ellipse_a, b=ellipse_b, n=ellipse_n, zc=zc, xc=xc),
                    edge_length=unified_sizing_function_extrapolated,
                    bbox=d_bbox,
                    subdomains=[rectangle_subdomain], 
                    pfix=fixed_points,
                    h0=hmin
                )

                #This code extracts points that should conform to rectangular boundary
                centroids = np.zeros((len(cells), 2))
                for i, cell in enumerate(cells):
                    centroids[i] = np.mean(points[cell], axis=0)
                # Finding elements inside rectangle 
                inside_rect = ((centroids[:, 0] >= -Lz) & (centroids[:, 0] <= 0.0) & 
                               (centroids[:, 1] >= 0.0) & (centroids[:, 1] <= Lx))
                cells_inside_rect = cells[inside_rect]
                # Extract cube/rectangle boundary points (CBP)
                if len(cells_inside_rect) > 0:
                    rect_boundary_edges = SeismicMesh.geometry.get_winded_boundary_edges(cells_inside_rect)
                    rect_boundary_point_indices = np.unique(rect_boundary_edges.flatten())
                    CBP = points[rect_boundary_point_indices]
                else:
                    CBP = np.empty((0, 2))

                # Adjustments to make rectangular subdomain boundary elements conforming to mesh
                CBPc = snap_to_rectangle_boundary(CBP, bbox)
                points = substitute_corrected_points(points, CBP, CBPc, tolerance=1e-10)
                points = filter_points_near_boundary(points, bbox, hmin/2*hmin/min_value)
                #points = remove_close_boundary_points_iterative(points, bbox, hmin/2*hmin/min_value, boundary_tolerance=1e-6)
                points = remove_close_boundary_points_variable_size(points, bbox, ef_segy,hmin,min_value, boundary_tolerance=1e-6)
                # Create new Delaunay for modified points
                fixed_triangulation = Delaunay(points)
                cells = fixed_triangulation.simplices   
                
                if self.comm.comm.rank == 0:
                    meshio.write_points_cells(
                        "automatic_mesh.msh",
                        points,
                        [("triangle", cells)],
                        file_format="gmsh22",
                        binary=False
                    )
    
                    meshio.write_points_cells(
                        "automatic_mesh.vtk",
                        points,
                        [("triangle", cells)],
                        file_format="vtk"
                    )
                
        self.comm.comm.barrier()
        mesh = fire.Mesh(
            'automatic_mesh.msh',
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
            comm=self.comm.comm,
        )

        return mesh

    def create_seismicmesh_2D_mesh_homogeneous(self):
        """
        Creates a 2D mesh based on SeismicMesh meshing utilities, with homogeneous velocity model.
        """
        Lz = self.length_z
        Lx = self.length_x
        pad = self.abc_pad
        padding_type = self.padding_type

        if pad is not None:
            real_lz = Lz + pad
            real_lx = Lx + 2 * pad
        else:
            real_lz = Lz
            real_lx = Lx
            pad = 0.0

        edge_length = self.edge_length
        if edge_length is None:
            edge_length = self.minimum_velocity/(self.source_frequency*self.cpw)

        if padding_type == "rectangular" or padding_type == None:
            bbox = (-real_lz, 0.0, -pad, real_lx - pad)
            rectangle = SeismicMesh.geometry.Rectangle(bbox)
            
            points, cells = SeismicMesh.generation.generate_mesh(
                domain=rectangle,
                edge_length=edge_length
            )
    
            points, cells = SeismicMesh.geometry.delete_boundary_entities(
                points, cells, min_qual=0.6
            )
    
            meshio.write_points_cells(
                self.output_file_name,
                points,
                [("triangle", cells)],
                file_format="gmsh22",
                binary=False,
            )
            meshio.write_points_cells(
                self.output_file_name + ".vtk",
                points,
                [("triangle", cells)],
                file_format="vtk",
            )
        else:
            # Ellipse exponent
            ellipse_n = self.ellipse_n
            domain_pad = pad
            # Rectangle center 
            zc = -Lz / 2.0     
            xc = Lx / 2.0     
            # Ellipse semi-axes 
            ellipse_a = (-Lz / 2.0) - domain_pad  
            ellipse_b = (Lx / 2.0) + domain_pad  
            # Domain bounding box 
            domain_zmin = zc + ellipse_a
            domain_zmax =  0.0
            domain_xmin = xc - ellipse_b 
            domain_xmax = xc + ellipse_b 
            d_bbox = (domain_zmin, domain_zmax, domain_xmin, domain_xmax)

            fixed_points = [[0.0,0.0],[0.0,Lx],[-Lz,Lx],[-Lz,0.0]]
            rectangle_subdomain = SeismicMesh.geometry.Rectangle((-Lz, 0.0, 0.0, Lx))
            
            # Generate mesh
            points, cells = SeismicMesh.generation.generate_mesh(
                domain=lambda p: domain_sdf(p, a=ellipse_a, b=ellipse_b, n=ellipse_n, zc=zc, xc=xc),
                edge_length=edge_length,
                bbox=d_bbox,
                subdomains=[rectangle_subdomain], 
                pfix=fixed_points,
            )

            bbox = (-Lz, 0.0, 0.0, Lx)
            centroids = np.zeros((len(cells), 2))
            for i, cell in enumerate(cells):
                centroids[i] = np.mean(points[cell], axis=0)
            # Finding elements inside rectangle 
            inside_rect = ((centroids[:, 0] >= -Lz) & (centroids[:, 0] <= 0.0) & 
                           (centroids[:, 1] >= 0.0) & (centroids[:, 1] <= Lx))
            cells_inside_rect = cells[inside_rect]
            # Extract cube/rectangle boundary points (CBP)
            if len(cells_inside_rect) > 0:
                rect_boundary_edges = SeismicMesh.geometry.get_winded_boundary_edges(cells_inside_rect)
                rect_boundary_point_indices = np.unique(rect_boundary_edges.flatten())
                CBP = points[rect_boundary_point_indices]
            else:
                CBP = np.empty((0, 2))

            # Adjustments to make rectangular subdomain boundary elements conforming to mesh
            CBPc = snap_to_rectangle_boundary(CBP, bbox)
            points = substitute_corrected_points(points, CBP, CBPc, tolerance=1e-10)
            points = filter_points_near_boundary(points, bbox, edge_length/2)
            points = remove_close_boundary_points_iterative(points, bbox, edge_length/2, boundary_tolerance=1e-6)
            # Create new Delaunay for modified points
            fixed_triangulation = Delaunay(points)
            cells = fixed_triangulation.simplices 
            
            print('entering spatial rank 0 after mesh generation')
            if self.comm.comm.rank == 0:
                meshio.write_points_cells(
                    "automatic_mesh.msh",
                    points,
                    [("triangle", cells)],
                    file_format="gmsh22",
                    binary=False
                )

                meshio.write_points_cells(
                    "automatic_mesh.vtk",
                    points,
                    [("triangle", cells)],
                    file_format="vtk"
                )

        return fire.Mesh(self.output_file_name)


def calculate_edge_length(cpw, minimum_velocity, frequency):
    v_min = minimum_velocity

    lbda_min = v_min/frequency

    edge_length = lbda_min/cpw
    return edge_length

# def create_firedrake_3D_mesh_based_on_parameters(dx, cell_type):
#     nx = int(self.length_x / dx)
#     nz = int(self.length_z / dx)
#     ny = int(self.length_y / dx)
#     if self.cell_type == "quadrilateral":
#         quadrilateral = True
#     else:
#         quadrilateral = False

#     return spyro.BoxMesh(
#         nz,
#         nx,
#         ny,
#         self.length_z,
#         self.length_x,
#         self.length_y,
#         quadrilateral=quadrilateral,
#     )


def RectangleMesh(nx, ny, Lx, Ly, pad=None, comm=None, quadrilateral=False):
    """Create a rectangle mesh based on the Firedrake mesh.
    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

    Parameters
    ----------
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    nx : int
        Number of elements in the x direction.
    ny : int
        Number of elements in the y direction.
    pad : float, optional
        Padding to be added to the domain. The default is None.
    comm : MPI communicator, optional
        MPI communicator. The default is None.
    quadrilateral : bool, optional
        If True, the mesh is quadrilateral. The default is False.

    Returns
    -------
    mesh : Firedrake Mesh
        Mesh
    """
    if pad is not None:
        Lx += pad
        Ly += 2 * pad
    else:
        pad = 0
    mesh = fire.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral, comm=comm)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def PeriodicRectangleMesh(
    nx, ny, Lx, Ly, pad=None, comm=None, quadrilateral=False
):
    """Create a periodic rectangle mesh based on the Firedrake mesh.
    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

    Parameters
    ----------
    Lx : float
        Length of the domain in the x direction.
    Ly : float
        Length of the domain in the y direction.
    nx : int
        Number of elements in the x direction.
    ny : int
        Number of elements in the y direction.
    pad : float, optional
        Padding to be added to the domain. The default is None.
    comm : MPI communicator, optional
        MPI communicator. The default is None.
    quadrilateral : bool, optional
        If True, the mesh is quadrilateral. The default is False.

    Returns
    -------
    mesh : Firedrake Mesh
        Mesh

    """
    if pad is not None:
        Lx += pad
        Ly += 2 * pad
    else:
        pad = 0
    mesh = fire.PeriodicRectangleMesh(
        nx, ny, Lx, Ly, quadrilateral=quadrilateral, comm=comm
    )
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def BoxMesh(nx, ny, nz, Lx, Ly, Lz, pad=None, quadrilateral=False):
    if pad is not None:
        Lx += pad
        Ly += 2 * pad
        Lz += 2 * pad
    else:
        pad = 0
    if quadrilateral:
        quad_mesh = fire.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)
        quad_mesh.coordinates.dat.data[:, 0] *= -1.0
        quad_mesh.coordinates.dat.data[:, 1] -= pad
        layer_height = Lz / nz
        mesh = fire.ExtrudedMesh(quad_mesh, nz, layer_height=layer_height)
    else:
        mesh = fire.BoxMesh(nx, ny, nz, Lx, Ly, Lz)
        mesh.coordinates.dat.data[:, 0] *= -1.0

    return mesh

def stable_superellipse_sdf(p, a, b, n, zc, xc):
    a = abs(a)
    b = abs(b)
    
    z = (p[:, 0] - zc) / a
    x = (p[:, 1] - xc) / b
    
    eps = 1e-8
    z_n = np.power(np.abs(z) + eps, n)
    x_n = np.power(np.abs(x) + eps, n)
    
    r = np.power(z_n + x_n, 1.0 / n)
    sdf = r - 1.0
    scale = min(abs(a), abs(b))  
    return sdf * scale

def top_cut_sdf(p, z_cut=0.0):
    return (p[:, 0] - z_cut)  

def domain_sdf(p, a, b, n, zc, xc):
    return np.maximum(
        stable_superellipse_sdf(p, a=a, b=b, n=n, zc=zc, xc=xc), 
        top_cut_sdf(p, z_cut=0.0)
    )

def create_padded_segy_function(ef_segy, box_zmin, box_zmax, box_xmin, box_xmax, 
                               domain_zmin, domain_zmax, domain_xmin, domain_xmax,
                               nz_original=200, nx_original=500, 
                               pad_mode='edge', pad_size=None,
                               transition_width=600):

    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    
    """
    Create an extended SEGY function with padding.
    If pad_mode == 'constant', instead of a sharp jump, values
    grow smoothly from the rectangle to the padding value.
    transition_width is in grid cells.
    """

    # Original rectangle grid
    z_rect = np.linspace(box_zmin, box_zmax, nz_original)
    x_rect = np.linspace(box_xmin, box_xmax, nx_original)
    Z_rect, X_rect = np.meshgrid(z_rect, x_rect, indexing='ij')
    rect_points = np.column_stack([Z_rect.ravel(), X_rect.ravel()])
    
    # Evaluate original function
    rect_values = ef_segy.eval(rect_points)
    rect_grid = rect_values.reshape(nz_original, nx_original)
    
    # Resolution
    dz_rect = (box_zmax - box_zmin) / (nz_original - 1)
    dx_rect = (box_xmax - box_xmin) / (nx_original - 1)
    
    # Padding sizes
    pad_z_bottom = int(np.ceil((box_zmin - domain_zmin) / dz_rect))
    pad_z_top    = int(np.ceil((domain_zmax - box_zmax) / dz_rect))
    pad_x_left   = int(np.ceil((box_xmin - domain_xmin) / dx_rect))
    pad_x_right  = int(np.ceil((domain_xmax - box_xmax) / dx_rect))
    
    pad_width = ((pad_z_bottom, pad_z_top), (pad_x_left, pad_x_right))
    
    # Start with constant padding
    if pad_mode == 'constant':
        extended_grid = np.full(
            (nz_original + pad_z_bottom + pad_z_top,
             nx_original + pad_x_left + pad_x_right),
            pad_size,
            dtype=float
        )
    else:
        # Use np.pad for non-constant modes
        extended_grid = np.pad(rect_grid, pad_width, mode=pad_mode)
    
    # Put original rectangle values in center
    extended_grid[pad_z_bottom:pad_z_bottom + nz_original,
                  pad_x_left:pad_x_left + nx_original] = rect_grid

    # If constant padding → smooth transition
    if pad_mode == 'constant':
        mask = np.zeros_like(extended_grid, dtype=bool)
        mask[pad_z_bottom:pad_z_bottom + nz_original,
             pad_x_left:pad_x_left + nx_original] = True

        # Distance outside the rectangle
        dist = distance_transform_edt(~mask)

        # Compute weights: 1 inside, decreasing to 0 over transition_width
        weight = np.clip(1 - (dist / transition_width), 0, 1)

        # Blend original rectangle outward to padding value
        original_extended = np.pad(rect_grid, pad_width, mode='edge')
        extended_grid = weight * original_extended + (1 - weight) * pad_size

    # Coordinates for extended domain
    nz_extended, nx_extended = extended_grid.shape
    z_extended = np.linspace(domain_zmin, domain_zmax, nz_extended)
    x_extended = np.linspace(domain_xmin, domain_xmax, nx_extended)
    
    # Final interpolator
    interpolator = RegularGridInterpolator(
        (z_extended, x_extended), 
        extended_grid, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    
    def ef_segy_extended(p):
        p = np.asarray(p)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        return interpolator(p)
    
    return ef_segy_extended

def place_vertical_boundary(points, z_min, z_max, x_boundary, sizing_function, N, scaler):
    """
    Places points greedily along a vertical line (x=x_boundary) from z_min to z_max using a seismic mesh sizing function,
    walking symmetrically from both ends and handling middle overlaps or gaps.
    
    Note: Handles coordinate systems where z_min is negative (e.g., -3000) and z_max is positive.
    
    Parameters:
    -----------
    points : list
        Existing points in the mesh in (z, x) format
    z_min, z_max : float
        Vertical bounds of the boundary (z_min can be negative, z_max typically positive)
    x_boundary : float
        Horizontal position of the vertical boundary (either x_min or x_max)
    sizing_function : object
        Seismic mesh sizing function with .eval() method
    N : int
        Number of sampling points for walking along the boundary
    scaler : float
        Scaling factor for the sizing function
        
    Returns:
    --------
    numpy.ndarray
        Combined array of points in (z, x) format
    """
    # Remove old corner points if present 
    tolerance = 1e-6  
    points = [pt for pt in points 
              if not ((abs(pt[1] - x_boundary) < tolerance and abs(pt[0] - z_min) < tolerance) or
                      (abs(pt[1] - x_boundary) < tolerance and abs(pt[0] - z_max) < tolerance))]
    
    # Calculate midpoint 
    mid_z = 0.5 * (z_min + z_max)
    
    # Start from endpoints 
    lower_centers = [(z_min, x_boundary)]  # Start from most negative z
    upper_centers = [(z_max, x_boundary)]  # Start from most positive z
    
    # Walk from lower bound (z_min) toward middle, moving upward in z
    last_z = z_min
    for z in np.linspace(z_min, mid_z, N):
        if z <= last_z:  # Skip if not moving upward
            continue
        r = sizing_function.eval(np.array([[z, x_boundary]]))[0] * scaler
        r_last = sizing_function.eval(np.array([[last_z, x_boundary]]))[0] * scaler
        if z - last_z >= r + r_last:  # Check sufficient spacing moving upward
            lower_centers.append((z, x_boundary))
            last_z = z
    
    # Walk from upper bound (z_max) toward middle, moving downward in z
    last_z = z_max
    for z in np.linspace(z_max, mid_z, N):
        if z >= last_z:  # Skip if not moving downward
            continue
        r = sizing_function.eval(np.array([[z, x_boundary]]))[0] * scaler
        r_last = sizing_function.eval(np.array([[last_z, x_boundary]]))[0] * scaler
        if last_z - z >= r + r_last:  # Check sufficient spacing moving downward
            upper_centers.append((z, x_boundary))
            last_z = z
    
    # Handle middle zone overlap/gap
    last_lower = lower_centers[-1]
    last_upper = upper_centers[-1]
    z_lower = last_lower[0]  # z coordinate from lower walk
    z_upper = last_upper[0]  # z coordinate from upper walk
    
    r_lower = sizing_function.eval(np.array([[last_lower[0], last_lower[1]]]))[0] * scaler
    r_upper = sizing_function.eval(np.array([[last_upper[0], last_upper[1]]]))[0] * scaler
    
    dist = z_upper - z_lower  # Distance between the two closest points
    gap = dist - (r_lower + r_upper)
    
    if dist < r_lower + r_upper:
        # Overlapping → merge to midpoint
        z_merge = 0.5 * (z_lower + z_upper)
        merged = (z_merge, x_boundary)
        lower_centers[-1] = merged
        upper_centers.pop()
    elif gap > max(r_lower, r_upper) / 2:
        # Significant gap → insert one more point
        z_extra = 0.5 * (z_lower + z_upper)
        lower_centers.append((z_extra, x_boundary))
    
    # Combine and deduplicate (reverse upper_centers to maintain order)
    centers = lower_centers + upper_centers[::-1]
    centers = list(dict.fromkeys(centers))
    
    # Convert to numpy array and combine with existing points
    all_points = points + centers
    return np.array(all_points)
    
def place_horizontal_boundary(points, x_min, x_max, z_min, sizing_function, N,scaler):
    """
    Places points greedily along a horizontal line (z=z_min) from x_min to x_max using a seismic mesh sizing function,
    walking symmetrically from both ends and handling middle overlaps or gaps.
    
    Parameters:
    -----------
    points : list
        Existing points in the mesh in (z, x) format
    x_min, x_max : float
        Horizontal bounds of the boundary
    z_min : float
        Vertical position of the horizontal boundary
    sizing_function : object
        Seismic mesh sizing function with .eval() method
    N : int
        Number of sampling points for walking along the boundary
        
    Returns:
    --------
    list
        Combined list of points in (z, x) format
    """

    # Remove old corner points if present 
    tolerance = 1e-6  
    points = [pt for pt in points 
              if not ((abs(pt[1] - x_min) < tolerance and abs(pt[0] - z_min) < tolerance) or
                      (abs(pt[1] - x_max) < tolerance and abs(pt[0] - z_min) < tolerance))]

    mid_x = 0.5 * (x_min + x_max)

    left_centers = [(z_min, x_min)]
    right_centers = [(z_min, x_max)]

    # Walk from left to mid
    last_x = x_min
    for x in np.linspace(x_min, mid_x, N):
        r = sizing_function.eval(np.array([[z_min, x]]))[0]*scaler
        r_last = sizing_function.eval(np.array([[z_min, last_x]]))[0]*scaler
        if x - last_x >= r + r_last:
            left_centers.append((z_min, x))
            last_x = x

    # Walk from right to mid
    last_x = x_max
    for x in np.linspace(x_max, mid_x, N):
        r = sizing_function.eval(np.array([[z_min, x]]))[0]*scaler
        r_last = sizing_function.eval(np.array([[z_min, last_x]]))[0]*scaler
        if last_x - x >= r + r_last:
            right_centers.append((z_min, x))
            last_x = x

    # Handle middle zone
    last_left = left_centers[-1]
    last_right = right_centers[-1]
    x_left = last_left[1]  # x coordinate is now at index 1
    x_right = last_right[1]  # x coordinate is now at index 1
    r_left = sizing_function.eval(np.array([[last_left[0], last_left[1]]]))[0]
    r_right = sizing_function.eval(np.array([[last_right[0], last_right[1]]]))[0]
    dist = x_right - x_left
    gap = dist - (r_left + r_right)

    if dist < r_left + r_right:
        # Overlapping → merge to midpoint
        x_merge = 0.5 * (x_left + x_right)
        merged = (z_min, x_merge)
        left_centers[-1] = merged
        right_centers.pop()
    elif gap > max(r_left, r_right) / 2:
        # Significant gap → insert one more
        x_extra = 0.5 * (x_left + x_right)
        left_centers.append((z_min, x_extra))

    # Combine and deduplicate
    centers = left_centers + right_centers[::-1]
    centers = list(dict.fromkeys(centers))
    
    # Convert to numpy array and combine with existing points
    all_points = points + centers
    return np.array(all_points)

def snap_to_rectangle_boundary(cbp, segy_bbox):
    """
    Move non-conforming boundary points to snap exactly onto rectangle boundary.
    
    Parameters:
    -----------
    cbp : numpy.ndarray
        Array of shape (N, 2) containing non-conforming boundary points [z, x]
    segy_bbox : tuple
        Rectangle definition (z_min, z_max, x_min, x_max)
    
    Returns:
    --------
    numpy.ndarray
        Array of corrected points snapped to rectangle boundary
    """
    if len(cbp) == 0:
        return cbp
    
    z_min, z_max, x_min, x_max = segy_bbox
    corrected_points = cbp.copy()
    
    # For each point, find the closest rectangle edge and snap to it
    for i, point in enumerate(cbp):
        z, x = point
        
        # Calculate distance to each of the 4 rectangle edges
        distances = []
        projections = []
        
        # Bottom edge (z = z_min, x in [x_min, x_max])
        if x_min <= x <= x_max:
            dist_bottom = abs(z - z_min)
            proj_bottom = np.array([z_min, x])
        else:
            # Distance to closest corner of bottom edge
            x_proj = np.clip(x, x_min, x_max)
            dist_bottom = np.sqrt((z - z_min)**2 + (x - x_proj)**2)
            proj_bottom = np.array([z_min, x_proj])
        distances.append(dist_bottom)
        projections.append(proj_bottom)
        
        # Top edge (z = z_max, x in [x_min, x_max])
        if x_min <= x <= x_max:
            dist_top = abs(z - z_max)
            proj_top = np.array([z_max, x])
        else:
            x_proj = np.clip(x, x_min, x_max)
            dist_top = np.sqrt((z - z_max)**2 + (x - x_proj)**2)
            proj_top = np.array([z_max, x_proj])
        distances.append(dist_top)
        projections.append(proj_top)
        
        # Left edge (x = x_min, z in [z_min, z_max])
        if z_min <= z <= z_max:
            dist_left = abs(x - x_min)
            proj_left = np.array([z, x_min])
        else:
            z_proj = np.clip(z, z_min, z_max)
            dist_left = np.sqrt((z - z_proj)**2 + (x - x_min)**2)
            proj_left = np.array([z_proj, x_min])
        distances.append(dist_left)
        projections.append(proj_left)
        
        # Right edge (x = x_max, z in [z_min, z_max])
        if z_min <= z <= z_max:
            dist_right = abs(x - x_max)
            proj_right = np.array([z, x_max])
        else:
            z_proj = np.clip(z, z_min, z_max)
            dist_right = np.sqrt((z - z_proj)**2 + (x - x_max)**2)
            proj_right = np.array([z_proj, x_max])
        distances.append(dist_right)
        projections.append(proj_right)
        
        # Find the closest edge and use its projection
        closest_edge_idx = np.argmin(distances)
        corrected_points[i] = projections[closest_edge_idx]
    
    return corrected_points

def substitute_corrected_points(points, cbp, corrected_cbp, tolerance=1e-10):
    """
    Substitute cbp points in the points array with their corrected versions.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 2) containing all mesh points [z, x]
    cbp : numpy.ndarray
        Array of shape (M, 2) containing original non-conforming boundary points [z, x]
    corrected_cbp : numpy.ndarray
        Array of shape (M, 2) containing corrected boundary points [z, x]
    tolerance : float, optional
        Tolerance for finding matching points (default: 1e-10)
    
    Returns:
    --------
    numpy.ndarray
        Updated points array with cbp points replaced by corrected_cbp points
    """
    if len(cbp) == 0:
        return points.copy()
    
    if len(cbp) != len(corrected_cbp):
        raise ValueError("cbp and corrected_cbp must have the same length")
    
    # Create a copy of points to avoid modifying the original
    updated_points = points.copy()
    
    # Keep track of substitutions for verification
    substitutions_made = 0
    
    # For each cbp point, find its location in points array and replace it
    for i, (original_point, corrected_point) in enumerate(zip(cbp, corrected_cbp)):
        # Calculate distances from this cbp point to all points in the array
        distances = np.linalg.norm(points - original_point, axis=1)
        
        # Find the closest match
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # If the closest match is within tolerance, replace it
        if closest_distance <= tolerance:
            updated_points[closest_idx] = corrected_point
            substitutions_made += 1
        else:
            print(f"Warning: CBP point {i} at {original_point} not found in points array "
                  f"(closest distance: {closest_distance:.2e})")
    
    print(f"Successfully substituted {substitutions_made} out of {len(cbp)} CBP points")
    
    return updated_points

def filter_points_near_boundary(points, segy_bbox, size, boundary_tolerance=1e-6):
    """
    Remove points that are too close to any boundary of the rectangle,
    but keep points that are actually ON the boundary.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 2) containing mesh points [z, x]
    segy_bbox : tuple
        Rectangle definition (z_min, z_max, x_min, x_max)
    size : float
        Minimum distance from boundary. Points closer than this will be removed,
        except for points actually on the boundary.
    boundary_tolerance : float, optional
        Tolerance for considering a point "on" the boundary (default: 1e-6)
    
    Returns:
    --------
    numpy.ndarray
        Filtered points array with near-boundary points removed but boundary points kept
    numpy.ndarray
        Boolean mask indicating which points were kept (True) or removed (False)
    """
    
    # Input validation
    if points is None:
        raise ValueError("points cannot be None")
    if len(segy_bbox) != 4:
        raise ValueError(f"segy_bbox must have 4 elements, got {len(segy_bbox)}")
    if size < 0:
        raise ValueError("size must be non-negative")
        
    points = np.asarray(points)
    if len(points) == 0:
        return points.copy(), np.array([], dtype=bool)
    if points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    
    z_min, z_max, x_min, x_max = segy_bbox
    z = points[:, 0]
    x = points[:, 1]
    
    # Calculate distance to each boundary edge 
    # Excluding top boundary (z_max) from filtering 
    
    # Distance to bottom boundary (z = z_min) - only for points within x range
    dist_to_bottom = np.full(len(points), np.inf)
    bottom_mask = (x >= x_min) & (x <= x_max)
    dist_to_bottom[bottom_mask] = np.where(
        z[bottom_mask] >= z_min, 
        z[bottom_mask] - z_min, 
        z_min - z[bottom_mask]
    )
    
    # Distance to left boundary (x = x_min) - only for points within z range
    dist_to_left = np.full(len(points), np.inf)
    left_mask = (z >= z_min) & (z <= z_max)
    dist_to_left[left_mask] = np.where(
        x[left_mask] >= x_min, 
        x[left_mask] - x_min, 
        x_min - x[left_mask]
    )
    
    # Distance to right boundary (x = x_max) - only for points within z range
    dist_to_right = np.full(len(points), np.inf)
    right_mask = (z >= z_min) & (z <= z_max)
    dist_to_right[right_mask] = np.where(
        x[right_mask] <= x_max, 
        x_max - x[right_mask], 
        x[right_mask] - x_max
    )
    
    # Find minimum distance to any boundary for each point
    min_distances = np.minimum.reduce([
        dist_to_bottom, dist_to_left, dist_to_right
    ])
    
    # Check if points are ON the boundary (within tolerance)
    on_boundary = min_distances <= boundary_tolerance
    
    # Keep points that are either:
    # 1. At least 'size' distance away from any boundary, OR
    # 2. Actually ON the boundary (within tolerance)
    keep_mask = (min_distances >= size) | on_boundary
    
    filtered_points = points[keep_mask]
    return filtered_points

    from scipy.spatial.distance import pdist, squareform
    
def remove_close_boundary_points_iterative(points, segy_bbox, size, boundary_tolerance=1e-6):
    """
    Remove points that are close to each other, while on the boundary.
    """
    # Input validation (same as above)
    if points is None:
        raise ValueError("points cannot be None")
    if len(segy_bbox) != 4:
        raise ValueError(f"segy_bbox must have 4 elements, got {len(segy_bbox)}")
    if size < 0:
        raise ValueError("size must be non-negative")
        
    points = np.asarray(points)
    if len(points) == 0:
        return points.copy()
    if points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    
    z_min, z_max, x_min, x_max = segy_bbox
    z = points[:, 0]
    x = points[:, 1]
    
    # Identify boundary points
    on_bottom = np.abs(z - z_min) <= boundary_tolerance
    on_top = np.abs(z - z_max) <= boundary_tolerance
    on_left = np.abs(x - x_min) <= boundary_tolerance
    on_right = np.abs(x - x_max) <= boundary_tolerance
    
    on_boundary = on_bottom | on_top | on_left | on_right
    boundary_indices = np.where(on_boundary)[0]
    
    if len(boundary_indices) <= 1:
        return points.copy()
    
    # Iteratively remove close points
    keep_indices = []
    remaining_boundary_indices = boundary_indices.tolist()
    
    while remaining_boundary_indices:
        # Take the first remaining point
        current_idx = remaining_boundary_indices.pop(0)
        keep_indices.append(current_idx)
        current_point = points[current_idx]
        
        # Remove all remaining points that are too close to current point
        indices_to_remove = []
        for i, other_idx in enumerate(remaining_boundary_indices):
            other_point = points[other_idx]
            distance = np.linalg.norm(current_point - other_point)
            if distance < size:
                indices_to_remove.append(i)
        
        # Remove indices in reverse order to maintain correct indexing
        for i in reversed(indices_to_remove):
            remaining_boundary_indices.pop(i)
    
    # Create final mask
    keep_mask = np.ones(len(points), dtype=bool)
    boundary_indices_set = set(boundary_indices)
    keep_indices_set = set(keep_indices)
    
    for idx in boundary_indices_set:
        if idx not in keep_indices_set:
            keep_mask[idx] = False
    
    return points[keep_mask]

def remove_close_boundary_points_variable_size(points, segy_bbox, ef_segy, hmin, min_value, boundary_tolerance=1e-6):
    """
    Remove points that are ON the boundary lines and too close to each other using variable sizing.
    
    This function identifies points that lie exactly on the rectangle boundary lines
    and removes one point from each pair that are closer than the sizing function threshold.
    Once a point is kept from a pair, it is protected from future removals.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 2) containing mesh points [z, x]
    segy_bbox : tuple
        Rectangle definition (z_min, z_max, x_min, x_max)
    ef_segy : SeismicMesh sizing function
        Sizing function that can be evaluated with ef_segy.eval(points)
        to get the desired mesh size at each point location
    boundary_tolerance : float, optional
        Tolerance for considering a point "on" the boundary line (default: 1e-6)
    
    Returns:
    --------
    numpy.ndarray
        Modified points array with close boundary points removed
    """
    # Input validation
    if points is None:
        raise ValueError("points cannot be None")
    if len(segy_bbox) != 4:
        raise ValueError(f"segy_bbox must have 4 elements, got {len(segy_bbox)}")
    if ef_segy is None:
        raise ValueError("ef_segy cannot be None")
        
    points = np.asarray(points)
    if len(points) == 0:
        return points.copy()
    if points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    
    z_min, z_max, x_min, x_max = segy_bbox
    z = points[:, 0]
    x = points[:, 1]
    
    # Identify points that are exactly ON the boundary lines
    on_bottom = np.abs(z - z_min) <= boundary_tolerance
    on_top = np.abs(z - z_max) <= boundary_tolerance
    on_left = np.abs(x - x_min) <= boundary_tolerance
    on_right = np.abs(x - x_max) <= boundary_tolerance
    
    # Only consider points that are actually ON the boundary lines
    on_boundary = on_bottom | on_top | on_left | on_right
    
    # Further filter to ensure points are within the rectangle bounds
    within_z_bounds = (z >= z_min - boundary_tolerance) & (z <= z_max + boundary_tolerance)
    within_x_bounds = (x >= x_min - boundary_tolerance) & (x <= x_max + boundary_tolerance)
    
    # Final boundary condition: on boundary line AND within rectangle
    on_boundary = on_boundary & within_z_bounds & within_x_bounds
    
    # If no boundary points or only one, return original points
    boundary_indices = np.where(on_boundary)[0]
    if len(boundary_indices) <= 1:
        return points.copy()
    
    # Get boundary points and evaluate sizing function
    boundary_points = points[boundary_indices]
    segy_sizes = ef_segy.eval(boundary_points)*hmin/min_value/2
    
    # Track which boundary points to keep (start with all)
    keep_boundary_mask = np.ones(len(boundary_indices), dtype=bool)
    protected_indices = set()  # Points that are protected from removal
    
    # Process pairs iteratively
    for i in range(len(boundary_indices)):
        if not keep_boundary_mask[i] or i in protected_indices:
            continue  # Skip if already removed or protected
            
        point_i = boundary_points[i]
        size_i = segy_sizes[i]
        
        # Check against all subsequent points
        for j in range(i + 1, len(boundary_indices)):
            if not keep_boundary_mask[j] or j in protected_indices:
                continue  # Skip if already removed or protected
                
            point_j = boundary_points[j]
            size_j = segy_sizes[j]
            
            # Calculate distance
            distance = np.linalg.norm(point_i - point_j)
            size_threshold = min(size_i, size_j)
            
            if distance < size_threshold:
                # Remove the point with higher index, protect the other
                keep_boundary_mask[j] = False  # Remove point j
                protected_indices.add(i)  # Protect point i
                break  # Move to next i since we found a pair
    
    # Apply the mask to get final boundary points to keep
    final_boundary_indices = boundary_indices[keep_boundary_mask]
    
    # Create mask for all points
    keep_mask = np.ones(len(points), dtype=bool)
    boundary_indices_set = set(boundary_indices)
    final_boundary_indices_set = set(final_boundary_indices)
    
    # Remove boundary points that weren't selected to keep
    for idx in boundary_indices_set:
        if idx not in final_boundary_indices_set:
            keep_mask[idx] = False
    
    return points[keep_mask]