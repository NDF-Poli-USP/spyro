import firedrake as fire
import meshio

try:
    import SeismicMesh
except ImportError:
    SeismicMesh = None


def cells_per_wavelength(method, degree, dimension):
    cell_per_wavelength_dictionary = {
        'mlt2tri': 7.02,
        'mlt3tri': 3.70,
        'mlt4tri': 2.67,
        'mlt5tri': 2.03,
        'mlt2tet': 6.12,
        'mlt3tet': 3.72,
        'sem2quad': None,
        'sem4quad': None,
        'sem6quad': None,
        'sem8quad': None,
        'cg_triangle1tri': None,
    }

    if dimension == 2 and (method == 'mass_lumped_triangle' or method == "MLT"):
        cell_type = 'tri'
        method = 'mlt'
    if dimension == 3 and (method == 'mass_lumped_triangle' or method == "MLT"):
        cell_type = 'tet'
        method = 'mlt'
    if dimension == 2 and method == 'spectral_quadrilateral':
        cell_type = 'quad'
        method = 'sem'
    if dimension == 3 and method == 'spectral_quadrilateral':
        cell_type = 'quad'
        method = 'sem'
    if method == 'CG_triangle':
        cell_type = 'tri'

    key = method.lower()+str(degree)+cell_type

    return cell_per_wavelength_dictionary.get(key)


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
        self, comm=None, mesh_parameters=None
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
        self.dimension = mesh_parameters["dimension"]
        self.length_z = mesh_parameters["length_z"]
        self.length_x = mesh_parameters["length_x"]
        self.length_y = mesh_parameters["length_y"]
        self.cell_type = mesh_parameters["cell_type"]
        self.comm = comm
        if mesh_parameters["abc_pad_length"] is None:
            self.abc_pad = 0.0
        elif mesh_parameters["abc_pad_length"] >= 0.0:
            self.abc_pad = mesh_parameters["abc_pad_length"]
        else:
            raise ValueError("abc_pad must be positive")
        self.mesh_type = mesh_parameters["mesh_type"]

        # Firedrake mesh only parameters
        self.dx = mesh_parameters["dx"]
        self.quadrilateral = False
        self.periodic = mesh_parameters["periodic"]
        if self.dx is None:
            self.dx = mesh_parameters["edge_length"]

        # SeismicMesh only parameters
        self.cpw = mesh_parameters["cells_per_wavelength"]
        self.source_frequency = mesh_parameters["source_frequency"]
        self.minimum_velocity = mesh_parameters["minimum_velocity"]
        self.lbda = None
        self.velocity_model = mesh_parameters["velocity_model_file"]
        self.edge_length = mesh_parameters["edge_length"]
        self.output_file_name = "automatic_mesh.msh"

    def set_mesh_size(self, length_z=None, length_x=None, length_y=None):
        """
        Parameters
        ----------
        length_z : float, optional
            Length of the domain in the z direction. The default is None.
        length_x : float, optional
            Length of the domain in the x direction. The default is None.
        length_y : float, optional
            Length of the domain in the y direction. The default is None.

        Returns
        -------
        None
        """

        if length_z is not None:
            self.length_z = length_z
        if length_x is not None:
            self.length_x = length_x
        if length_y is not None:
            self.length_y = length_y

    def set_meshing_parameters(self, dx=None, cell_type=None, mesh_type=None):
        """
        Parameters
        ----------
        dx : float, optional
            Mesh size. The default is None.
        cell_type : str, optional
            Type of the cell. The default is None.
        mesh_type : str, optional
            Type of the mesh. The default is None.

        Returns
        -------
        None
        """
        if cell_type is not None:
            self.cell_type = cell_type
        if self.cell_type == "quadrilateral":
            self.quadrilateral = True
        if dx is not None:
            self.dx = dx
        if mesh_type is not None:
            self.mesh_type = mesh_type

        if self.mesh_type != "firedrake_mesh":
            raise ValueError("mesh_type is not supported")

    def set_seismicmesh_parameters(
        self,
        cpw=None,
        velocity_model=None,
        edge_length=None,
        output_file_name=None,
    ):
        """
        Parameters
        ----------
        cpw : float, optional
            Cells per wavelength parameter. The default is None.
        velocity_model : str, optional
            Velocity model. The default is None.
        edge_length : float, optional
            Edge length. The default is None.
        output_file_name : str, optional
            Output file name. The default is None.

        Returns
        -------
        None
        """
        if SeismicMesh is None:
            raise ImportError("SeismicMesh is not available. Please install it to use this function.")
        if cpw is not None:
            self.cpw = cpw
        if velocity_model is not None:
            self.velocity_model = velocity_model
        if edge_length is not None:
            self.edge_length = edge_length
        if output_file_name is not None:
            self.output_file_name = output_file_name
        elif self.output_file_name is None:
            self.output_file_name = "automatically_generated_mesh.msh"

        if self.mesh_type != "SeismicMesh":
            raise ValueError("mesh_type is not supported")

    def make_periodic(self):
        """
        Sets the mesh boundaries periodic.
        Only works for firedrake_mesh.
        """
        self.periodic = True
        if self.mesh_type != "firedrake_mesh":
            raise ValueError(
                "periodic mesh is only supported for firedrake_mesh"
            )

    def create_mesh(self):
        """
        Creates the mesh.

        Returns
        -------
        mesh : Mesh
            Mesh
        """
        # if self.comm.rank == 0:
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
        if self.dx is None and self.cpw is not None:
            self.dx = calculate_edge_length(self.cpw, self.minimum_velocity, self.source_frequency)
        if self.abc_pad:
            nx = int((self.length_x + 2*self.abc_pad) / self.dx)
            nz = int((self.length_z + self.abc_pad) / self.dx)
        else:
            nx = int(self.length_x / self.dx)
            nz = int(self.length_z / self.dx)

        comm = self.comm
        if self.cell_type == "quadrilateral":
            quadrilateral = True
        else:
            quadrilateral = False

        if self.periodic:
            return PeriodicRectangleMesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=quadrilateral,
                comm=comm.comm,
                pad=self.abc_pad,
            )
        else:
            return RectangleMesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=quadrilateral,
                comm=comm.comm,
                pad=self.abc_pad,
            )

    def create_firedrake_3D_mesh(self):
        """
        Creates a 3D mesh based on Firedrake meshing utilities.
        """
        dx = self.dx
        nx = int(self.length_x / dx)
        nz = int(self.length_z / dx)
        ny = int(self.length_y / dx)
        if self.cell_type == "quadrilateral":
            quadrilateral = True
        else:
            quadrilateral = False

        return BoxMesh(
            nz,
            nx,
            ny,
            self.length_z,
            self.length_x,
            self.length_y,
            quadrilateral=quadrilateral,
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
            if self.cpw is None:
                self.cpw = cells_per_wavelength('MLT', 4, 2)
            C = self.cpw  # cells_per_wavelength(method, degree, dimension)

            Lz = self.length_z
            Lx = self.length_x
            domain_pad = self.abc_pad
            lbda_min = v_min/frequency

            bbox = (-Lz, 0.0, 0.0, Lx)
            domain = SeismicMesh.Rectangle(bbox)

            hmin = lbda_min/C
            self.comm.comm.barrier()

            ef = SeismicMesh.get_sizing_function_from_segy(
                self.velocity_model,
                bbox,
                hmin=hmin,
                wl=C,
                freq=frequency,
                grade=0.15,
                domain_pad=domain_pad,
                pad_style="edge",
                units='km/s',
                comm=self.comm.comm,
            )
            self.comm.comm.barrier()

            # Creating rectangular mesh
            points, cells = SeismicMesh.generate_mesh(
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

        real_lz = Lz + pad
        real_lx = Lx + 2 * pad

        edge_length = self.edge_length
        if edge_length is None:
            edge_length = self.minimum_velocity/(self.source_frequency*self.cpw)

        bbox = (-real_lz, 0.0, -pad, real_lx - pad)
        rectangle = SeismicMesh.Rectangle(bbox)

        points, cells = SeismicMesh.generate_mesh(
            domain=rectangle,
            edge_length=edge_length,
            verbose=0,
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

        return fire.Mesh(self.output_file_name)
        # raise NotImplementedError("Not implemented yet")


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
