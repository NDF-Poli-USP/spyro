import numpy as np
from firedrake import Mesh as FireMeshReader

from .firedrake_based_wrappers import box_mesh, periodic_rectangle_mesh, rectangle_mesh
from spyro.mpi.spyro_mpi import SpyroEnsemble
from .meshing_utils import calculate_edge_length
from .seismic_mesh_based_wrappers import create_seismicmesh_2D_mesh_with_velocity_model
from .seismic_mesh_based_wrappers import create_seismicmesh_2D_mesh_homogeneous
from .gmsh_based_methods import build_big_rect_with_inner_element_group
from .meshing_2d import generate_gmsh_2d_mesh
from .meshing_3d import generate_gmsh_3d_mesh


try:
    import SeismicMesh
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
    create_gmsh_2D_mesh()
        Creates a 2D mesh using Gmsh with padding and smoothing.
    """

    def __init__(self, mesh_parameters=None):
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
        - 'dimension' : int, optional. Dimension of the mesh. The default is 2.
        - 'length_z' : float, optional. Length of the mesh in the z-direction.
        - 'length_x' : float, optional. Length of the mesh in the x-direction.
        - 'length_y' : float, optional. Length of the mesh in the y-direction.
        - 'cell_type' : str, optional. Type of the mesh cells.
        - 'mesh_type' : str, optional. Type of the mesh.

        For mesh with absorbing layer only:
        - 'abc_pad_length' : float, optional. Length of the absorbing boundary condition padding.

        For Firedrake mesh only:
        - 'dx' : float, optional. Mesh element size.
        - 'periodic' : bool, optional. Whether the mesh is periodic.
        - 'edge_length' : float, optional. Length of the mesh edges.

        For SeismicMesh only:
        - 'cells_per_wavelength' : float, optional. Number of cells per wavelength.
        - 'source_frequency' : float, optional. Frequency of the source.
        - 'minimum_velocity' : float, optional. Minimum velocity.
        - 'velocity_model_file' : str, optional. File containing the velocity model.
        - 'edge_length' : float, optional. Length of the mesh edges.
        """
        self.dimension = mesh_parameters.dimension
        self.length_z = mesh_parameters.length_z
        self.length_x = mesh_parameters.length_x
        self.length_y = mesh_parameters.length_y
        self.quadrilateral = mesh_parameters.quadrilateral
        self.comm = mesh_parameters.comm
        self.mesh_type = mesh_parameters.mesh_type
        self.edge_length = mesh_parameters.edge_length
        self.edge_length_z = getattr(mesh_parameters, "edge_length_z", None)
        self.edge_length_x = getattr(mesh_parameters, "edge_length_x", None)
        self.edge_length_y = getattr(mesh_parameters, "edge_length_y", None)
        self.abc_pad = mesh_parameters.abc_pad_length
        self.mesh_parameters = mesh_parameters

        # Firedrake mesh only parameters

        self.periodic = mesh_parameters.periodic

        # SeismicMesh only parameters
        self.cpw = mesh_parameters.cells_per_wavelength
        self.source_frequency = mesh_parameters.source_frequency
        self.minimum_velocity = mesh_parameters.minimum_velocity
        self.lbda = None
        self.velocity_model = mesh_parameters.velocity_model
        self.output_file_name = mesh_parameters.output_filename

    def _resolved_edge_length(self, axis):
        axis_value = getattr(self, f"edge_length_{axis}", None)
        if axis_value is not None:
            return axis_value
        return self.edge_length

    def create_mesh(self):
        """
        Creates the mesh.

        Returns
        -------
        mesh : Mesh
            Mesh
        """
        self.mesh_parameters.check_completeness()
        if self.mesh_parameters.is_complete is False:
            SpyroEnsemble.print(
                "Skipping mesh generation, since we don't have all the parameters"
            )
            return None
        SpyroEnsemble.print(f"Creating {self.mesh_type} type mesh.")
        if self.mesh_type == "firedrake_mesh":
            return self.create_firedrake_mesh()
        elif self.mesh_type == "SeismicMesh":
            if SeismicMesh is None:
                raise ImportError(
                    "SeismicMesh is not available. Please "
                    + "install it to use this function."
                )
            return self.create_seismicmesh_mesh()
        elif self.mesh_type == "spyro_mesh":
            self.create_spyro_mesh()
            return self._load_mesh_file()
        elif self.mesh_type == "gmsh_mesh":
            if self.dimension == 2:
                return self.create_gmsh_2D_mesh()
            elif self.dimension == 3:
                return self.create_gmsh_3D_mesh()
        else:
            raise ValueError("mesh_type is not supported")

    def ensure_common_origin(self, mesh, pad=0.0):
        """Ensures that the mesh has a common origin.

        Parameters
        ----------
        mesh : Mesh
            The mesh to be adjusted.
        pad : float, optional
            The padding to be added to the mesh. The default is 0

        Returns
        -------
        None
        """

        # Adjusting coordinates
        if self.dimension == 3:  # 3D
            min_y = mesh.coordinates.dat.data_with_halos[:, 2].min()
            if abs(min_y / pad) != 1.0:  # Forcing node at (0,0,0)
                SpyroEnsemble.print("Adjusting Mesh Y-coordinates")
                err_y = (1.0 - abs(min_y / pad)) * pad
                err_y *= -np.sign(err_y)
                mesh.coordinates.dat.data_with_halos[:, 2] += err_y

        # Adjusting coordinates
        min_x = mesh.coordinates.dat.data_with_halos[:, 1].min()
        if abs(min_x / pad) != 1.0:  # Forcing node at (0,0)
            SpyroEnsemble.print("Adjusting Mesh X-coordinates")
            err_x = (1.0 - abs(min_x / pad)) * pad
            err_x *= -np.sign(err_x)
            mesh.coordinates.dat.data_with_halos[:, 1] += err_x

    def create_firedrake_mesh(self):
        """Creates a mesh based on Firedrake meshing utilities.

        Returns
        -------
        mesh : `Firedrake.Mesh`
            The generated mesh.

        Raises
        ------
        ValueError
            If the dimension is not supported (must be 2 or 3).
        """
        if self.dimension == 2:
            typ_ele_str = "Area Elements"
            mesh = self.create_firedrake_2D_mesh()

        elif self.dimension == 3:
            typ_ele_str = "Volume Elements"
            mesh = self.create_firedrake_3D_mesh()

        else:
            raise ValueError("dimension is not supported")

        if self.abc_pad is not None and self.abc_pad > 0.0:
            self.ensure_common_origin(mesh, pad=self.abc_pad)

        # Mesh data
        SpyroEnsemble.print(
            f"Mesh Created with {mesh.num_vertices()} Nodes and "
            + f"{mesh.num_cells()} "
            + typ_ele_str
        )

        return mesh

    def define_discretization_for_mesh(self):
        """Define the discretization of the mesh.

        Returns
        -------
        discretization : `tuple`
            Number of elements in each dimension of the mesh.
            Structure: (nz, nx) for 2D and (nz, nx, ny) for 3D
        """

        # Compute the edge length if there is no one
        if (
            self.edge_length is None
            and self.edge_length_z is None
            and self.edge_length_x is None
            and self.edge_length_y is None
            and self.cpw is not None
        ):
            self.edge_length = calculate_edge_length(
                self.cpw, self.minimum_velocity, self.source_frequency
            )

        edge_length_z = self._resolved_edge_length("z")
        edge_length_x = self._resolved_edge_length("x")
        edge_length_y = self._resolved_edge_length("y")

        # Number of elements
        pad = 0.0 if self.abc_pad is None else self.abc_pad
        n_pad_z = round(pad / edge_length_z, 0) if edge_length_z is not None else 0
        n_pad_x = round(pad / edge_length_x, 0) if edge_length_x is not None else 0
        nz = int(round(self.length_z / edge_length_z, 0)) + int(n_pad_z)
        nx = int(round(self.length_x / edge_length_x, 0)) + int(2 * n_pad_x)
        discretization = (nz, nx)
        if self.dimension == 3:
            n_pad_y = round(pad / edge_length_y, 0) if edge_length_y is not None else 0
            ny = int(round(self.length_y / edge_length_y, 0)) + int(2 * n_pad_y)
            discretization += (ny,)

        return discretization

    def create_firedrake_2D_mesh(self):
        """
        Creates a 2D mesh based on Firedrake meshing utilities.

        Returns
        -------
        mesh : `Firedrake.Mesh`
            The generated 2D mesh.

        Notes
        -----
        If edge_length is not specified but cells_per_wavelength(cpw) is provided,
        the edge length will be calculated automatically. The method creates either
        a periodic or non-periodic rectangular mesh based on the periodic attribute.
        """
        # Define the discretization
        nz, nx = self.define_discretization_for_mesh()

        if self.comm is not None:
            comm = self.comm.comm
        else:
            comm = None

        if self.periodic:
            return periodic_rectangle_mesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=self.quadrilateral,
                comm=comm,
                pad=self.abc_pad,
            )
        else:
            return rectangle_mesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=self.quadrilateral,
                comm=comm,
                pad=self.abc_pad,
            )

    def create_firedrake_3D_mesh(self):
        """
        Creates a 3D mesh based on Firedrake meshing utilities.

        Returns
        -------
        mesh : `Firedrake.Mesh`
            The generated 3D box mesh.

        Notes
        -----
        Uses the edge_length parameter to determine the number of elements
        in each direction(x, y, z).
        """

        # Define the discretization
        nz, nx, ny = self.define_discretization_for_mesh()

        if self.comm is not None:
            comm = self.comm.comm
        else:
            comm = None

        return box_mesh(
            nz,
            nx,
            ny,
            self.length_z,
            self.length_x,
            self.length_y,
            pad=self.abc_pad,
            quadrilateral=self.quadrilateral,
            comm=comm,
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

        Returns
        -------
        mesh : Firedrake Mesh
            The generated 2D mesh.

        Notes
        -----
        If a velocity model is provided, the mesh will be refined based on
        the velocity model. Otherwise, a homogeneous mesh is created.
        """
        print(f"velocity_model{self.velocity_model}", flush=True)
        if self.velocity_model is None:
            return create_seismicmesh_2D_mesh_homogeneous(self)
        else:
            return create_seismicmesh_2D_mesh_with_velocity_model(self)

    def create_spyro_mesh(self):
        """
        Creates a mesh using spyro's internal meshing utilities based on gmsh calls. This
        mesh has tags that define the dx integration in Firedrake.

        Returns
        -------
        mesh : `Firedrake.Mesh`
            The generated mesh.

        Raises
        ------
        ValueError
            If dimension is not supported (must be 2).
        NotImplementedError
            If dimension is 3 (3D meshing not yet implemented).

        Notes
        -----
        Currently, only 2D meshing is implemented via build_big_rect_with_inner_element_group.
        """
        if self.dimension == 2:
            return build_big_rect_with_inner_element_group(self.mesh_parameters)
        elif self.dimension == 3:
            raise NotImplementedError("Not implemented yet")
            # return self.create_seismicmesh_3D_mesh()
        else:
            raise ValueError("dimension is not supported")

    def create_gmsh_2D_mesh(self):
        """
        Creates a 2D mesh using Gmsh with optional water interface,
        hyperelliptical/rectangular padding, structured Winslow smoothing,
        or unstructured full-quadrilateral meshing.

        Returns
        -------
        mesh : Firedrake Mesh
            The loaded Firedrake mesh object.
        """
        self.velocity_model = generate_gmsh_2d_mesh(self.mesh_parameters)
        return self._load_mesh_file()

    def create_gmsh_3D_mesh(self):
        """Create a 3-D Gmsh mesh with the supplied volumetric logic."""
        self.velocity_model = generate_gmsh_3d_mesh(self.mesh_parameters)
        return self._load_mesh_file()

    def _load_mesh_file(self):
        """Wait for the mesh-writing rank, then load the mesh file.

        Returns
        -------
        firedrake.MeshGeometry
            Mesh read from ``self.output_file_name``.
        """
        if self.comm is not None:
            # Ensure all processes wait for mesh creation to complete
            # Need to sync both ensemble and spatial communicators
            if hasattr(self.comm, "ensemble_comm"):
                self.comm.ensemble_comm.barrier()
            self.comm.comm.barrier()
            SpyroEnsemble.print("Loading mesh into Firedrake.")
            return FireMeshReader(self.output_file_name, comm=self.comm.comm)
        return FireMeshReader(self.output_file_name)
