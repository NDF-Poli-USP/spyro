import numpy as np
from firedrake import Mesh as FireMeshReader
from scipy.interpolate import RegularGridInterpolator
from ..io import parallel_print
from ..io import create_segy_from_grid
from ..utils import run_in_one_core
from .meshing_gmsh2d import build_gmsh_geometry_and_groups, apply_structured_winslow_smoothing2d
from .meshing_utils import create_sizing_function
from .firedrake_based_wrappers import rectangle_mesh, periodic_rectangle_mesh, box_mesh
from .seismic_mesh_based_wrappers import create_seismicmesh_2D_mesh_with_velocity_model
from .seismic_mesh_based_wrappers import create_seismicmesh_2D_mesh_homogeneous

try:
    import gmsh
except ImportError:
    gmsh = None


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
            parallel_print("Skipping mesh generation, since we don't have all the parameters", comm=self.comm)
            return None
        parallel_print(f"Creating {self.mesh_type} type mesh.", comm=self.comm)
        if self.mesh_type == "firedrake_mesh":
            return self.create_firedrake_mesh()
        elif self.mesh_type == "SeismicMesh":
            if SeismicMesh is None:
                raise ImportError("SeismicMesh is not available. Please "
                                  + "install it to use this function.")
            return self.create_seismicmesh_mesh()
        elif self.mesh_type == "spyro_mesh":
            self.create_spyro_mesh()
            if self.comm is not None:
                # Ensure all processes wait for mesh creation to complete
                # Need to sync both ensemble and spatial communicators
                if hasattr(self.comm, 'ensemble_comm'):
                    self.comm.ensemble_comm.barrier()
                self.comm.comm.barrier()
                parallel_print("Loading mesh.", comm=self.comm)
                return FireMeshReader(self.output_file_name, comm=self.comm.comm)
            else:
                return FireMeshReader(self.output_file_name)
        elif self.mesh_type == "gmsh_mesh":
            return self.create_gmsh_2D_mesh()
        else:
            raise ValueError("mesh_type is not supported")

    def ensure_common_origin(self, mesh, pad=0.):
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
            if abs(min_y / pad) != 1.:  # Forcing node at (0,0,0)
                parallel_print("Adjusting Mesh Y-coordinates", comm=self.comm)
                err_y = (1. - abs(min_y / pad)) * pad
                err_y *= -np.sign(err_y)
                mesh.coordinates.dat.data_with_halos[:, 2] += err_y

        # Adjusting coordinates
        min_x = mesh.coordinates.dat.data_with_halos[:, 1].min()
        if abs(min_x / pad) != 1.:  # Forcing node at (0,0)
            parallel_print("Adjusting Mesh X-coordinates", comm=self.comm)
            err_x = (1. - abs(min_x / pad)) * pad
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

        if self.abc_pad is not None and self.abc_pad > 0.:
            self.ensure_common_origin(mesh, pad=self.abc_pad)

        # Mesh data
        msh_str = f"Mesh Created with {mesh.num_vertices()} Nodes " + \
            f"and {mesh.num_cells()} " + typ_ele_str
        parallel_print(msh_str, comm=self.comm)

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
        if self.edge_length is None and self.cpw is not None:
            self.edge_length = calculate_edge_length(
                self.cpw, self.minimum_velocity, self.source_frequency)

        # Number of elements
        pad = 0. if self.abc_pad is None else self.abc_pad
        n_pad = round(pad / self.edge_length, 0)  # Elements in the layer
        nz = int(round(self.length_z / self.edge_length, 0)) + int(n_pad)
        nx = int(round(self.length_x / self.edge_length, 0)) + int(2 * n_pad)
        discretization = (nz, nx)
        if self.dimension == 3:
            ny = int(round(self.length_y / self.edge_length, 0)) + int(2 * n_pad)
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

        return box_mesh(
            nz,
            nx,
            ny,
            self.length_z,
            self.length_x,
            self.length_y,
            pad=self.abc_pad,
            quadrilateral=self.quadrilateral,
            comm=self.comm.comm)

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
        hyperelliptical/rectangular padding, and Winslow smoothing.

        Returns
        -------
        mesh : Firedrake Mesh
            The loaded Firedrake mesh object.
        """
        if gmsh is None:
            raise ImportError("gmsh is not available. Please install it.")

        if self.mesh_parameters.segy_velocity_model is None:
            if self.mesh_parameters.velocity_model is not None:
                filename = "tmp_velocity_model.segy"
                vp = self.mesh_parameters.velocity_model["vp_values"]
                create_segy_from_grid(vp, filename, rotate=True)
                self.velocity_model = filename

        # if self.mesh_parameters.segy_velocity_model is None:
        #     raise ValueError("Gmsh mesher temporarily only works with segy files.")

        if self.comm is None or self.comm.ensemble_comm.rank == 0:
            parallel_print("Generating Gmsh mesh...", comm=self.comm)

            depth_z = -abs(self.length_z)
            length_x = self.length_x
            padding_z = self.mesh_parameters.padding_z
            padding_x = self.mesh_parameters.padding_x
            hyper_n = self.mesh_parameters.hyper_n
            fname = self.velocity_model
            hmin_segy = self.mesh_parameters.hmin_segy
            wl = self.cpw
            freq = self.source_frequency
            grade = self.mesh_parameters.grade
            water_search_value = self.mesh_parameters.water_search_value
            padding_type = self.mesh_parameters.padding_type
            output_file = self.output_file_name
            water_interface = self.mesh_parameters.water_interface
            vp_water = self.mesh_parameters.vp_water
            structured_mesh = self.mesh_parameters.structured_mesh
            minElementSize = self.mesh_parameters.min_element_size
            winslow_implementation = self.mesh_parameters.winslow_implementation
            apply_winslow = self.mesh_parameters.apply_winslow
            winslow_iterations = self.mesh_parameters.winslow_iterations
            winslow_omega = self.mesh_parameters.winslow_omega
            extend_segy = self.mesh_parameters.extend_segy
            h_padding = self.mesh_parameters.h_padding
            segy_bbox = (depth_z, 0, 0, length_x)

            # Calculating domain bounding box according to selected padding
            box_xmax = length_x
            box_zmin = depth_z
            xc = box_xmax / 2.0
            zc = box_zmin / 2.0

            if padding_type is None:
                domain_xmin, domain_xmax = 0.0, length_x
                domain_zmax, domain_zmin = depth_z, 0.0

            if padding_type == "hyperelliptical":
                hyper_a = (box_zmin / 2.0) - padding_z
                hyper_b = (box_xmax / 2.0) + padding_x
                domain_zmin = zc + hyper_a
                domain_zmax = 0.0
                domain_xmin = xc - hyper_b
                domain_xmax = xc + hyper_b

            if padding_type == "rectangular":
                domain_zmin = box_zmin - padding_z
                domain_zmax = 0.0
                domain_xmin = 0.0 - padding_x
                domain_xmax = box_xmax + padding_x

            # Interpolating sizing function from segy file
            ef_segy2, f_min, f_max, n_samples, n_traces = create_sizing_function(
                fname=fname, hmin=hmin_segy, bbox=segy_bbox, wl=wl, freq=freq,
                pad_type=padding_type, pad_size_x=padding_x, pad_size_z=padding_z,
                grade=grade, vp_water=vp_water
            )

            gmsh.initialize()
            gmsh.option.setNumber("Geometry.Tolerance", 1e-16)
            gmsh.model.add("seismic_model")

            geom_params = build_gmsh_geometry_and_groups(
                gmsh=gmsh, fname=fname, length_x=length_x, depth_z=depth_z,
                padding_type=padding_type, padding_x=padding_x, padding_z=padding_z,
                hyper_n=hyper_n, water_interface=water_interface,
                water_search_value=water_search_value, structured_mesh=structured_mesh,
                minElementSize=minElementSize
            )

            # Standard Params
            z_water_L = geom_params.get("z_water_L")
            z_water_R = geom_params.get("z_water_R")
            pad_x_min = geom_params.get("pad_x_min")
            pad_x_max = geom_params.get("pad_x_max")
            pad_z_min = geom_params.get("pad_z_min")
            a_val = geom_params.get("a_val")
            b_val = geom_params.get("b_val")
            xc = geom_params.get("xc")
            zc = geom_params.get("zc")

            def mesh_size_callback(dim, tag, x, y, z, lc):
                if extend_segy:
                    return float(ef_segy2(np.array([x]), np.array([y]))[0])
                else:
                    z_min_segy, z_max_segy, x_min_segy, x_max_segy = segy_bbox
                    if (x_min_segy <= x <= x_max_segy) and (z_min_segy <= y <= z_max_segy):
                        return float(ef_segy2(np.array([x]), np.array([y]))[0])
                    else:
                        x_proj, y_proj = min(max(x, x_min_segy), x_max_segy), min(max(y, z_min_segy), z_max_segy)
                        base_size = float(ef_segy2(np.array([x_proj]), np.array([y_proj]))[0])
                        tx = abs(x - x_proj) / padding_x if padding_x > 0 else 0.0
                        ty = abs(y - y_proj) / padding_z if padding_z > 0 else 0.0
                        t = min((tx**hyper_n + ty**hyper_n)**(1.0 / hyper_n) if padding_type == "hyperelliptical" else max(tx, ty), 1.0)
                        return float(base_size + t * (h_padding - base_size))

            gmsh.model.mesh.setSizeCallback(mesh_size_callback)
            gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            if structured_mesh and padding_type != "hyperelliptical":
                gmsh.option.setNumber('Mesh.MeshSizeMin', minElementSize)
                gmsh.option.setNumber('Mesh.MeshSizeMax', minElementSize)
                gmsh.model.mesh.setTransfiniteAutomatic()
            gmsh.model.mesh.generate(2)

            if structured_mesh:
                apply_structured_winslow_smoothing2d(
                    gmsh=gmsh, comm=self.comm, geom_params=geom_params,
                    length_x=length_x, depth_z=depth_z, padding_type=padding_type,
                    water_interface=water_interface, hyper_n=hyper_n,
                    winslow_implementation=winslow_implementation,
                    winslow_iterations=winslow_iterations, winslow_omega=winslow_omega,
                    n_samples=n_samples, n_traces=n_traces,
                    domain_xmin=domain_xmin, domain_xmax=domain_xmax,
                    domain_zmin=domain_zmin, domain_zmax=domain_zmax,
                    ef_segy2=ef_segy2, parallel_print=parallel_print,
                    z_water_L=z_water_L, z_water_R=z_water_R, pad_x_min=pad_x_min,
                    pad_x_max=pad_x_max, pad_z_min=pad_z_min, a_val=a_val,
                    b_val=b_val, xc=xc, zc=zc, apply_winslow=apply_winslow
                )
            # Rotating mesh for axis (z,x,y)
            if padding_type in ["rectangular", "hyperelliptical"]:
                rotate_xz = [
                    0.0, 1.0, 0.0, 0.0,
                    -1.0, 0.0, 0.0, domain_xmax,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                ]
            else:
                rotate_xz = [
                    0.0, 1.0, 0.0, -domain_zmin,
                    -1.0, 0.0, 0.0, domain_xmax,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                ]

            gmsh.model.mesh.affineTransform(rotate_xz)
            gmsh.write(output_file)
            parallel_print(f"Gmsh mesh written to {output_file}", comm=self.comm)
            gmsh.finalize()

        # MPI Sync
        if self.comm is not None:
            if hasattr(self.comm, 'ensemble_comm'):
                self.comm.ensemble_comm.barrier()
            self.comm.comm.barrier()
            parallel_print("Loading mesh into Firedrake.", comm=self.comm)
            return FireMeshReader(self.output_file_name, comm=self.comm.comm)
        else:
            return FireMeshReader(self.output_file_name)


def calculate_edge_length(cpw, minimum_velocity, frequency):
    """
    Calculate the edge length for mesh generation.

    Parameters
    ----------
    cpw : float
        Cells per wavelength.
    minimum_velocity : float
        Minimum velocity in the domain.
    frequency : float
        Source frequency.

    Returns
    -------
    edge_length : float
        Calculated edge length for mesh elements.
    """
    if cpw == 0.0 or cpw is None:
        raise ValueError("cpw value of {cpw} invalid for edge length calculation.")
    v_min = minimum_velocity

    lbda_min = v_min / frequency

    edge_length = lbda_min / cpw
    return edge_length


def vp_to_sizing(vp, cpw, frequency):
    """
    Convert velocity field to mesh sizing function.

    Parameters
    ----------
    vp : numpy.ndarray
        P-wave velocity field.
    cpw : float
        Cells per wavelength(must be positive).
    frequency : float
        Source frequency in Hz(must be positive).

    Returns
    -------
    sizing : numpy.ndarray
        Mesh element sizes corresponding to the velocity field.

    Raises
    ------
    ValueError
        If cpw or frequency is not positive.

    Notes
    -----
    The mesh size is calculated as: `size = vp / (frequency * cpw)`
    This ensures that the mesh has the specified number of cells per
    wavelength throughout the domain.
    """
    if cpw < 0.0 or cpw == 0.0:
        raise ValueError(f"Cells-per-wavelength value of {cpw} not supported.")
    if frequency < 0.0 or frequency == 0.0:
        raise ValueError(f"Frequency must be positive and non zero, not {frequency}")

    return vp / (frequency * cpw)


@run_in_one_core
def build_big_rect_with_inner_element_group(mesh_parameters):
    """
    Build a rectangular mesh with optional inner element group using GMSH.

    Parameters
    ----------
    mesh_parameters : object
        Object containing mesh parameters with the following attributes:
        - length_z : float
            Length of domain in z direction.
        - length_x : float
            Length of domain in x direction.
        - output_filename : str
            Path for output mesh file.
        - edge_length : float, optional
            Uniform edge length(if grid_velocity_data is None).
        - grid_velocity_data : dict, optional
            Dictionary with 'vp_values' key containing velocity field for
            adaptive meshing.
        - source_frequency : float, optional
            Source frequency for adaptive meshing.
        - cells_per_wavelength : float, optional
            Cells per wavelength for adaptive meshing.
        - gradient_mask : dict, optional
            Dictionary with keys 'z_min', 'z_max', 'x_min', 'x_max' defining
            a rectangular region to be marked as an inner element group.

    Returns
    -------
    None
        Writes mesh to file specified by mesh_parameters.output_filename.

    Notes
    -----
    This function creates a 2D rectangular mesh with:
    - Adaptive sizing based on velocity field(if provided)
    - Physical boundary groups for absorbing boundary conditions
      (1=Top, 2=Bottom, 3=Right, 4=Left)
    - Optional inner element group for gradient masking

    The function uses GMSH for mesh generation and supports velocity-based
    mesh refinement through interpolation of a regular grid velocity model.
    If a gradient_mask is provided, elements within the specified region
    are separated into an 'Inner' physical group, while elements outside
    are placed in an 'Outer' physical group.
    """
    if gmsh is None:
        raise ImportError("gmsh is not available. Please install it to use this function.")

    length_z = mesh_parameters.length_z
    length_x = mesh_parameters.length_x
    outfile = mesh_parameters.output_filename

    gmsh.initialize()
    gmsh.model.add("BigRect_InnerElements")

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # --- Geometry: on length_x the big rectangle ---
    surf_tag = gmsh.model.occ.addRectangle(-length_z, 0.0, 0.0, length_z, length_x)
    gmsh.model.occ.synchronize()

    # Get boundary edges for tagging
    boundary_entities = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    edge_tags = [abs(entity[1]) for entity in boundary_entities if entity[0] == 1]

    # Identify boundary edges by their geometric center
    boundary_tag_map = {}
    for edge_tag in edge_tags:
        # Get center of mass of the edge
        com = gmsh.model.occ.getCenterOfMass(1, edge_tag)
        x_center, y_center = com[0], com[1]

        # Classify edges based on position
        # Top edge: z ≈ 0
        if abs(x_center - 0.0) < 1e-10:
            boundary_tag_map[edge_tag] = 1  # Top boundary
        # Bottom edge: z ≈ -length_z
        elif abs(x_center - (-length_z)) < 1e-10:
            boundary_tag_map[edge_tag] = 2  # Bottom boundary
        # Right edge: y ≈ length_x
        elif abs(y_center - length_x) < 1e-10:
            boundary_tag_map[edge_tag] = 3  # Right boundary
        # Left edge: y ≈ 0
        elif abs(y_center - 0.0) < 1e-10:
            boundary_tag_map[edge_tag] = 4  # Left boundary

    if mesh_parameters.grid_velocity_data is None:
        h_min = mesh_parameters.edge_length
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h_min))

        def mesh_size_callback(dim, tag, x, y, z, lc):
            h = x + y
            return float(h)
    else:
        frequency = mesh_parameters.source_frequency
        cpw = mesh_parameters.cells_per_wavelength
        vp = mesh_parameters.grid_velocity_data["vp_values"]
        nz, nx = vp.shape
        z_grid = np.linspace(-length_z, 0.0, nz, dtype=np.float32)
        x_grid = np.linspace(0.0, length_x, nx, dtype=np.float32)
        cell_sizes = vp_to_sizing(vp, cpw, frequency)
        interpolator = RegularGridInterpolator(
            (z_grid, x_grid), cell_sizes, bounds_error=False
        )
        gmsh.model.occ.synchronize()

        def mesh_size_callback(dim, tag, x, y, z, lc):
            size = float(interpolator([[x, y]])[0])
            return size
    gmsh.model.mesh.setSizeCallback(mesh_size_callback)

    # --- Mesh the single surface ---
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()

    # --- Collect elements & classify by centroid into inner/outer ---
    if mesh_parameters.gradient_mask is not None:
        # Get elements of the (onlength_x) geometric surface
        types, elemTags, nodeTags = gmsh.model.mesh.getElements(2, surf_tag)
        # Build a node coordinate map
        node_ids, node_xyz, _ = gmsh.model.mesh.getNodes()
        # node_ids may be unsorted; build dict id->(x,y,z)
        coords = np.array(node_xyz).reshape(-1, 3)
        id2idx = {int(nid): i for i, nid in enumerate(node_ids)}

        # Small rectangle bounds
        z_min = mesh_parameters.gradient_mask["z_min"]
        z_max = mesh_parameters.gradient_mask["z_max"]
        x_min = mesh_parameters.gradient_mask["x_min"]
        x_max = mesh_parameters.gradient_mask["x_max"]

        # Prepare per-type lists for inner/outer
        inner_elem_by_type = []
        inner_conn_by_type = []
        outer_elem_by_type = []
        outer_conn_by_type = []

        for t, tags, conn in zip(types, elemTags, nodeTags):
            tags = np.array(tags, dtype=np.int64)
            # number of nodes per element type t:
            nPer = gmsh.model.mesh.getElementProperties(t)[3]  # returns (name, dim, order, numNodes, ...)[3]
            conn = np.array(conn, dtype=np.int64).reshape(-1, nPer)

            # Compute centroids
            # (x_c, y_c) = average of node coords
            xyz = coords[[id2idx[int(n)] for n in conn.flatten()]].reshape(-1, nPer, 3)
            centroids = xyz.mean(axis=1)  # (nelem, 3)
            cz_e = centroids[:, 0]
            cx_e = centroids[:, 1]

            inside = (cz_e >= z_min) & (cz_e <= z_max) & (cx_e >= x_min) & (cx_e <= x_max)

            inner_elem_by_type.append(tags[inside])
            inner_conn_by_type.append(conn[inside])

            outer_elem_by_type.append(tags[~inside])
            outer_conn_by_type.append(conn[~inside])

        # --- Move inner elements to a DISCRETE surface entity ---
        # 1) Remove ALL elements from the original geometric surface
        gmsh.model.mesh.removeElements(2, surf_tag)

        # 2) Re-add the outer elements back to the geometric surface
        for t, tags, conn in zip(types, outer_elem_by_type, outer_conn_by_type):
            if tags.size == 0:
                continue
            gmsh.model.mesh.addElements(2, surf_tag, [t], [tags.tolist()], [conn.flatten().tolist()])

        # 3) Create a discrete surface and add the inner elements
        inner_surf_tag = gmsh.model.addDiscreteEntity(2)
        # Pure element set container.
        for t, tags, conn in zip(types, inner_elem_by_type, inner_conn_by_type):
            if tags.size == 0:
                continue
            gmsh.model.mesh.addElements(2, inner_surf_tag, [t], [tags.tolist()], [conn.flatten().tolist()])

        # --- Define Physical groups on entities ---
        pg_outer = gmsh.model.addPhysicalGroup(2, [surf_tag])
        gmsh.model.setPhysicalName(2, pg_outer, "Outer")
        pg_inner = gmsh.model.addPhysicalGroup(2, [inner_surf_tag])
        gmsh.model.setPhysicalName(2, pg_inner, "Inner")

    # Create physical groups for boundary edges (for absorbing boundary conditions)
    for edge_tag, boundary_id in boundary_tag_map.items():
        # Set physical group ID explicitly to match ds() tags (1=top, 2=bottom, 3=right, 4=left)
        pg_boundary = gmsh.model.addPhysicalGroup(1, [edge_tag], boundary_id)  # noqa: F841
        boundary_names = {1: "Top", 2: "Bottom", 3: "Right", 4: "Left"}
        gmsh.model.setPhysicalName(1, boundary_id, f"Boundary_{boundary_names.get(boundary_id, boundary_id)}")
        # This ensures ds(1), ds(2), ds(3), ds(4) work correctly

    # Save
    gmsh.write(outfile)

    print(f"Written mesh to: {outfile}")
    print(f"Boundary tags created: {len(boundary_tag_map)} edges")
    for edge_tag, boundary_id in boundary_tag_map.items():
        boundary_names = {1: "Top", 2: "Bottom", 3: "Right", 4: "Left"}
        print(f"  Edge {edge_tag} -> Boundary {boundary_id} ({boundary_names.get(boundary_id, 'Unknown')})")

    if mesh_parameters.gradient_mask is not None:
        print(f"Geometric surface tag      (Outer): {surf_tag}")
        print(f"Discrete surface tag       (Inner): {inner_surf_tag}")
        print(f"Physical group Outer tag: {pg_outer}")
        print(f"Physical group Inner tag: {pg_inner}")

    gmsh.finalize()
