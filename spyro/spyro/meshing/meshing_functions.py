import numpy as np
from firedrake import Mesh as FireMeshReader
from ..io.basicio import parallel_print
from ..io.segy_io import create_segy_from_grid
from .meshing_gmsh2d import build_gmsh_geometry_and_groups, apply_structured_winslow_smoothing2d
from .meshing_utils import create_sizing_function, calculate_edge_length
from .firedrake_based_wrappers import rectangle_mesh, periodic_rectangle_mesh, box_mesh
from .seismic_mesh_based_wrappers import create_seismicmesh_2D_mesh_with_velocity_model
from .seismic_mesh_based_wrappers import create_seismicmesh_2D_mesh_homogeneous
from .gmsh_based_methods import build_big_rect_with_inner_element_group

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
        if (
            self.edge_length is None
            and self.edge_length_z is None
            and self.edge_length_x is None
            and self.edge_length_y is None
            and self.cpw is not None
        ):
            self.edge_length = calculate_edge_length(
                self.cpw, self.minimum_velocity, self.source_frequency)

        edge_length_z = self._resolved_edge_length("z")
        edge_length_x = self._resolved_edge_length("x")
        edge_length_y = self._resolved_edge_length("y")

        # Number of elements
        pad = 0. if self.abc_pad is None else self.abc_pad
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
            comm=comm)

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

        if self.mesh_parameters.segy_velocity_model is not None:
            self.velocity_model = self.mesh_parameters.segy_velocity_model

        elif self.mesh_parameters.velocity_model is not None:
            velocity_model = self.mesh_parameters.velocity_model

            if not isinstance(velocity_model, dict):
                raise TypeError(
                    "velocity_model must be a grid dictionary when "
                    "segy_velocity_model is not provided."
                )

            if "vp_values" not in velocity_model:
                raise ValueError(
                    "Grid velocity_model must contain the key 'vp_values'."
                )

            vp = np.asarray(velocity_model["vp_values"])

            if vp.ndim != 2:
                raise ValueError(
                    "Grid velocity_model['vp_values'] must be a 2-D array, "
                    f"but received shape {vp.shape}."
                )

            vp_for_gmsh = np.ascontiguousarray(vp[::-1, :])

            filename = "tmp_velocity_model.segy"
            create_segy_from_grid(vp_for_gmsh, filename)

            self.velocity_model = filename

        else:
            raise ValueError(
                "Gmsh meshing requires either 'segy_velocity_model' "
                "or a grid 'velocity_model'."
            )

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
