import firedrake as fire
import meshio
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ..io import parallel_print
from ..utils import run_in_one_core

try:
    import gmsh
except ImportError:
    gmsh = None


try:
    import SeismicMesh
except ImportError:
    SeismicMesh = None

try:
    from .meshing_utils import (
        create_sizing_function, generate_water_profile_from_segy,
        get_surface_entities_by_physical_name, get_nodes_on_surface_entities,
        get_water_interface_node_indices, align_water_columns_to_interface_x,
        winslow_smooth_numba, winslow_smooth_vectorized, winslow_smooth_default
    )
except ImportError as e:
    raise ImportError(
        "Could not load internal meshing utilities."
    ) from e


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
                return fire.Mesh(self.output_file_name, comm=self.comm.comm)
            else:
                return fire.Mesh(self.output_file_name)
        elif self.mesh_type == "gmsh_mesh":
            return self.create_gmsh_2D_mesh()
        else:
            raise ValueError("mesh_type is not supported")

    def create_firedrake_mesh(self):
        """
        Creates a mesh based on Firedrake meshing utilities.

        Returns
        -------
        mesh : Firedrake Mesh
            The generated mesh.

        Raises
        ------
        ValueError
            If dimension is not supported (must be 2 or 3).
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

        Returns
        -------
        mesh : Firedrake Mesh
            The generated 2D mesh.

        Notes
        -----
        If edge_length is not specified but cells_per_wavelength (cpw) is provided,
        the edge length will be calculated automatically. The method creates either
        a periodic or non-periodic rectangular mesh based on the periodic attribute.
        """
        if self.edge_length is None and self.cpw is not None:
            self.edge_length = calculate_edge_length(
                self.cpw, self.minimum_velocity, self.source_frequency)
        if self.abc_pad:
            nx = int(round((self.length_x + 2*self.abc_pad) / self.edge_length, 0))
            nz = int(round((self.length_z + self.abc_pad) / self.edge_length, 0))
        else:
            nx = int(round(self.length_x / self.edge_length, 0))
            nz = int(round(self.length_z / self.edge_length, 0))

        if self.comm is not None:
            comm = self.comm.comm
        else:
            comm = None

        if self.periodic:
            return PeriodicRectangleMesh(
                nz,
                nx,
                self.length_z,
                self.length_x,
                quadrilateral=self.quadrilateral,
                comm=comm,
                pad=self.abc_pad,
            )
        else:
            return RectangleMesh(
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
        mesh : Firedrake Mesh
            The generated 3D box mesh.

        Notes
        -----
        Uses the edge_length parameter to determine the number of elements
        in each direction (x, y, z).
        """
        dx = self.edge_length
        nx = int(round(self.length_x / dx, 0))
        nz = int(round(self.length_z / dx, 0))
        ny = int(round(self.length_y / dx, 0))

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
            return self.create_seismicmesh_2D_mesh_homogeneous()
        else:
            return self.create_seismicmesh_2D_mesh_with_velocity_model()

    def create_seismicmesh_2D_mesh_with_velocity_model(self):
        """
        Creates a 2D mesh with velocity-based refinement using SeismicMesh.

        Returns
        -------
        mesh : Firedrake Mesh
            The generated 2D mesh with velocity-based sizing.

        Notes
        -----
        This method uses the velocity model to determine the mesh element sizes,
        ensuring appropriate resolution for wave propagation. The sizing function
        is derived from the velocity model SEGY file. Only the ensemble rank 0
        performs the mesh generation, and the mesh is then distributed across
        all processes.
        """
        if self.comm.ensemble_comm.rank == 0:
            v_min = self.minimum_velocity
            frequency = self.source_frequency

            C = self.cpw

            length_z = self.length_z
            length_x = self.length_x
            domain_pad = self.abc_pad
            lbda_min = v_min/frequency

            bbox = (-length_z, 0.0, 0.0, length_x)
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

        Returns
        -------
        mesh : Firedrake Mesh
            The generated 2D mesh with uniform element sizes.

        Notes
        -----
        This method creates a rectangular mesh with uniform element sizing.
        The edge length is either user-specified or calculated based on the
        minimum velocity, source frequency, and cells per wavelength.
        Boundary entities with low quality are removed to improve mesh quality.
        """
        length_z = self.length_z
        length_x = self.length_x
        pad = self.abc_pad

        if pad is not None:
            real_lz = length_z + pad
            real_lx = length_x + 2 * pad
        else:
            real_lz = length_z
            real_lx = length_x
            pad = 0.0

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

    def create_spyro_mesh(self):
        """
        Creates a mesh using spyro's internal meshing utilities based on gmsh calls. This
        mesh has tags that define the dx integration in Firedrake.

        Returns
        -------
        mesh : Firedrake Mesh
            The generated mesh.

        Raises
        ------
        ValueError
            If dimension is not supported (must be 2).
        NotImplementedError
            If dimension is 3 (3D meshing not yet implemented).

        Notes
        -----
        Currently only 2D meshing is implemented via build_big_rect_with_inner_element_group.
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
        elliptical/rectangular padding, and Winslow smoothing.

        Returns
        -------
        mesh : Firedrake Mesh
            The loaded Firedrake mesh object.
        """
        if gmsh is None:
            raise ImportError("gmsh is not available. Please install it.")

        # MPI Setup: Only generate the mesh on rank 0
        rank = 0
        if self.comm is not None:
            if hasattr(self.comm, 'ensemble_comm'):
                rank = self.comm.ensemble_comm.rank
            else:
                rank = self.comm.comm.rank

        if rank == 0:
            parallel_print("Generating Gmsh mesh on Rank 0...", comm=self.comm)
            
            # --- Map Class Parameters to Local Variables ---
            depth_z = -abs(self.length_z) # Ensure depth is negative for the script math
            length_x = self.length_x
            padding_z = self.mesh_parameters.padding_z
            padding_x = self.mesh_parameters.padding_x
            ellipse_n = self.mesh_parameters.ellipse_n
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
            MinElementSize = self.mesh_parameters.min_element_size
            winslow_implementation = self.mesh_parameters.winslow_implementation
            winslow_iterations = self.mesh_parameters.winslow_iterations
            winslow_omega = self.mesh_parameters.winslow_omega
            extend_segy = self.mesh_parameters.extend_segy
            h_padding = self.mesh_parameters.h_padding
            segy_bbox = (depth_z, 0, 0, length_x)

            # Calculating domain bounding box according to selected padding
            box_xmin = 0.0
            box_xmax = length_x
            box_zmax = 0.0
            box_zmin = depth_z          
            xc = box_xmax / 2.0 
            zc = box_zmin / 2.0     
            
            if padding_type is None: 
                domain_xmin, domain_xmax = 0.0, length_x
                domain_zmax, domain_zmin = depth_z, 0.0

            if padding_type == "elliptical":
                ellipse_a = (box_zmin / 2.0) - padding_z  
                ellipse_b = (box_xmax / 2.0) + padding_x  
                domain_zmin = zc + ellipse_a
                domain_zmax =  0.0
                domain_xmin = xc - ellipse_b 
                domain_xmax = xc + ellipse_b 

            if padding_type == "rectangular":  
                domain_zmin = box_zmin - padding_z 
                domain_zmax =  0.0
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

            if water_interface: 
                Xs, Z_bottom = generate_water_profile_from_segy(
                    fname, z_min=0.0, z_max=depth_z, x_min=0.0, x_max=length_x,
                    value=water_search_value, tolerance=1.0
                )
                
                pB = [gmsh.model.occ.addPoint(x, float(z), 0.0) for x, z in zip(Xs, Z_bottom)]
                bottom_curve = gmsh.model.occ.addSpline(pB)
                gmsh.model.occ.synchronize()
                
                pt_top_left = gmsh.model.occ.addPoint(float(Xs[0]), 0.0, 0.0)
                pt_top_right = gmsh.model.occ.addPoint(float(Xs[-1]), 0.0, 0.0)
                line_left = gmsh.model.occ.addLine(pt_top_left, pB[0])
                line_right = gmsh.model.occ.addLine(pB[-1], pt_top_right)
                line_top = gmsh.model.occ.addLine(pt_top_right, pt_top_left)
                
                curve_loop = gmsh.model.occ.addCurveLoop([line_left, bottom_curve, line_right, line_top])
                water_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
                gmsh.model.occ.synchronize()
                
                if padding_type is None: 
                    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
                    gmsh.model.occ.synchronize()
                    fragment_result, fragment_map = gmsh.model.occ.fragment([(2, rectangle_tag), (2, water_surface)], [])
                    gmsh.model.occ.synchronize()
                    water_tags = [tag for dim, tag in fragment_map[1]]
                    clipped_rect_tags = [tag for dim, tag in fragment_map[0] if tag not in water_tags]
                    gmsh.model.addPhysicalGroup(2, water_tags, name="WaterSurface")
                    gmsh.model.addPhysicalGroup(2, clipped_rect_tags, name="SubSurface")
                
                if padding_type == "rectangular": 
                    pad_x_min, pad_x_max = -padding_x, length_x + padding_x
                    pad_z_min = depth_z - padding_z
                    z_water_L, z_water_R = float(Z_bottom[0]), float(Z_bottom[-1])
                
                    pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
                    pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
                    pt_pad_tl_top = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
                    pt_pad_tl_bot = gmsh.model.occ.addPoint(pad_x_min, z_water_L, 0.0)
                    pt_pad_l_bot  = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
                    pt_pad_tr_top = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)
                    pt_pad_tr_bot = gmsh.model.occ.addPoint(pad_x_max, z_water_R, 0.0)
                    pt_pad_r_bot  = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
                    pt_pad_b_left   = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
                    pt_pad_bc_left  = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
                    pt_pad_bc_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
                    pt_pad_b_right  = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)
                
                    rock_right  = gmsh.model.occ.addLine(pB[-1], pt_rock_br)
                    rock_bottom = gmsh.model.occ.addLine(pt_rock_br, pt_rock_bl)
                    rock_left   = gmsh.model.occ.addLine(pt_rock_bl, pB[0])
                    pad_tr_top   = gmsh.model.occ.addLine(pt_top_right, pt_pad_tr_top)
                    pad_tr_right = gmsh.model.occ.addLine(pt_pad_tr_top, pt_pad_tr_bot)
                    pad_tr_bot   = gmsh.model.occ.addLine(pt_pad_tr_bot, pB[-1])
                    pad_mr_right = gmsh.model.occ.addLine(pt_pad_tr_bot, pt_pad_r_bot)
                    pad_mr_bot   = gmsh.model.occ.addLine(pt_pad_r_bot, pt_rock_br)
                    pad_tl_top   = gmsh.model.occ.addLine(pt_top_left, pt_pad_tl_top)
                    pad_tl_left  = gmsh.model.occ.addLine(pt_pad_tl_top, pt_pad_tl_bot)
                    pad_tl_bot   = gmsh.model.occ.addLine(pt_pad_tl_bot, pB[0])
                    pad_ml_left  = gmsh.model.occ.addLine(pt_pad_tl_bot, pt_pad_l_bot)
                    pad_ml_bot   = gmsh.model.occ.addLine(pt_pad_l_bot, pt_rock_bl)
                    pad_bl_left  = gmsh.model.occ.addLine(pt_pad_l_bot, pt_pad_b_left)
                    pad_bl_bot   = gmsh.model.occ.addLine(pt_pad_b_left, pt_pad_bc_left)
                    pad_bl_right = gmsh.model.occ.addLine(pt_pad_bc_left, pt_rock_bl)
                    pad_bc_bot   = gmsh.model.occ.addLine(pt_pad_bc_left, pt_pad_bc_right)
                    pad_bc_right = gmsh.model.occ.addLine(pt_pad_bc_right, pt_rock_br)
                    pad_br_bot   = gmsh.model.occ.addLine(pt_pad_bc_right, pt_pad_b_right)
                    pad_br_right = gmsh.model.occ.addLine(pt_pad_b_right, pt_pad_r_bot)
                
                    surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([bottom_curve, rock_right, rock_bottom, rock_left])])
                    surf_pad_tr = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, pad_tr_top, pad_tr_right, pad_tr_bot])])
                    surf_pad_mr = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_tr_bot, pad_mr_right, pad_mr_bot, -rock_right])])
                    surf_pad_tl = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([pad_tl_top, pad_tl_left, pad_tl_bot, -line_left])])
                    surf_pad_ml = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_tl_bot, pad_ml_left, pad_ml_bot, rock_left])])
                    surf_pad_bl = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_ml_bot, pad_bl_left, pad_bl_bot, pad_bl_right])])
                    surf_pad_bc = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_bl_right, pad_bc_bot, pad_bc_right, rock_bottom])])
                    surf_pad_br = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_bc_right, pad_br_bot, pad_br_right, pad_mr_bot])])
                
                    gmsh.model.occ.synchronize()
                    gmsh.model.occ.removeAllDuplicates()
                    gmsh.model.occ.synchronize()
                
                    gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
                    gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
                    gmsh.model.addPhysicalGroup(2, [surf_pad_tr, surf_pad_mr, surf_pad_tl, surf_pad_ml, surf_pad_bl, surf_pad_bc, surf_pad_br], name="Padding")
                
                if padding_type == "elliptical":
                    a_val = (length_x / 2.0) + padding_x
                    b_val = abs(depth_z / 2.0) + padding_z
                    
                    def intersect(x0, z0, dx, dz):
                        def f(s):
                            x, z = x0 + s * dx, z0 + s * dz
                            return (abs(x - xc) / a_val)**ellipse_n + (abs(z - zc) / b_val)**ellipse_n - 1.0
                        s_low, s_high = 0.0, 1.0
                        while f(s_high) < 0: s_high *= 2.0
                        for _ in range(100):
                            s_mid = (s_low + s_high) / 2.0
                            if f(s_mid) > 0: s_high = s_mid
                            else: s_low = s_mid
                        return x0 + s_mid * dx, z0 + s_mid * dz
                
                    z_water_L, z_water_R = float(Z_bottom[0]), float(Z_bottom[-1])
                    pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
                    pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
                    z_mid_L, z_mid_R = (z_water_L + depth_z) / 2.0, (z_water_R + depth_z) / 2.0
                    x_mid_L, x_mid_R = length_x * 0.25, length_x * 0.75 
                    
                    pt_mid_L = gmsh.model.occ.addPoint(0.0, z_mid_L, 0.0)
                    pt_mid_R = gmsh.model.occ.addPoint(length_x, z_mid_R, 0.0)
                    pt_bot_midL = gmsh.model.occ.addPoint(x_mid_L, depth_z, 0.0)
                    pt_bot_midR = gmsh.model.occ.addPoint(x_mid_R, depth_z, 0.0)
                
                    x_O_TL, z_O_TL = intersect(0.0, 0.0, -1, 0)
                    x_O_WL, z_O_WL = intersect(0.0, z_water_L, -1, 0)
                    x_O_ML, z_O_ML = intersect(0.0, z_mid_L, -1, 0)
                    x_O_BL_45, z_O_BL_45 = intersect(0.0, depth_z, -1, -1)
                    x_O_BML, z_O_BML = intersect(x_mid_L, depth_z, 0, -1)
                    x_O_BMR, z_O_BMR = intersect(x_mid_R, depth_z, 0, -1)
                    x_O_BR_45, z_O_BR_45 = intersect(length_x, depth_z, 1, -1) 
                    x_O_MR, z_O_MR = intersect(length_x, z_mid_R, 1, 0)
                    x_O_WR, z_O_WR = intersect(length_x, z_water_R, 1, 0)
                    x_O_TR, z_O_TR = intersect(length_x, 0.0, 1, 0)
                
                    pt_O_TL = gmsh.model.occ.addPoint(x_O_TL, z_O_TL, 0.0)
                    pt_O_WL = gmsh.model.occ.addPoint(x_O_WL, z_O_WL, 0.0)
                    pt_O_ML = gmsh.model.occ.addPoint(x_O_ML, z_O_ML, 0.0)
                    pt_O_BL_45 = gmsh.model.occ.addPoint(x_O_BL_45, z_O_BL_45, 0.0)
                    pt_O_BML = gmsh.model.occ.addPoint(x_O_BML, z_O_BML, 0.0)
                    pt_O_BMR = gmsh.model.occ.addPoint(x_O_BMR, z_O_BMR, 0.0)
                    pt_O_BR_45 = gmsh.model.occ.addPoint(x_O_BR_45, z_O_BR_45, 0.0)
                    pt_O_MR = gmsh.model.occ.addPoint(x_O_MR, z_O_MR, 0.0)
                    pt_O_WR = gmsh.model.occ.addPoint(x_O_WR, z_O_WR, 0.0)
                    pt_O_TR = gmsh.model.occ.addPoint(x_O_TR, z_O_TR, 0.0)
                
                    rock_R_upper = gmsh.model.occ.addLine(pB[-1], pt_mid_R)
                    rock_R_lower = gmsh.model.occ.addLine(pt_mid_R, pt_rock_br)
                    rock_B_right = gmsh.model.occ.addLine(pt_rock_br, pt_bot_midR)
                    rock_B_mid   = gmsh.model.occ.addLine(pt_bot_midR, pt_bot_midL)
                    rock_B_left  = gmsh.model.occ.addLine(pt_bot_midL, pt_rock_bl)
                    rock_L_lower = gmsh.model.occ.addLine(pt_rock_bl, pt_mid_L)
                    rock_L_upper = gmsh.model.occ.addLine(pt_mid_L, pB[0])
                
                    ray_TL    = gmsh.model.occ.addLine(pt_top_left, pt_O_TL)
                    ray_WL    = gmsh.model.occ.addLine(pB[0], pt_O_WL)
                    ray_ML    = gmsh.model.occ.addLine(pt_mid_L, pt_O_ML)
                    ray_BL_45 = gmsh.model.occ.addLine(pt_rock_bl, pt_O_BL_45)
                    ray_BML   = gmsh.model.occ.addLine(pt_bot_midL, pt_O_BML)
                    ray_BMR   = gmsh.model.occ.addLine(pt_bot_midR, pt_O_BMR)
                    ray_BR_45 = gmsh.model.occ.addLine(pt_rock_br, pt_O_BR_45)
                    ray_MR    = gmsh.model.occ.addLine(pt_mid_R, pt_O_MR)
                    ray_WR    = gmsh.model.occ.addLine(pB[-1], pt_O_WR)
                    ray_TR    = gmsh.model.occ.addLine(pt_top_right, pt_O_TR)
                
                    def make_arc(p1_tag, p2_tag, x1, z1, x2, z2, num_pts=25):
                        def get_theta(x, z):
                            vx, vz = (x - xc) / a_val, (z - zc) / b_val
                            vx = vx if abs(vx) > 1e-12 else 0.0
                            vz = vz if abs(vz) > 1e-12 else 0.0
                            return np.arctan2(np.sign(vz) * np.abs(vz)**(ellipse_n/2.0), 
                                              np.sign(vx) * np.abs(vx)**(ellipse_n/2.0))
                        t1, t2 = get_theta(x1, z1), get_theta(x2, z2)
                        if t2 - t1 > np.pi: t1 += 2 * np.pi
                        elif t1 - t2 > np.pi: t2 += 2 * np.pi
                        pts = [p1_tag]
                        for t in np.linspace(t1, t2, num_pts)[1:-1]:
                            cos_t, sin_t = np.cos(t), np.sin(t)
                            x = xc + a_val * np.sign(cos_t) * np.abs(cos_t)**(2.0/ellipse_n)
                            z = zc + b_val * np.sign(sin_t) * np.abs(sin_t)**(2.0/ellipse_n)
                            pts.append(gmsh.model.occ.addPoint(x, z, 0.0))
                        pts.append(p2_tag)
                        return gmsh.model.occ.addSpline(pts)
                
                    arc_TL_WL    = make_arc(pt_O_TL, pt_O_WL, x_O_TL, z_O_TL, x_O_WL, z_O_WL)
                    arc_WL_ML    = make_arc(pt_O_WL, pt_O_ML, x_O_WL, z_O_WL, x_O_ML, z_O_ML)
                    arc_ML_BL45  = make_arc(pt_O_ML, pt_O_BL_45, x_O_ML, z_O_ML, x_O_BL_45, z_O_BL_45)
                    arc_BL45_BML = make_arc(pt_O_BL_45, pt_O_BML, x_O_BL_45, z_O_BL_45, x_O_BML, z_O_BML)
                    arc_BML_BMR  = make_arc(pt_O_BML, pt_O_BMR, x_O_BML, z_O_BML, x_O_BMR, z_O_BMR)
                    arc_BMR_BR45 = make_arc(pt_O_BMR, pt_O_BR_45, x_O_BMR, z_O_BMR, x_O_BR_45, z_O_BR_45)
                    arc_BR45_MR  = make_arc(pt_O_BR_45, pt_O_MR, x_O_BR_45, z_O_BR_45, x_O_MR, z_O_MR)
                    arc_MR_WR    = make_arc(pt_O_MR, pt_O_WR, x_O_MR, z_O_MR, x_O_WR, z_O_WR)
                    arc_WR_TR    = make_arc(pt_O_WR, pt_O_TR, x_O_WR, z_O_WR, x_O_TR, z_O_TR)
                
                    surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([bottom_curve, rock_R_upper, rock_R_lower, rock_B_right, rock_B_mid, rock_B_left, rock_L_lower, rock_L_upper])])
                    surf_pad_TL  = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_left, ray_WL, -arc_TL_WL, -ray_TL])])
                    surf_pad_ML1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_L_upper, ray_ML, -arc_WL_ML, -ray_WL])])
                    surf_pad_ML2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_L_lower, ray_BL_45, -arc_ML_BL45, -ray_ML])])
                    surf_pad_B_L = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_left, ray_BML, -arc_BL45_BML, -ray_BL_45])])
                    surf_pad_B_M = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_mid, ray_BMR, -arc_BML_BMR, -ray_BML])])
                    surf_pad_B_R = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_right, ray_BR_45, -arc_BMR_BR45, -ray_BMR])])
                    surf_pad_MR2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_R_lower, ray_MR, -arc_BR45_MR, -ray_BR_45])])
                    surf_pad_MR1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_R_upper, ray_WR, -arc_MR_WR, -ray_MR])])
                    surf_pad_TR  = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, ray_TR, -arc_WR_TR, -ray_WR])])
                
                    gmsh.model.occ.synchronize()
                    gmsh.model.occ.removeAllDuplicates()
                    gmsh.model.occ.synchronize()
                
                    gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
                    gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
                    gmsh.model.addPhysicalGroup(2, [surf_pad_TL, surf_pad_ML1, surf_pad_ML2, surf_pad_B_L, surf_pad_B_M, surf_pad_B_R, surf_pad_MR2, surf_pad_MR1, surf_pad_TR], name="Padding")
                
                if structured_mesh and padding_type == "elliptical":
                    len_radial = max(padding_x, padding_z)
                    len_x_left, len_x_mid, len_x_right = length_x * 0.25, length_x * 0.50, length_x * 0.25
                    len_z_water = abs(z_water_L)
                    len_z_rock_upper, len_z_rock_lower = abs(z_mid_L - z_water_L), abs(depth_z - z_mid_L)
                
                    N_radial = max(2, int(np.ceil(len_radial / MinElementSize )) + 1)
                    N_X_left  = max(2, int(np.ceil(len_x_left / MinElementSize)) + 1) 
                    N_X_mid   = max(2, int(np.ceil(len_x_mid / MinElementSize)) + 1) 
                    N_X_right = max(2, int(np.ceil(len_x_right / MinElementSize)) + 1) 
                    N_X_total = N_X_left + N_X_mid + N_X_right - 2  
                    N_Z_water      = max(2, int(np.ceil(len_z_water / MinElementSize)) + 1) 
                    N_Z_rock_upper = max(2, int(np.ceil(len_z_rock_upper / MinElementSize)) + 1) 
                    N_Z_rock_lower = max(2, int(np.ceil(len_z_rock_lower / MinElementSize)) + 1) 
                    
                    for curve in [ray_TL, ray_WL, ray_ML, ray_BL_45, ray_BML, ray_BMR, ray_BR_45, ray_MR, ray_WR, ray_TR]: gmsh.model.mesh.setTransfiniteCurve(curve, N_radial)
                    for curve in [line_left, line_right, arc_TL_WL, arc_WR_TR]: gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_water)
                    for curve in [rock_L_upper, rock_R_upper, arc_WL_ML, arc_MR_WR]: gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_rock_upper)
                    for curve in [rock_L_lower, rock_R_lower, arc_ML_BL45, arc_BR45_MR]: gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_rock_lower)
                    for curve in [rock_B_left, arc_BL45_BML]: gmsh.model.mesh.setTransfiniteCurve(curve, N_X_left)
                    for curve in [rock_B_mid, arc_BML_BMR]: gmsh.model.mesh.setTransfiniteCurve(curve, N_X_mid)
                    for curve in [rock_B_right, arc_BMR_BR45]: gmsh.model.mesh.setTransfiniteCurve(curve, N_X_right)
                    for curve in [line_top, bottom_curve]: gmsh.model.mesh.setTransfiniteCurve(curve, N_X_total)
                
                    padding_surfs = [surf_pad_TL, surf_pad_ML1, surf_pad_ML2, surf_pad_B_L, surf_pad_B_M, surf_pad_B_R, surf_pad_MR2, surf_pad_MR1, surf_pad_TR]
                    for surf in padding_surfs:
                        gmsh.model.mesh.setTransfiniteSurface(surf)
                        gmsh.model.mesh.setRecombine(2, surf)
                        
                    gmsh.model.mesh.setTransfiniteSurface(water_surface, cornerTags=[pt_top_left, pt_top_right, pB[-1], pB[0]])
                    gmsh.model.mesh.setRecombine(2, water_surface)
                    gmsh.model.mesh.setTransfiniteSurface(surf_rock, cornerTags=[pB[0], pB[-1], pt_rock_br, pt_rock_bl])
                    gmsh.model.mesh.setRecombine(2, surf_rock)

            if not water_interface:
                if padding_type is None: 
                    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
                    gmsh.model.occ.synchronize()
                    gmsh.model.addPhysicalGroup(2, [rectangle_tag], name="SubSurface")

                if padding_type == "rectangular": 
                    pad_x_min, pad_x_max = -padding_x, length_x + padding_x
                    pad_z_min = depth_z - padding_z

                    pt_tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
                    pt_tr = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
                    pt_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
                    pt_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

                    pt_pad_tl = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
                    pt_pad_bl = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
                    pt_pad_b_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
                    pt_pad_bc_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
                    pt_pad_bc_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
                    pt_pad_b_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)
                    pt_pad_br = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
                    pt_pad_tr = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)

                    line_top = gmsh.model.occ.addLine(pt_tl, pt_tr)
                    line_right = gmsh.model.occ.addLine(pt_tr, pt_br)
                    line_bottom = gmsh.model.occ.addLine(pt_br, pt_bl)
                    line_left = gmsh.model.occ.addLine(pt_bl, pt_tl)

                    pad_top_left = gmsh.model.occ.addLine(pt_pad_tl, pt_tl)
                    pad_left = gmsh.model.occ.addLine(pt_pad_bl, pt_pad_tl)
                    pad_bot_left_horiz = gmsh.model.occ.addLine(pt_bl, pt_pad_bl)

                    pad_corner_bl_left = gmsh.model.occ.addLine(pt_pad_b_left, pt_pad_bl)
                    pad_corner_bl_bot = gmsh.model.occ.addLine(pt_pad_bc_left, pt_pad_b_left)
                    pad_bot_left_vert = gmsh.model.occ.addLine(pt_bl, pt_pad_bc_left)

                    pad_bot_mid = gmsh.model.occ.addLine(pt_pad_bc_right, pt_pad_bc_left)
                    pad_bot_right_vert = gmsh.model.occ.addLine(pt_br, pt_pad_bc_right)

                    pad_corner_br_bot = gmsh.model.occ.addLine(pt_pad_b_right, pt_pad_bc_right)
                    pad_corner_br_right = gmsh.model.occ.addLine(pt_pad_br, pt_pad_b_right)

                    pad_bot_right_horiz = gmsh.model.occ.addLine(pt_pad_br, pt_br)
                    pad_right = gmsh.model.occ.addLine(pt_pad_tr, pt_pad_br)
                    pad_top_right = gmsh.model.occ.addLine(pt_tr, pt_pad_tr)

                    loop_internal = gmsh.model.occ.addCurveLoop([line_top, line_right, line_bottom, line_left])
                    surf_internal = gmsh.model.occ.addPlaneSurface([loop_internal])
                    loop_pad_left = gmsh.model.occ.addCurveLoop([pad_top_left, -line_left, -pad_bot_left_horiz, pad_left])
                    surf_pad_left = gmsh.model.occ.addPlaneSurface([loop_pad_left])
                    loop_pad_bl = gmsh.model.occ.addCurveLoop([pad_bot_left_horiz, -pad_corner_bl_left, -pad_corner_bl_bot, -pad_bot_left_vert])
                    surf_pad_bl = gmsh.model.occ.addPlaneSurface([loop_pad_bl])
                    loop_pad_bot = gmsh.model.occ.addCurveLoop([pad_bot_left_vert, -pad_bot_mid, -pad_bot_right_vert, line_bottom])
                    surf_pad_bot = gmsh.model.occ.addPlaneSurface([loop_pad_bot])
                    loop_pad_br = gmsh.model.occ.addCurveLoop([pad_bot_right_vert, -pad_corner_br_bot, -pad_corner_br_right, -pad_bot_right_horiz])
                    surf_pad_br = gmsh.model.occ.addPlaneSurface([loop_pad_br])
                    loop_pad_right = gmsh.model.occ.addCurveLoop([line_right, -pad_bot_right_horiz, -pad_right, -pad_top_right])
                    surf_pad_right = gmsh.model.occ.addPlaneSurface([loop_pad_right])

                    gmsh.model.occ.synchronize()
                    gmsh.model.occ.removeAllDuplicates()
                    gmsh.model.occ.synchronize()

                    gmsh.model.addPhysicalGroup(2, [surf_internal], name="SubSurface")
                    gmsh.model.addPhysicalGroup(2, [surf_pad_left, surf_pad_bl, surf_pad_bot, surf_pad_br, surf_pad_right], name="Padding")

                if padding_type == "elliptical":
                    a_val = (length_x / 2.0) + padding_x
                    b_val = abs(depth_z / 2.0) + padding_z
                    
                    def intersect(x0, z0, dx, dz):
                        def f(s):
                            x, z = x0 + s * dx, z0 + s * dz
                            return (abs(x - xc) / a_val)**ellipse_n + (abs(z - zc) / b_val)**ellipse_n - 1.0
                        s_low, s_high = 0.0, 1.0
                        while f(s_high) < 0: s_high *= 2.0
                        for _ in range(100):
                            s_mid = (s_low + s_high) / 2.0
                            if f(s_mid) > 0: s_high = s_mid
                            else: s_low = s_mid
                        return x0 + s_mid * dx, z0 + s_mid * dz

                    pt_tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
                    pt_tr = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
                    pt_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
                    pt_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

                    x_mid_L, x_mid_R = length_x * 0.25, length_x * 0.75 
                    pt_bot_midL = gmsh.model.occ.addPoint(x_mid_L, depth_z, 0.0)
                    pt_bot_midR = gmsh.model.occ.addPoint(x_mid_R, depth_z, 0.0)

                    x_O_TL, z_O_TL = intersect(0.0, 0.0, -1, 0)
                    x_O_BL, z_O_BL = intersect(0.0, depth_z, -1, -1)
                    x_O_BML, z_O_BML = intersect(x_mid_L, depth_z, 0, -1)
                    x_O_BMR, z_O_BMR = intersect(x_mid_R, depth_z, 0, -1)
                    x_O_BR, z_O_BR = intersect(length_x, depth_z, 1, -1) 
                    x_O_TR, z_O_TR = intersect(length_x, 0.0, 1, 0)

                    pt_O_TL = gmsh.model.occ.addPoint(x_O_TL, z_O_TL, 0.0)
                    pt_O_BL = gmsh.model.occ.addPoint(x_O_BL, z_O_BL, 0.0)
                    pt_O_BML = gmsh.model.occ.addPoint(x_O_BML, z_O_BML, 0.0)
                    pt_O_BMR = gmsh.model.occ.addPoint(x_O_BMR, z_O_BMR, 0.0)
                    pt_O_BR = gmsh.model.occ.addPoint(x_O_BR, z_O_BR, 0.0)
                    pt_O_TR = gmsh.model.occ.addPoint(x_O_TR, z_O_TR, 0.0)

                    line_top = gmsh.model.occ.addLine(pt_tl, pt_tr)
                    line_right = gmsh.model.occ.addLine(pt_tr, pt_br)
                    line_bot_right = gmsh.model.occ.addLine(pt_br, pt_bot_midR)
                    line_bot_mid   = gmsh.model.occ.addLine(pt_bot_midR, pt_bot_midL)
                    line_bot_left  = gmsh.model.occ.addLine(pt_bot_midL, pt_bl)
                    line_left = gmsh.model.occ.addLine(pt_bl, pt_tl)

                    ray_TL = gmsh.model.occ.addLine(pt_tl, pt_O_TL)
                    ray_BL = gmsh.model.occ.addLine(pt_bl, pt_O_BL)
                    ray_BML = gmsh.model.occ.addLine(pt_bot_midL, pt_O_BML)
                    ray_BMR = gmsh.model.occ.addLine(pt_bot_midR, pt_O_BMR)
                    ray_BR = gmsh.model.occ.addLine(pt_br, pt_O_BR)
                    ray_TR = gmsh.model.occ.addLine(pt_tr, pt_O_TR)

                    def make_arc(p1_tag, p2_tag, x1, z1, x2, z2, num_pts=25):
                        def get_theta(x, z):
                            vx, vz = (x - xc) / a_val, (z - zc) / b_val
                            vx = vx if abs(vx) > 1e-12 else 0.0
                            vz = vz if abs(vz) > 1e-12 else 0.0
                            return np.arctan2(np.sign(vz) * np.abs(vz)**(ellipse_n/2.0), 
                                              np.sign(vx) * np.abs(vx)**(ellipse_n/2.0))
                        t1, t2 = get_theta(x1, z1), get_theta(x2, z2)
                        if t2 - t1 > np.pi: t1 += 2 * np.pi
                        elif t1 - t2 > np.pi: t2 += 2 * np.pi
                        pts = [p1_tag]
                        for t in np.linspace(t1, t2, num_pts)[1:-1]:
                            cos_t, sin_t = np.cos(t), np.sin(t)
                            x = xc + a_val * np.sign(cos_t) * np.abs(cos_t)**(2.0/ellipse_n)
                            z = zc + b_val * np.sign(sin_t) * np.abs(sin_t)**(2.0/ellipse_n)
                            pts.append(gmsh.model.occ.addPoint(x, z, 0.0))
                        pts.append(p2_tag)
                        return gmsh.model.occ.addSpline(pts)

                    arc_TL_BL  = make_arc(pt_O_TL, pt_O_BL, x_O_TL, z_O_TL, x_O_BL, z_O_BL)
                    arc_BL_BML = make_arc(pt_O_BL, pt_O_BML, x_O_BL, z_O_BL, x_O_BML, z_O_BML)
                    arc_BML_BMR= make_arc(pt_O_BML, pt_O_BMR, x_O_BML, z_O_BML, x_O_BMR, z_O_BMR)
                    arc_BMR_BR = make_arc(pt_O_BMR, pt_O_BR, x_O_BMR, z_O_BMR, x_O_BR, z_O_BR)
                    arc_BR_TR  = make_arc(pt_O_BR, pt_O_TR, x_O_BR, z_O_BR, x_O_TR, z_O_TR)

                    surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_top, line_right, line_bot_right, line_bot_mid, line_bot_left, line_left])])
                    surf_pad_left = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_left, ray_TL, arc_TL_BL, -ray_BL])])
                    surf_pad_bot_left = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_left, ray_BL, arc_BL_BML, -ray_BML])])
                    surf_pad_bot_mid = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_mid, ray_BML, arc_BML_BMR, -ray_BMR])])
                    surf_pad_bot_right = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_right, ray_BMR, arc_BMR_BR, -ray_BR])])
                    surf_pad_right = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, ray_BR, arc_BR_TR, -ray_TR])])
                    
                    gmsh.model.occ.synchronize()
                    gmsh.model.occ.removeAllDuplicates()
                    gmsh.model.occ.synchronize()
                    
                    gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
                    gmsh.model.addPhysicalGroup(2, [surf_pad_left, surf_pad_bot_left, surf_pad_bot_mid, surf_pad_bot_right, surf_pad_right], name="Padding")

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
                        t = min((tx**ellipse_n + ty**ellipse_n)**(1.0 / ellipse_n) if padding_type == "elliptical" else max(tx, ty), 1.0)
                        return float(base_size + t * (h_padding - base_size))
                        
            gmsh.model.mesh.setSizeCallback(mesh_size_callback)
            gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            if structured_mesh and padding_type != "elliptical":
                gmsh.option.setNumber('Mesh.MeshSizeMin', MinElementSize)
                gmsh.option.setNumber('Mesh.MeshSizeMax', MinElementSize)
                gmsh.model.mesh.setTransfiniteAutomatic()
            gmsh.model.mesh.generate(2)

            if structured_mesh:
                if padding_type is None:
                    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                    points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)
                    points_2d = points_3d[:, :2]
                
                    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
                    all_quad_nodes = []
                    for i, t_type in enumerate(elem_types):
                        if t_type == 3:  
                            all_quad_nodes.extend(elem_node_tags[i])
                    if not all_quad_nodes:
                        raise ValueError("No quadrilaterals found! Smoothing requires quads.")
                
                    tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
                    quads = np.asarray([tag_to_index[tag] for tag in all_quad_nodes], dtype=np.int64).reshape(-1, 4)
                
                    move_all = set()
                    move_X_only = set()
                    move_Z_only = set()
                    locked = set()
                    tol = 1e-3
                
                    if water_interface:
                        water_surface_entities = get_surface_entities_by_physical_name("WaterSurface")
                        water_surface_nodes = get_nodes_on_surface_entities(tag_to_index, water_surface_entities)
                        interface_nodes = get_water_interface_node_indices(
                            tag_to_index=tag_to_index, water_surface_entities=water_surface_entities, length_x=length_x, tol=1e-8
                        )
                        parallel_print("Aligning water columns with water spline X positions...", comm=self.comm)
                        points_2d, n_snapped, n_cols = align_water_columns_to_interface_x(
                            points_2d=points_2d, water_surface_nodes=water_surface_nodes, interface_nodes=interface_nodes, quads=quads
                        )
                        parallel_print(f"Snapped {n_snapped} water-surface nodes onto {n_cols} spline-X columns.", comm=self.comm)
                    else:
                        interface_nodes = set()
                
                    corners_to_lock = [(0.0, 0.0), (length_x, 0.0), (0.0, depth_z), (length_x, depth_z)]
                
                    for i, pt in enumerate(points_2d):
                        x, z = pt
                        is_locked = False
                        for cx, cz in corners_to_lock:
                            if abs(x - cx) < tol and abs(z - cz) < tol:
                                locked.add(i); is_locked = True; break
                        if is_locked: continue
                
                        if water_interface and i in interface_nodes:
                            locked.add(i)
                            continue
                
                        if abs(x - 0.0) < tol or abs(x - length_x) < tol:
                            move_Z_only.add(i)
                            continue
                        if abs(z - 0.0) < tol or abs(z - depth_z) < tol:
                            move_X_only.add(i)
                            continue
                
                        move_all.add(i)
                
                    total_movable = move_all | move_X_only | move_Z_only
                
                    parallel_print(f"Nodes Breakdown | Total: {len(points_2d)}", comm=self.comm)
                    parallel_print(f"Move All: {len(move_all)} | X-Slide: {len(move_X_only)} | Z-Slide: {len(move_Z_only)} | Locked: {len(locked)}", comm=self.comm)
                
                    parallel_print("Applying Winslow smoothing...", comm=self.comm)

                    if winslow_implementation in ("fast", "numba"):
                        # Shared Grid Setup
                        nx_grid, nz_grid = n_samples, n_traces
                        segy_grid_x = np.linspace(domain_xmin, domain_xmax, nx_grid)
                        segy_grid_z = np.linspace(domain_zmin, domain_zmax, nz_grid)
                        X_grid, Z_grid = np.meshgrid(segy_grid_x, segy_grid_z, indexing='ij')
                        sizes_flat = ef_segy2(X_grid.flatten(), Z_grid.flatten())
                        segy_grid_vals = sizes_flat.reshape((nx_grid, nz_grid))

                        if winslow_implementation == "fast":
                            smoothed_points_2d = winslow_smooth_vectorized(
                                points=points_2d, 
                                quads=quads, 
                                segy_grid_x=segy_grid_x,       
                                segy_grid_z=segy_grid_z,       
                                segy_grid_vals=segy_grid_vals, 
                                move_all=move_all, 
                                move_X_only=move_X_only, 
                                move_Z_only=move_Z_only,
                                iterations=winslow_iterations, 
                                omega=winslow_omega
                            )
                        elif winslow_implementation == "numba":
                            smoothed_points_2d = winslow_smooth_numba(
                                points=points_2d, 
                                quads=quads, 
                                segy_grid_x=segy_grid_x,       
                                segy_grid_z=segy_grid_z,       
                                segy_grid_vals=segy_grid_vals, 
                                move_all=move_all, 
                                move_X_only=move_X_only, 
                                move_Z_only=move_Z_only,
                                iterations=winslow_iterations, 
                                omega=winslow_omega
                            )
                    else:
                        smoothed_points_2d = winslow_smooth_default(
                            points=points_2d, 
                            quads=quads, 
                            sizing_fn=ef_segy2,
                            move_all=move_all, 
                            move_X_only=move_X_only, 
                            move_Z_only=move_Z_only,
                            iterations=winslow_iterations, 
                            omega=winslow_omega
                        )
                
                    smoothed_points_3d = np.zeros_like(points_3d)
                    smoothed_points_3d[:, :2] = smoothed_points_2d
                
                    parallel_print("Updating nodes in Gmsh...", comm=self.comm)
                    for i, tag in enumerate(node_tags):
                        gmsh.model.mesh.setNode(int(tag), smoothed_points_3d[i].tolist(), [])
                        
                elif padding_type == "rectangular":
                    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                    points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)
                    points_2d = points_3d[:, :2]  
                    
                    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
                    all_quad_nodes = []
                    for i, t_type in enumerate(elem_types):
                        if t_type == 3:  
                            all_quad_nodes.extend(elem_node_tags[i])
                    
                    if not all_quad_nodes:
                        raise ValueError("No quadrilaterals found! Winslow requires quads.")
                    
                    tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
                    quads = np.asarray([tag_to_index[tag] for tag in all_quad_nodes], dtype=np.int64).reshape(-1, 4)
                    
                    if water_interface:
                        # Apply Water Surface Alignment Before Smoothing 
                        water_surface_entities = get_surface_entities_by_physical_name("WaterSurface")
                        water_surface_nodes = get_nodes_on_surface_entities(tag_to_index, water_surface_entities)
                        interface_nodes = get_water_interface_node_indices(
                            tag_to_index=tag_to_index, water_surface_entities=water_surface_entities, length_x=length_x, tol=1e-8
                        )
                        parallel_print("Aligning water columns with water spline X positions...", comm=self.comm)
                        points_2d, n_snapped, n_cols = align_water_columns_to_interface_x(
                            points_2d=points_2d, water_surface_nodes=water_surface_nodes, interface_nodes=interface_nodes, quads=quads
                        )
                        parallel_print(f"Snapped {n_snapped} water-surface nodes onto {n_cols} spline-X columns.", comm=self.comm)
                    else:
                        water_surface_nodes = set()
                    
                    move_all = set()
                    move_X_only = set()
                    move_Z_only = set()
                    locked = set()
                    tol = 1e-3
                    
                    if water_interface:
                        corners_to_lock = [
                            (0.0, depth_z), (length_x, depth_z), (pad_x_min, depth_z), (pad_x_max, depth_z),
                            (0.0, pad_z_min), (length_x, pad_z_min), (pad_x_min, pad_z_min), (pad_x_max, pad_z_min)
                        ]
                    else:
                        corners_to_lock = [
                            (0.0, 0.0), (length_x, 0.0), (pad_x_min, 0.0), (pad_x_max, 0.0),
                            (0.0, depth_z), (length_x, depth_z), (pad_x_min, depth_z), (pad_x_max, depth_z),
                            (0.0, pad_z_min), (length_x, pad_z_min), (pad_x_min, pad_z_min), (pad_x_max, pad_z_min)
                        ]
                    
                    for i, pt in enumerate(points_2d):
                        x, z = pt
                        is_locked = False
                        for cx, cz in corners_to_lock:
                            if abs(x - cx) < tol and abs(z - cz) < tol:
                                locked.add(i); is_locked = True; break
                        if is_locked: continue
                        
                        if water_interface:
                            if (x <= 0.0 + tol and z >= z_water_L - tol) or (x >= length_x - tol and z >= z_water_R - tol):
                                locked.add(i)
                                continue
                            if i in water_surface_nodes:
                                locked.add(i)
                                continue
                            
                        if abs(x - pad_x_min) < tol or abs(x - pad_x_max) < tol or abs(x - 0.0) < tol or abs(x - length_x) < tol:
                            move_Z_only.add(i)
                            continue
                            
                        if abs(z - pad_z_min) < tol or abs(z - depth_z) < tol or (not water_interface and abs(z - 0.0) < tol):
                            move_X_only.add(i)
                            continue
                            
                        move_all.add(i)
                    
                    total_movable = move_all | move_X_only | move_Z_only
                    parallel_print(f"Nodes Breakdown | Total: {len(points_2d)}", comm=self.comm)
                    parallel_print(f"Move All: {len(move_all)} | X-Slide: {len(move_X_only)} | Z-Slide: {len(move_Z_only)} | Locked: {len(locked)}", comm=self.comm)
                    
                    parallel_print(f"Applying Winslow smoothing...", comm=self.comm)
                    
                    if winslow_implementation in ("fast", "numba"):
                        # Shared Grid Setup
                        nx_grid, nz_grid = n_samples, n_traces
                        segy_grid_x = np.linspace(domain_xmin, domain_xmax, nx_grid)
                        segy_grid_z = np.linspace(domain_zmin, domain_zmax, nz_grid)
                        X_grid, Z_grid = np.meshgrid(segy_grid_x, segy_grid_z, indexing='ij')
                        sizes_flat = ef_segy2(X_grid.flatten(), Z_grid.flatten())
                        segy_grid_vals = sizes_flat.reshape((nx_grid, nz_grid))

                        if winslow_implementation == "fast":
                            smoothed_points_2d = winslow_smooth_vectorized(
                                points=points_2d, 
                                quads=quads, 
                                segy_grid_x=segy_grid_x,       
                                segy_grid_z=segy_grid_z,       
                                segy_grid_vals=segy_grid_vals, 
                                move_all=move_all, 
                                move_X_only=move_X_only, 
                                move_Z_only=move_Z_only,
                                iterations=winslow_iterations, 
                                omega=winslow_omega
                            )
                        elif winslow_implementation == "numba":
                            smoothed_points_2d = winslow_smooth_numba(
                                points=points_2d, 
                                quads=quads, 
                                segy_grid_x=segy_grid_x,       
                                segy_grid_z=segy_grid_z,       
                                segy_grid_vals=segy_grid_vals, 
                                move_all=move_all, 
                                move_X_only=move_X_only, 
                                move_Z_only=move_Z_only,
                                iterations=winslow_iterations, 
                                omega=winslow_omega
                            )
                    else:
                        smoothed_points_2d = winslow_smooth_default(
                            points=points_2d, 
                            quads=quads, 
                            sizing_fn=ef_segy2,
                            move_all=move_all, 
                            move_X_only=move_X_only, 
                            move_Z_only=move_Z_only,
                            iterations=winslow_iterations, 
                            omega=winslow_omega
                        )
                    smoothed_points_3d = np.zeros_like(points_3d)
                    smoothed_points_3d[:, :2] = smoothed_points_2d
                    
                    parallel_print("Updating nodes in Gmsh...", comm=self.comm)
                    for i, tag in enumerate(node_tags):
                        gmsh.model.mesh.setNode(int(tag), smoothed_points_3d[i].tolist(), [])
                        
                elif padding_type == "elliptical":
                    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                    points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)
                    points_2d = points_3d[:, :2]  
                    
                    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
                    all_quad_nodes = []
                    for i, t_type in enumerate(elem_types):
                        if t_type == 3:  
                            all_quad_nodes.extend(elem_node_tags[i])
                    
                    if not all_quad_nodes:
                        raise ValueError("No quadrilaterals found! Winslow requires quads.")
                    
                    tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
                    quads = np.asarray([tag_to_index[tag] for tag in all_quad_nodes], dtype=np.int64).reshape(-1, 4)
                    
                    if water_interface:
                        # Apply Water Surface Alignment Before Smoothing 
                        water_surface_entities = get_surface_entities_by_physical_name("WaterSurface")
                        water_surface_nodes = get_nodes_on_surface_entities(tag_to_index, water_surface_entities)
                        interface_nodes = get_water_interface_node_indices(
                            tag_to_index=tag_to_index, water_surface_entities=water_surface_entities, length_x=length_x, tol=1e-8
                        )
                        parallel_print("Aligning water columns with water spline X positions...", comm=self.comm)
                        points_2d, n_snapped, n_cols = align_water_columns_to_interface_x(
                            points_2d=points_2d, water_surface_nodes=water_surface_nodes, interface_nodes=interface_nodes, quads=quads
                        )
                        parallel_print(f"Snapped {n_snapped} water-surface nodes onto {n_cols} spline-X columns.", comm=self.comm)

                    move_all = set()
                    move_X_only = set()
                    move_Z_only = set()
                    move_ellipse = set()
                    locked = set()

                    def get_nodes(dim, tags):
                        nodes = set()
                        for t in tags:
                            n, _, _ = gmsh.model.mesh.getNodes(dim, t, includeBoundary=True)
                            if len(n) > 0: nodes.update([tag_to_index[node] for node in n])
                        return nodes

                    if water_interface:
                        locked_surfs = [water_surface, surf_pad_TL, surf_pad_TR]
                        locked_surface_nodes = get_nodes(2, locked_surfs)
                        outer_middle_arcs = [arc_WL_ML, arc_ML_BL45, arc_BL45_BML, arc_BML_BMR, arc_BMR_BR45, arc_BR45_MR, arc_MR_WR]
                        z_slide_curves = [rock_L_lower, rock_L_upper, rock_R_lower, rock_R_upper]
                        x_slide_curves = [rock_B_left, rock_B_mid, rock_B_right]
                        diag_rays = [ray_BL_45, ray_BR_45]
                    else:
                        locked_surface_nodes = set()
                        outer_middle_arcs = [arc_TL_BL, arc_BL_BML, arc_BML_BMR, arc_BMR_BR, arc_BR_TR]
                        z_slide_curves = [line_left, line_right]
                        x_slide_curves = [line_bot_left, line_bot_mid, line_bot_right]
                        diag_rays = [ray_TL, ray_BL, ray_BR, ray_TR]

                    outer_arc_nodes = get_nodes(1, outer_middle_arcs)
                    z_slide_nodes = get_nodes(1, z_slide_curves)
                    x_slide_nodes = get_nodes(1, x_slide_curves)
                    locked_diag_nodes = get_nodes(1, diag_rays)

                    corner_indices = set()
                    tol = 1e-3
                    for i, pt in enumerate(points_2d):
                        x, z = pt
                        if water_interface:
                            if (abs(x - 0.0) < tol and abs(z - depth_z) < tol) or \
                               (abs(x - length_x) < tol and abs(z - depth_z) < tol) or \
                               (abs(x - 0.0) < tol and abs(z - z_water_L) < tol) or \
                               (abs(x - length_x) < tol and abs(z - z_water_R) < tol):
                                corner_indices.add(i)
                        else:
                            if (abs(x - 0.0) < tol and abs(z - depth_z) < tol) or \
                               (abs(x - length_x) < tol and abs(z - depth_z) < tol) or \
                               (abs(x - 0.0) < tol and abs(z - 0.0) < tol) or \
                               (abs(x - length_x) < tol and abs(z - 0.0) < tol):
                                corner_indices.add(i)

                    for i in range(len(points_2d)):
                        if i in corner_indices: 
                            locked.add(i)
                        elif i in locked_diag_nodes: 
                            locked.add(i)                
                        elif water_interface and i in locked_surface_nodes:
                            if i in x_slide_nodes: move_X_only.add(i)
                            elif i in z_slide_nodes: move_Z_only.add(i)
                            else: locked.add(i) 
                        elif not water_interface and abs(points_2d[i][1] - 0.0) < tol:
                            move_X_only.add(i) # Lock top boundary Z axis, slide in X
                        elif i in outer_arc_nodes: 
                            move_ellipse.add(i)          
                        elif i in x_slide_nodes: 
                            move_X_only.add(i)
                        elif i in z_slide_nodes: 
                            move_Z_only.add(i)
                        else: 
                            move_all.add(i)
                            
                    total_movable = move_all | move_X_only | move_Z_only | move_ellipse
                    
                    parallel_print(f"Nodes Breakdown | Total: {len(points_2d)}", comm=self.comm)
                    parallel_print(f"Move All: {len(move_all)} | X-Slide: {len(move_X_only)} | Z-Slide: {len(move_Z_only)} | Ellipse: {len(move_ellipse)} | Locked: {len(locked)}", comm=self.comm)
                    
                    if len(total_movable) > 0:
                        parallel_print(f"Applying Winslow smoothing...", comm=self.comm)

                    if winslow_implementation in ("fast", "numba"):
                        # Shared Grid Setup
                        nx_grid, nz_grid = n_samples, n_traces
                        segy_grid_x = np.linspace(domain_xmin, domain_xmax, nx_grid)
                        segy_grid_z = np.linspace(domain_zmin, domain_zmax, nz_grid)
                        X_grid, Z_grid = np.meshgrid(segy_grid_x, segy_grid_z, indexing='ij')
                        sizes_flat = ef_segy2(X_grid.flatten(), Z_grid.flatten())
                        segy_grid_vals = sizes_flat.reshape((nx_grid, nz_grid))

                        if winslow_implementation == "fast":
                            smoothed_points_2d = winslow_smooth_vectorized(
                                points=points_2d, 
                                quads=quads, 
                                segy_grid_x=segy_grid_x,       
                                segy_grid_z=segy_grid_z,       
                                segy_grid_vals=segy_grid_vals, 
                                move_all=move_all, 
                                move_X_only=move_X_only, 
                                move_Z_only=move_Z_only,
                                move_ellipse=move_ellipse,                        
                                ellipse_params=(a_val, b_val, xc, zc, ellipse_n),
                                iterations=winslow_iterations, 
                                omega=winslow_omega
                            )
                        elif winslow_implementation == "numba":
                            smoothed_points_2d = winslow_smooth_numba(
                                points=points_2d, 
                                quads=quads, 
                                segy_grid_x=segy_grid_x,       
                                segy_grid_z=segy_grid_z,       
                                segy_grid_vals=segy_grid_vals, 
                                move_all=move_all, 
                                move_X_only=move_X_only, 
                                move_Z_only=move_Z_only,
                                move_ellipse=move_ellipse,                        
                                ellipse_params=(a_val, b_val, xc, zc, ellipse_n),
                                iterations=winslow_iterations, 
                                omega=winslow_omega
                            )
                    else:
                        smoothed_points_2d = winslow_smooth_default(
                            points=points_2d, 
                            quads=quads, 
                            sizing_fn=ef_segy2,
                            move_all=move_all, 
                            move_X_only=move_X_only, 
                            move_Z_only=move_Z_only,
                            move_ellipse=move_ellipse,                        
                            ellipse_params=(a_val, b_val, xc, zc, ellipse_n),
                            iterations=winslow_iterations, 
                            omega=winslow_omega
                        )        
                        
                    smoothed_points_3d = np.zeros_like(points_3d)
                    smoothed_points_3d[:, :2] = smoothed_points_2d
                        
                    parallel_print("Updating nodes in Gmsh...", comm=self.comm)
                        
                    for i, tag in enumerate(node_tags):
                        gmsh.model.mesh.setNode(int(tag), smoothed_points_3d[i].tolist(), [])

            gmsh.write(output_file)
            parallel_print(f"Gmsh mesh written to {output_file}", comm=self.comm)
            gmsh.finalize()

        # MPI Sync
        if self.comm is not None:
            if hasattr(self.comm, 'ensemble_comm'):
                self.comm.ensemble_comm.barrier()
            self.comm.comm.barrier()
            parallel_print("Loading mesh into Firedrake.", comm=self.comm)
            return fire.Mesh(self.output_file_name, comm=self.comm.comm)
        else:
            return fire.Mesh(self.output_file_name)

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
    v_min = minimum_velocity

    lbda_min = v_min/frequency

    edge_length = lbda_min/cpw
    return edge_length


def RectangleMesh(nx, ny, length_x, length_y, pad=None, comm=None, quadrilateral=False):
    """Create a rectangle mesh based on the Firedrake mesh.

    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

    Parameters
    ----------
    length_x: float
      Length of the domain in the x direction.
    length_y: float
      Length of the domain in the y direction.
    nx: int
      Number of elements in the x direction.
    ny: int
      Number of elements in the y direction.
    pad: float, optional
      Padding to be added to the domain. The default is None.
    comm: MPI communicator, optional
      MPI communicator. The default is None.
    quadrilateral: bool, optional
      If True, the mesh is quadrilateral. The default is False.

    Returns
    -------
    mesh: Firedrake Mesh
      Mesh
    """
    if pad is not None:
        length_x += pad
        length_y += 2 * pad
    else:
        pad = 0

    if comm is None:
        mesh = fire.RectangleMesh(nx, ny, length_x, length_y, quadrilateral=quadrilateral)
    else:
        mesh = fire.RectangleMesh(nx, ny, length_x, length_y, quadrilateral=quadrilateral, comm=comm)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def PeriodicRectangleMesh(
    nx, ny, length_x, length_y, pad=None, comm=None, quadrilateral=False
):
    """Create a periodic rectangle mesh based on the Firedrake mesh.

    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

    Parameters
    ----------
    length_x : float
        Length of the domain in the x direction.
    length_y : float
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
        length_x += pad
        length_y += 2 * pad
    else:
        pad = 0
    mesh = fire.PeriodicRectangleMesh(
        nx, ny, length_x, length_y, quadrilateral=quadrilateral, comm=comm
    )
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def BoxMesh(nx, ny, nz, length_x, length_y, length_z, pad=None, quadrilateral=False):
    """
    Create a 3D box mesh based on Firedrake mesh utilities.

    Parameters
    ----------
    nx : int
        Number of elements in the x direction.
    ny: int
        Number of elements in the y direction.
    nz: int
        Number of elements in the z direction.
    length_x: float
        Length of the domain in the x direction.
    length_y: float
        Length of the domain in the y direction.
    length_z: float
        Length of the domain in the z direction.
    pad: float, optional
        Padding to be added to the domain. The default is None.
    quadrilateral: bool, optional
        If True, the mesh is created by extruding a quadrilateral mesh.
        The default is False.

    Returns
    -------
    mesh: Firedrake Mesh
        The generated 3D box mesh.

    Notes
    -----
    The first coordinate is negated(multiplied by - 1) to match the expected
    coordinate system. If quadrilateral is True, the mesh is created by
    extruding a 2D quadrilateral mesh in the z direction.
    """
    if pad is not None:
        length_x += pad
        length_y += 2 * pad
        length_z += 2 * pad
    else:
        pad = 0
    if quadrilateral:
        quad_mesh = fire.RectangleMesh(nx, ny, length_x, length_y, quadrilateral=quadrilateral)
        quad_mesh.coordinates.dat.data[:, 0] *= -1.0
        quad_mesh.coordinates.dat.data[:, 1] -= pad
        layer_height = length_z / nz
        mesh = fire.ExtrudedMesh(quad_mesh, nz, layer_height=layer_height)
    else:
        mesh = fire.BoxMesh(nx, ny, nz, length_x, length_y, length_z)
        mesh.coordinates.dat.data[:, 0] *= -1.0

    return mesh


def vp_to_sizing(vp, cpw, frequency):
    """
    Convert velocity field to mesh sizing function.

    Parameters
    ----------
    vp: numpy.ndarray
        P-wave velocity field.
    cpw: float
        Cells per wavelength(must be positive).
    frequency: float
        Source frequency in Hz(must be positive).

    Returns
    -------
    sizing: numpy.ndarray
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
    mesh_parameters: object
        Object containing mesh parameters with the following attributes:
        - length_z: float
            Length of domain in z direction.
        - length_x: float
            Length of domain in x direction.
        - output_filename: str
            Path for output mesh file.
        - edge_length: float, optional
            Uniform edge length(if grid_velocity_data is None).
        - grid_velocity_data: dict, optional
            Dictionary with 'vp_values' key containing velocity field for
            adaptive meshing.
        - source_frequency: float, optional
            Source frequency for adaptive meshing.
        - cells_per_wavelength: float, optional
            Cells per wavelength for adaptive meshing.
        - gradient_mask: dict, optional
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

