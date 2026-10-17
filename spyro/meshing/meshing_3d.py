import numpy as np

from .meshing_gmsh3d import (
    apply_structured_winslow_smoothing3D,
    build_gmsh_geometry_and_groups3D,
    configure_gmsh_3d_meshing_mode,
    configure_gmsh_mesh_size3D,
)
from .meshing_utils3d import create_sizing_function3D
from spyro.mpi.spyro_mpi import SpyroEnsemble

try:
    import gmsh
except ImportError:
    gmsh = None


def generate_gmsh_3d_mesh(mesh_parameters) -> str:
    """Generate a 3-D Gmsh mesh file from a binary velocity model.

    The mesh is written to ``mesh_parameters.output_filename`` by ensemble
    rank 0 only.

    Parameters
    ----------
    mesh_parameters : MeshingParameters
        Complete meshing parameters for a 3-D ``gmsh_mesh``.

    Returns
    -------
    str
        Path to the velocity model used for meshing (the binary input, or a
        temporary binary file written from a grid ``velocity_model``).
    """
    if gmsh is None:
        raise ImportError("gmsh is not available. Please install it.")

    if mesh_parameters.dimension != 3:
        raise ValueError(
            "create_gmsh_3D_mesh requires dimension=3, "
            f"not {mesh_parameters.dimension}."
        )

    depth_z = -abs(mesh_parameters.length_z)
    length_x = mesh_parameters.length_x
    length_y = mesh_parameters.length_y
    padding_z = mesh_parameters.padding_z
    padding_x = mesh_parameters.padding_x
    padding_y = mesh_parameters.padding_y
    hyper_n = mesh_parameters.hyper_n
    hmin_segy = mesh_parameters.hmin_segy
    wl = mesh_parameters.cells_per_wavelength
    freq = mesh_parameters.source_frequency
    grade = mesh_parameters.grade
    water_search_value = mesh_parameters.water_search_value
    padding_type = mesh_parameters.padding_type
    output_file = mesh_parameters.output_filename
    water_interface = mesh_parameters.water_interface
    vp_water = mesh_parameters.vp_water
    structured_mesh = mesh_parameters.structured_mesh
    minElementSize = mesh_parameters.min_element_size
    winslow_implementation = mesh_parameters.winslow_implementation
    apply_winslow = mesh_parameters.apply_winslow
    winslow_iterations = mesh_parameters.winslow_iterations
    winslow_omega = mesh_parameters.winslow_omega
    extend_segy = mesh_parameters.extend_segy
    h_padding = mesh_parameters.h_padding
    parallel = mesh_parameters.gmsh_parallel
    gmsh_num_threads = mesh_parameters.gmsh_num_threads

    byte_order = mesh_parameters.segy_byte_order
    axes_order = tuple(mesh_parameters.segy_axes_order)
    axes_order_sort = mesh_parameters.segy_axes_order_sort
    binary_dtype = mesh_parameters.segy_dtype

    if mesh_parameters.segy_velocity_model is not None:
        velocity_model_file = mesh_parameters.segy_velocity_model
        fname = velocity_model_file
        nz = mesh_parameters.segy_nz
        nx = mesh_parameters.segy_nx
        ny = mesh_parameters.segy_ny
        dz = mesh_parameters.segy_dz
        dx = mesh_parameters.segy_dx
        dy = mesh_parameters.segy_dy
        if any(value is None for value in (nz, nx, ny, dz, dx, dy)):
            raise ValueError(
                "3-D binary Gmsh meshing requires segy_nz, segy_nx, "
                "segy_ny, segy_dz, segy_dx and segy_dy."
            )

    elif mesh_parameters.velocity_model is not None:
        velocity_model = mesh_parameters.velocity_model
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
        if vp.ndim != 3:
            raise ValueError(
                "Grid velocity_model['vp_values'] must be a 3-D array, "
                f"but received shape {vp.shape}."
            )
        nz, nx, ny = vp.shape
        dz = abs(depth_z) / max(nz - 1, 1)
        dx = length_x / max(nx - 1, 1)
        dy = length_y / max(ny - 1, 1)
        fname = "tmp_velocity_model3D.bin"
        dtype = np.dtype(binary_dtype).newbyteorder(
            ">" if byte_order == "big" else "<"
        )
        np.asarray(vp, dtype=dtype).ravel(
            order=axes_order_sort
        ).tofile(fname)
        velocity_model_file = fname
    else:
        raise ValueError(
            "Gmsh 3-D meshing requires either 'segy_velocity_model' "
            "or a grid 'velocity_model'."
        )

    if mesh_parameters.comm is None or mesh_parameters.comm.ensemble_comm.rank == 0:
        SpyroEnsemble.print("Generating 3D Gmsh mesh...")

        segy_bbox = (
            depth_z, 0.0,
            0.0, length_x,
            0.0, length_y,
        )

        if padding_type not in (None, "rectangular", "hyperelliptical"):
            raise ValueError(
                "padding_type must be None, 'rectangular', or "
                "'hyperelliptical'."
            )

        ef_segy3, *_ = (
            create_sizing_function3D(
                fname=fname,
                hmin=hmin_segy,
                bbox=segy_bbox,
                wl=wl,
                freq=freq,
                pad_type=padding_type,
                pad_size_x=padding_x,
                pad_size_y=padding_y,
                pad_size_z=padding_z,
                grade=grade,
                vp_water=vp_water,
                nz=nz,
                nx=nx,
                ny=ny,
                byte_order=byte_order,
                axes_order=axes_order,
                axes_order_sort=axes_order_sort,
                dtype=binary_dtype,
            )
        )

        gmsh.initialize()
        gmsh.clear()
        gmsh.option.setNumber("Geometry.Tolerance", 1e-16)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        gmsh.model.add("3d_Velocity_Model_Water_Interface")

        try:
            build_gmsh_geometry_and_groups3D(
                gmsh=gmsh,
                fname=fname,
                length_x=length_x,
                length_y=length_y,
                depth_z=depth_z,
                padding_type=padding_type,
                padding_x=padding_x,
                padding_y=padding_y,
                padding_z=padding_z,
                hyper_n=hyper_n,
                water_interface=water_interface,
                water_search_value=water_search_value,
                structured_mesh=structured_mesh,
                nz=nz,
                nx=nx,
                ny=ny,
                dz=dz,
                dx=dx,
                dy=dy,
                byte_order=byte_order,
                axes_order=axes_order,
                axes_order_sort=axes_order_sort,
                dtype=binary_dtype,
            )

            configure_gmsh_mesh_size3D(
                gmsh=gmsh,
                ef_segy3=ef_segy3,
                bbox=segy_bbox,
                structured_mesh=structured_mesh,
                parallel=parallel,
                extend_segy=extend_segy,
                padding_type=padding_type,
                padding_x=padding_x,
                padding_y=padding_y,
                padding_z=padding_z,
                hyper_n=hyper_n,
                h_padding=h_padding,
                length_x=length_x,
                length_y=length_y,
                depth_z=depth_z,
                nz=nz,
                nx=nx,
                ny=ny,
                gmsh_num_threads=gmsh_num_threads,
            )

            configure_gmsh_3d_meshing_mode(
                gmsh=gmsh,
                structured_mesh=structured_mesh,
                min_element_size=minElementSize,
            )
            gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.generate(3)

            if structured_mesh:
                apply_structured_winslow_smoothing3D(
                    gmsh=gmsh,
                    length_x=length_x,
                    length_y=length_y,
                    depth_z=depth_z,
                    padding_type=padding_type,
                    padding_x=padding_x,
                    padding_y=padding_y,
                    padding_z=padding_z,
                    water_interface=water_interface,
                    winslow_implementation=winslow_implementation,
                    apply_winslow=apply_winslow,
                    winslow_iterations=winslow_iterations,
                    winslow_omega=winslow_omega,
                    ef_segy3=ef_segy3,
                )

            # gmsh builds the model in physical coordinates (x, y, z=depth).
            # Spyro uses (z, x, y)
            gmsh_xyz_to_spyro_zxy = [
                0.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
            gmsh.model.mesh.affineTransform(gmsh_xyz_to_spyro_zxy)
            gmsh.write(output_file)
            SpyroEnsemble.print(f"Gmsh 3D mesh written to {output_file}")
        finally:
            gmsh.finalize()

    return velocity_model_file
