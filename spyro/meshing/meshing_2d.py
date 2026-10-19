import numpy as np

from ..io.segy_io import create_segy_from_grid
from spyro.mpi.spyro_mpi import SpyroEnsemble
from .meshing_gmsh2d import (
    build_gmsh_geometry_and_groups,
    apply_structured_winslow_smoothing2d,
    configure_gmsh_2d_meshing_mode,
)
from .meshing_utils import create_sizing_function

try:
    import gmsh
except ImportError:
    gmsh = None


def generate_gmsh_2d_mesh(mesh_parameters) -> str:
    """Generate a 2-D Gmsh mesh file from a velocity model.

    Supports an optional water interface, hyperelliptical or rectangular
    padding, structured Winslow smoothing, or unstructured
    full-quadrilateral meshing. The mesh is written to
    ``mesh_parameters.output_filename`` by ensemble rank 0 only.

    Parameters
    ----------
    mesh_parameters : MeshingParameters
        Complete meshing parameters for a 2-D ``gmsh_mesh``.

    Returns
    -------
    str
        Path to the velocity model used for meshing (the SEG-Y input, or a
        temporary SEG-Y file written from a grid ``velocity_model``).
    """
    if gmsh is None:
        raise ImportError("gmsh is not available. Please install it.")

    if mesh_parameters.segy_velocity_model is not None:
        velocity_model_file = mesh_parameters.segy_velocity_model

    elif mesh_parameters.velocity_model is not None:
        velocity_model = mesh_parameters.velocity_model

        if not isinstance(velocity_model, dict):
            raise TypeError(
                "velocity_model must be a grid dictionary when "
                "segy_velocity_model is not provided."
            )

        if "vp_values" not in velocity_model:
            raise ValueError("Grid velocity_model must contain the key 'vp_values'.")

        vp = np.asarray(velocity_model["vp_values"])

        if vp.ndim != 2:
            raise ValueError(
                "Grid velocity_model['vp_values'] must be a 2-D array, "
                f"but received shape {vp.shape}."
            )

        vp_for_gmsh = np.ascontiguousarray(vp)

        filename = "tmp_velocity_model.segy"
        create_segy_from_grid(vp_for_gmsh, filename)

        velocity_model_file = filename

    else:
        raise ValueError(
            "Gmsh meshing requires either 'segy_velocity_model' "
            "or a grid 'velocity_model'."
        )

    if mesh_parameters.comm is None or mesh_parameters.comm.ensemble_comm.rank == 0:
        SpyroEnsemble.print("Generating Gmsh mesh...")

        depth_z = -abs(mesh_parameters.length_z)
        length_x = mesh_parameters.length_x
        padding_z = mesh_parameters.padding_z
        padding_x = mesh_parameters.padding_x
        hyper_n = mesh_parameters.hyper_n
        fname = velocity_model_file
        hmin_segy = mesh_parameters.hmin_segy
        cells_per_wavelength = mesh_parameters.cells_per_wavelength
        freq = mesh_parameters.source_frequency
        grade = mesh_parameters.grade
        water_search_value = mesh_parameters.water_search_value
        padding_type = mesh_parameters.padding_type
        output_file = mesh_parameters.output_filename
        water_interface = mesh_parameters.water_interface
        vp_water = mesh_parameters.vp_water
        structured_mesh = mesh_parameters.structured_mesh
        unstructured_quad_mesh = mesh_parameters.unstructured_quad_mesh
        min_element_size = mesh_parameters.min_element_size
        winslow_implementation = mesh_parameters.winslow_implementation
        apply_winslow = mesh_parameters.apply_winslow
        winslow_iterations = mesh_parameters.winslow_iterations
        winslow_omega = mesh_parameters.winslow_omega
        extend_segy = mesh_parameters.extend_segy
        h_padding = mesh_parameters.h_padding
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
        ef_segy2, _, _, n_samples, n_traces = create_sizing_function(
            fname=fname,
            hmin=hmin_segy,
            bbox=segy_bbox,
            wl=cells_per_wavelength,
            freq=freq,
            pad_type=padding_type,
            pad_size_x=padding_x,
            pad_size_z=padding_z,
            grade=grade,
            vp_water=vp_water,
        )

        gmsh.initialize()
        gmsh.option.setNumber("Geometry.Tolerance", 1e-16)
        gmsh.model.add("seismic_model")

        geom_params = build_gmsh_geometry_and_groups(
            gmsh=gmsh,
            fname=fname,
            length_x=length_x,
            depth_z=depth_z,
            padding_type=padding_type,
            padding_x=padding_x,
            padding_z=padding_z,
            hyper_n=hyper_n,
            water_interface=water_interface,
            water_search_value=water_search_value,
            structured_mesh=structured_mesh,
            min_element_size=min_element_size,
        )

        # Standard Params
        z_water_left = geom_params.get("z_water_left")
        z_water_right = geom_params.get("z_water_right")
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
                    x_proj, y_proj = min(max(x, x_min_segy), x_max_segy), min(
                        max(y, z_min_segy), z_max_segy
                    )
                    base_size = float(
                        ef_segy2(np.array([x_proj]), np.array([y_proj]))[0]
                    )
                    tx = abs(x - x_proj) / padding_x if padding_x > 0 else 0.0
                    ty = abs(y - y_proj) / padding_z if padding_z > 0 else 0.0
                    t = min(
                        (
                            (tx**hyper_n + ty**hyper_n) ** (1.0 / hyper_n)
                            if padding_type == "hyperelliptical"
                            else max(tx, ty)
                        ),
                        1.0,
                    )
                    return float(base_size + t * (h_padding - base_size))

        gmsh.model.mesh.setSizeCallback(mesh_size_callback)
        gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        configure_gmsh_2d_meshing_mode(
            gmsh=gmsh,
            structured_mesh=structured_mesh,
            unstructured_quad_mesh=unstructured_quad_mesh,
            padding_type=padding_type,
            min_element_size=min_element_size,
        )
        gmsh.model.mesh.generate(2)
        if unstructured_quad_mesh:
            gmsh.model.mesh.optimize(
                "QuadQuasiStructured",
                force=False,
                niter=50,
            )

        if structured_mesh:
            apply_structured_winslow_smoothing2d(
                gmsh=gmsh,
                geom_params=geom_params,
                length_x=length_x,
                depth_z=depth_z,
                padding_type=padding_type,
                water_interface=water_interface,
                hyper_n=hyper_n,
                winslow_implementation=winslow_implementation,
                winslow_iterations=winslow_iterations,
                winslow_omega=winslow_omega,
                n_samples=n_samples,
                n_traces=n_traces,
                domain_xmin=domain_xmin,
                domain_xmax=domain_xmax,
                domain_zmin=domain_zmin,
                domain_zmax=domain_zmax,
                ef_segy2=ef_segy2,
                z_water_left=z_water_left,
                z_water_right=z_water_right,
                pad_x_min=pad_x_min,
                pad_x_max=pad_x_max,
                pad_z_min=pad_z_min,
                a_val=a_val,
                b_val=b_val,
                xc=xc,
                zc=zc,
                apply_winslow=apply_winslow,
            )
        # Rotating mesh for axis (z,x,y)
        gmsh_xy_to_spyro_zx = [
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        gmsh.model.mesh.affineTransform(gmsh_xy_to_spyro_zx)
        gmsh.write(output_file)
        SpyroEnsemble.print(f"Gmsh mesh written to {output_file}")
        gmsh.finalize()

    return velocity_model_file
