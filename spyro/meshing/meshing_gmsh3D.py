import numpy as np

from .meshing_gmsh_3d_functions import (
    boundary_faces_of_volume,
    checked_mesh_size,
    create_hyperellipsoid_volume,
    generate_rectangular_padding_no_water,
    generate_structured_rectangular_padding_water,
    generate_water_interface_volumes,
)
from .meshing_utils3D import define_winslow_points_3d
from .meshing_winslow3D import run_selected_winslow


def build_gmsh_geometry_and_groups3D(
    gmsh,
    fname,
    length_x,
    length_y,
    depth_z,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    hyper_n,
    water_interface,
    water_search_value,
    structured_mesh,
    minElementSize,
    nz,
    nx,
    ny,
    dz,
    dx,
    dy,
    byte_order="big",
    axes_order=(0, 1, 2),
    axes_order_sort="F",
    dtype="float32",
    *,
    comm=None,
    parallel_print,
):
    """Build the three-dimensional Gmsh geometry and physical groups for all supported padding and water cases.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    fname : str or pathlib.Path
        Path to the binary velocity model.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_type : str or None
        Padding geometry, either ``None`` or ``"rectangular"`` for Winslow.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    hyper_n : float
        Hyperellipsoid exponent used by the padding geometry and sizing law.
    water_interface : bool
        Whether the water/subsurface interface is geometrically delimited.
    water_search_value : float
        Velocity value used to identify the water layer.
    structured_mesh : bool
        Whether a structured hexahedral mesh is requested.
    minElementSize : float or None
        Minimum element-size input retained by the geometry API.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    dz : float
        Velocity-model spacing in the z direction.
    dx : float
        Velocity-model spacing in the x direction.
    dy : float
        Velocity-model spacing in the y direction.
    byte_order : {"big", "little"}
        Byte order of the velocity-model binary file.
    axes_order : tuple of int
        Permutation mapping binary axes to Spyro ``(z, x, y)`` order.
    axes_order_sort : {"C", "F"}
        Memory order used to reshape the binary velocity data.
    dtype : str or numpy.dtype
        Numeric data type stored in the velocity-model file.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    dict
        Geometry metadata and volume tags required by mesh generation and smoothing.

    Notes
    -----
    Supported padding types are ``None``, ``"rectangular"``, and ``"hyperelliptical"``.
    Hyperelliptical padding is restricted to unstructured meshing.
    """

    if padding_type not in (None, "rectangular", "hyperelliptical"):
        raise ValueError(
            "padding_type must be None, 'rectangular', "
            "or 'hyperelliptical'."
        )

    # Hyperelliptical padding
    if padding_type == "hyperelliptical" and structured_mesh:
        raise ValueError(
            "Hyperelliptical 3-D padding currently supports only "
            "structured_mesh=False. Winslow smoothing is not available "
            "for this geometry."
        )

    if water_interface and tuple(axes_order) != (0, 1, 2):
        raise ValueError(
            "The supplied water-interface geometry reads the binary directly "
            "in (z, x, y) order; axes_order must therefore be (0, 1, 2)."
        )

    box_xmin = 0.0
    box_xmax = length_x
    box_ymin = 0.0
    box_ymax = length_y
    box_zmin = -abs(depth_z)
    box_zmax = 0.0
    ellipse_a = length_x / 2.0 + padding_x
    ellipse_b = length_y / 2.0 + padding_y
    ellipse_c = abs(depth_z) / 2.0 + padding_z
    xc = length_x / 2.0
    yc = length_y / 2.0
    zc = -abs(depth_z) / 2.0

    z_min, z_max = depth_z, 0.0
    x_min, x_max = 0.0, length_x
    y_min, y_max = 0.0, length_y

    if not water_interface and padding_type is None:
        # No padding
        parallel_print(
            "Generating undelimited water+subsurface rectangular domain (no padding)...",
            comm=comm,
        )
        occ = gmsh.model.occ
        vol_subsoil = occ.addBox(
            x_min,
            y_min,
            z_min,
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )
        occ.synchronize()

        gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="SubsurfaceAndWater")
        vol_water = None

    elif not water_interface and padding_type == "rectangular":
        parallel_print(
            "Generating undelimited water+subsurface domain "
            "with rectangular padding...",
            comm=comm,
        )
        vol_subsoil, vol_water = generate_rectangular_padding_no_water(
            gmsh,
            length_x,
            length_y,
            depth_z,
            padding_x,
            padding_y,
            padding_z,
        )

    elif not water_interface and padding_type == "hyperelliptical":
        # Hyperelliptical padding
        parallel_print(
            "Generating undelimited water+subsurface domain with "
            "hyperelliptical padding...",
            comm=comm,
        )
        occ = gmsh.model.occ

        cube_volume_tag = occ.addBox(
            box_xmin,
            box_ymin,
            box_zmin,
            box_xmax - box_xmin,
            box_ymax - box_ymin,
            box_zmax - box_zmin,
        )
        ellipsoid_tag = create_hyperellipsoid_volume(
            gmsh=gmsh,
            a=ellipse_a,
            b=ellipse_b,
            c=ellipse_c,
            n=hyper_n,
            xc=xc,
            yc=yc,
            zc=zc,
            comm=comm,
            parallel_print=parallel_print,
        )
        if ellipsoid_tag is None:
            raise RuntimeError("Could not create the hyperelliptical outer volume.")

        occ.synchronize()
        _fragment_result, fragment_map = occ.fragment(
            [(3, ellipsoid_tag)],
            [(3, cube_volume_tag)],
            removeObject=True,
            removeTool=True,
        )
        occ.synchronize()

        ellipsoid_fragments = {tag for dim, tag in fragment_map[0] if dim == 3}
        cube_fragments = {tag for dim, tag in fragment_map[1] if dim == 3}

        if not cube_fragments:
            raise RuntimeError(
                "The internal water+subsurface rectangle was lost during "
                "the hyperellipsoid fragment operation."
            )

        padding_fragments = sorted(ellipsoid_fragments - cube_fragments)
        core_fragments = sorted(cube_fragments)

        if not padding_fragments:
            raise RuntimeError(
                "The hyperellipsoid fragment produced no padding volume."
            )

        vol_subsoil = core_fragments[0]
        vol_water = None

        # both water and subsurface.
        gmsh.model.addPhysicalGroup(3, core_fragments, name="SubsurfaceAndWater")
        gmsh.model.addPhysicalGroup(3, padding_fragments, name="Padding")

        ellipsoid_faces = set()
        for volume_tag in padding_fragments:
            ellipsoid_faces.update(boundary_faces_of_volume(gmsh, volume_tag))

        for surface_tag in ellipsoid_faces:
            gmsh.model.mesh.setAlgorithm(2, int(surface_tag), 1)

    else:
        # Cases 4-6: water delimitation enabled.
        vol_subsoil, vol_water = generate_water_interface_volumes(
            gmsh=gmsh,
            fname=fname,
            water_search_value=water_search_value,
            nz=nz,
            nx=nx,
            ny=ny,
            dz=dz,
            dx=dx,
            dy=dy,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            byte_order=byte_order,
            order=axes_order_sort,
            dtype=dtype,
            comm=comm,
            parallel_print=parallel_print,
        )

        if padding_type is None:

            gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="Subsurface")
            gmsh.model.addPhysicalGroup(3, [vol_water], name="Water")
        if padding_type == "hyperelliptical":
            cube_volume_tag = gmsh.model.occ.addBox(
                box_xmin,
                box_ymin,
                box_zmin,  # x, y, z of corner
                box_xmax - box_xmin,  # width in x
                box_ymax - box_ymin,  # width in y
                box_zmax - box_zmin,  # width in z
            )

            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            ellipsoid_tag = create_hyperellipsoid_volume(
                gmsh,
                a=ellipse_a,
                b=ellipse_b,
                c=ellipse_c,
                n=hyper_n,
                xc=xc,
                yc=yc,
                zc=zc,
                comm=comm,
                parallel_print=parallel_print,
            )
            gmsh.model.occ.synchronize()
            gmsh.model.occ.fragment(
                [
                    (3, ellipsoid_tag),
                    (3, cube_volume_tag),
                    (3, vol_subsoil),
                    (3, vol_water),
                ],
                [],  # Tool volumes (empty for self-fragmentation)
                removeObject=True,
                removeTool=False,
            )
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(3)
            volume_tags = [tag for dim, tag in volumes if dim == 3]

            water_volumes = [volume_tags[1]] if len(volume_tags) > 0 else []
            ellipsoid_volumes = [volume_tags[2]] if len(volume_tags) > 1 else []
            cube_volumes = [volume_tags[0]] if len(volume_tags) > 2 else []

            gmsh.model.addPhysicalGroup(3, cube_volumes, name="Subsurface")
            gmsh.model.addPhysicalGroup(3, water_volumes, name="Water")
            gmsh.model.addPhysicalGroup(3, ellipsoid_volumes, name="Padding")

            ellipsoid_faces = []
            for v in ellipsoid_volumes:
                ellipsoid_faces.extend(boundary_faces_of_volume(gmsh, v))
            ellipsoid_faces = set(ellipsoid_faces)
            algo_id = 1.0
            for surf_tag in ellipsoid_faces:
                gmsh.model.mesh.setAlgorithm(2, int(surf_tag), int(algo_id))
        if padding_type == "rectangular":
            generate_structured_rectangular_padding_water(
                gmsh=gmsh,
                vol_water=vol_water,
                vol_subsoil=vol_subsoil,
                padding_x=padding_x,
                padding_y=padding_y,
                padding_z=padding_z,
                comm=comm,
                parallel_print=parallel_print,
            )

    gmsh.model.occ.synchronize()
    return {
        "vol_subsoil": vol_subsoil,
        "vol_water": vol_water,
        "padding_type3D": padding_type,
        "water_interface": bool(water_interface),
        "water_delimited": bool(water_interface),
        "contains_water_from_velocity_model": True,
        "structured_mesh_supported": padding_type != "hyperelliptical",
        "winslow_supported": padding_type != "hyperelliptical",
        "ellipse_a": ellipse_a,
        "ellipse_b": ellipse_b,
        "ellipse_c": ellipse_c,
        "xc": xc,
        "yc": yc,
        "zc": zc,
    }


def configure_gmsh_mesh_size3D(
    gmsh,
    ef_segy3,
    bbox,
    structured_mesh,
    parallel,
    extend_segy,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    hyper_n,
    h_padding,
    length_x,
    length_y,
    depth_z,
    nz,
    nx,
    ny,
    *,
    comm=None,
    parallel_print,
):
    """Configure serial or parallel Gmsh mesh sizing from the velocity-model sizing function.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    ef_segy3 : callable
        Three-dimensional mesh-sizing function in ``(z, x, y)`` coordinates.
    bbox : sequence of float
        Bounding box ordered as ``(zmin, zmax, xmin, xmax, ymin, ymax)``.
    structured_mesh : bool
        Whether a structured hexahedral mesh is requested.
    parallel : bool
        Whether to build a point-cloud background field for parallel meshing.
    extend_segy : bool
        Whether velocity-model sizing is extended by nearest-edge projection.
    padding_type : str or None
        Padding geometry, either ``None`` or ``"rectangular"`` for Winslow.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    hyper_n : float
        Hyperellipsoid exponent used by the padding geometry and sizing law.
    h_padding : float
        Target element size at the outer padding boundary.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    None
        The Gmsh sizing callback or background field is configured in place.
    """
    ef_segy2 = ef_segy3

    if structured_mesh is False:
        if parallel is False:
            callback_calls = 0

            def mesh_size_callback3D(dim, tag, x, y, z, lc):
                """Evaluate the serial Gmsh mesh-size callback at one mesh point.

                Parameters
                ----------
                dim : int
                    Dimension of the Gmsh entity requesting the size.
                tag : int
                    Tag of the Gmsh entity requesting the size.
                x : float
                    X coordinate of the Gmsh sizing query.
                y : float
                    Y coordinate of the Gmsh sizing query.
                z : float
                    Z coordinate of the Gmsh sizing query.
                lc : float
                    Element size proposed by Gmsh before applying the callback.

                Returns
                -------
                float
                    Element size assigned to the queried mesh point.
                """
                nonlocal callback_calls
                callback_calls += 1

                if extend_segy:
                    # Edge-extended sizing

                    coords = np.array([[z, x, y]], dtype=float)
                    return checked_mesh_size(
                        ef_segy2,
                        coords,
                        "extend_segy=True",
                    )
                else:
                    (
                        z_min_segy,
                        z_max_segy,
                        x_min_segy,
                        x_max_segy,
                        y_min_segy,
                        y_max_segy,
                    ) = bbox

                    in_x = (x >= x_min_segy) and (x <= x_max_segy)
                    in_y = (y >= y_min_segy) and (y <= y_max_segy)
                    in_z = (z >= z_min_segy) and (z <= z_max_segy)

                    if in_x and in_y and in_z:
                        coords = np.array([[z, x, y]], dtype=float)
                        return checked_mesh_size(
                            ef_segy2,
                            coords,
                            "inside velocity model",
                        )

                    else:

                        x_proj = min(max(x, x_min_segy), x_max_segy)
                        y_proj = min(max(y, y_min_segy), y_max_segy)
                        z_proj = min(max(z, z_min_segy), z_max_segy)

                        coords_proj = np.array(
                            [[z_proj, x_proj, y_proj]],
                            dtype=float,
                        )
                        base_size = checked_mesh_size(
                            ef_segy2,
                            coords_proj,
                            "projected padding boundary",
                        )

                        dx = abs(x - x_proj)
                        dy = abs(y - y_proj)
                        dz = abs(z - z_proj)

                        tx = dx / padding_x if padding_x > 0 else 0.0
                        ty = (
                            dy / padding_y if padding_y > 0 else 0.0
                        )
                        tz = dz / padding_z if padding_z > 0 else 0.0

                        if padding_type == "hyperelliptical":
                            t = (tx**hyper_n + ty**hyper_n + tz**hyper_n) ** (
                                1.0 / hyper_n
                            )
                        else:
                            t = max(tx, ty, tz)

                        t = min(t, 1.0)  # Clamp between 0.0 and 1.0
                        max_padding_size = h_padding
                        graded_size = base_size + t * (max_padding_size - base_size)

                        return float(graded_size)

            gmsh.model.mesh.setSizeCallback(mesh_size_callback3D)

        if parallel:
            parallel_print("Computing mesh size callback for parallel meshing...", comm=comm)

            x_core = np.linspace(bbox[2], bbox[3], nx)
            y_core = np.linspace(bbox[4], bbox[5], ny)
            z_core = np.linspace(bbox[0], bbox[1], nz)

            XX_c, YY_c, ZZ_c = np.meshgrid(x_core, y_core, z_core, indexing="ij")
            coords_c = np.column_stack((ZZ_c.ravel(), XX_c.ravel(), YY_c.ravel()))

            size_core = ef_segy2(coords_c).ravel()

            sp_data_core = np.empty((XX_c.size, 4), dtype=np.float64)
            sp_data_core[:, 0] = XX_c.ravel()
            sp_data_core[:, 1] = YY_c.ravel()
            sp_data_core[:, 2] = ZZ_c.ravel()
            sp_data_core[:, 3] = size_core

            del XX_c, YY_c, ZZ_c, coords_c, size_core

            xmin_pad = -padding_x
            xmax_pad = length_x + padding_x
            ymin_pad = -padding_y
            ymax_pad = length_y + padding_y
            zmin_pad = -abs(depth_z) - padding_z
            zmax_pad = 0.0

            x_pad = np.linspace(xmin_pad, xmax_pad, 50)
            y_pad = np.linspace(ymin_pad, ymax_pad, 50)
            z_pad = np.linspace(zmin_pad, zmax_pad, 50)

            XX_p, YY_p, ZZ_p = np.meshgrid(x_pad, y_pad, z_pad, indexing="ij")

            eps = 1e-8
            out_mask = ~(
                (XX_p >= bbox[2] - eps)
                & (XX_p <= bbox[3] + eps)
                & (YY_p >= bbox[4] - eps)
                & (YY_p <= bbox[5] + eps)
                & (ZZ_p >= bbox[0] - eps)
                & (ZZ_p <= bbox[1] + eps)
            )

            XX_out = XX_p[out_mask]
            YY_out = YY_p[out_mask]
            ZZ_out = ZZ_p[out_mask]

            # Padding sizing
            if extend_segy:

                coords_p = np.column_stack((ZZ_out, XX_out, YY_out))
                size_pad = ef_segy2(coords_p).ravel()
            else:

                XX_proj = np.clip(XX_out, bbox[2], bbox[3])
                YY_proj = np.clip(YY_out, bbox[4], bbox[5])
                ZZ_proj = np.clip(ZZ_out, bbox[0], bbox[1])

                coords_p = np.column_stack((ZZ_proj, XX_proj, YY_proj))
                base_size_pad = ef_segy2(coords_p).ravel()

                tx = (
                    np.abs(XX_out - XX_proj) / padding_x
                    if padding_x > 0
                    else np.zeros_like(XX_out)
                )
                ty = (
                    np.abs(YY_out - YY_proj) / padding_y
                    if padding_y > 0
                    else np.zeros_like(YY_out)
                )
                tz = (
                    np.abs(ZZ_out - ZZ_proj) / padding_z
                    if padding_z > 0
                    else np.zeros_like(ZZ_out)
                )

                if padding_type == "hyperelliptical":
                    t = (tx**hyper_n + ty**hyper_n + tz**hyper_n) ** (1.0 / hyper_n)
                else:
                    t = np.maximum(np.maximum(tx, ty), tz)

                t = np.clip(t, 0.0, 1.0)
                size_pad = base_size_pad + t * (h_padding - base_size_pad)

            sp_data_pad = np.empty((XX_out.size, 4), dtype=np.float64)
            sp_data_pad[:, 0] = XX_out
            sp_data_pad[:, 1] = YY_out
            sp_data_pad[:, 2] = ZZ_out
            sp_data_pad[:, 3] = size_pad

            del XX_p, YY_p, ZZ_p, XX_out, YY_out, ZZ_out
            if not extend_segy:
                del XX_proj, YY_proj, ZZ_proj, base_size_pad, tx, ty, tz, t
            del coords_p, size_pad

            sp_data_combined = np.vstack((sp_data_core, sp_data_pad))
            total_points = len(sp_data_combined)

            sp_list = sp_data_combined.ravel().tolist()

            del sp_data_core, sp_data_pad, sp_data_combined

            parallel_print(f"Loading {total_points} total points into single Gmsh PostView...", comm=comm)
            view_tag = gmsh.view.add("background_size")

            gmsh.view.addListData(view_tag, "SP", total_points, sp_list)

            # Apply background field
            gmsh.model.mesh.field.add("PostView", 1)
            gmsh.model.mesh.field.setNumber(1, "ViewIndex", view_tag)
            gmsh.model.mesh.field.setAsBackgroundMesh(1)
            gmsh.option.setNumber("General.NumThreads", 24)


def apply_structured_winslow_smoothing3D(
    gmsh,
    comm,
    geom_params,
    length_x,
    length_y,
    depth_z,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    water_interface,
    hyper_n,
    winslow_implementation,
    apply_winslow,
    winslow_iterations,
    winslow_omega,
    n_samples,
    n_traces_x,
    n_traces_y,
    domain_xmin,
    domain_xmax,
    domain_ymin,
    domain_ymax,
    domain_zmin,
    domain_zmax,
    ef_segy3,
    parallel_print,
):
    """Apply the configured Winslow smoother to a structured three-dimensional Gmsh mesh.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    geom_params : dict
        Geometry metadata returned by the Gmsh geometry builder.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_type : str or None
        Padding geometry, either ``None`` or ``"rectangular"`` for Winslow.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    water_interface : bool
        Whether the water/subsurface interface is geometrically delimited.
    hyper_n : float
        Hyperellipsoid exponent used by the padding geometry and sizing law.
    winslow_implementation : str or None
        Requested Winslow implementation.
    apply_winslow : bool
        Whether Winslow smoothing should be applied.
    winslow_iterations : int
        Number of Winslow smoothing iterations.
    winslow_omega : float
        Winslow relaxation coefficient.
    n_samples : int
        Velocity-model sample count retained for API compatibility.
    n_traces_x : int
        Velocity-model x-trace count retained for API compatibility.
    n_traces_y : int
        Velocity-model y-trace count retained for API compatibility.
    domain_xmin : float
        Minimum model x coordinate retained for API compatibility.
    domain_xmax : float
        Maximum model x coordinate retained for API compatibility.
    domain_ymin : float
        Minimum model y coordinate retained for API compatibility.
    domain_ymax : float
        Maximum model y coordinate retained for API compatibility.
    domain_zmin : float
        Minimum model z coordinate retained for API compatibility.
    domain_zmax : float
        Maximum model z coordinate retained for API compatibility.
    ef_segy3 : callable
        Three-dimensional mesh-sizing function in ``(z, x, y)`` coordinates.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    None
        The Gmsh node coordinates are updated in place when smoothing is enabled.
    """
    if not apply_winslow:
        parallel_print("Skipping 3D Winslow smoothing.", comm=comm)
        return

    if padding_type == "hyperelliptical":
        raise ValueError(
            "3-D Winslow smoothing is not available for hyperelliptical padding."
        )
    if padding_type not in (None, "rectangular"):
        raise ValueError(
            "3-D Winslow smoothing supports only padding_type=None "
            "or padding_type='rectangular'."
        )

    selected_winslow = (
        "numba"
        if winslow_implementation is None
        else str(winslow_implementation).strip().lower()
    )
    if selected_winslow != "numba":
        raise ValueError(
            "Only winslow_implementation='numba' is currently available "
            "for 3-D structured smoothing. Received "
            f"{winslow_implementation!r}."
        )

    geometry_name = (
        ("Water Interface" if water_interface else "No Water Interface")
        + ", "
        + ("Rectangular Padding" if padding_type == "rectangular" else "No Padding")
    )
    parallel_print(f"Extracting nodes for 3D smoothing ({geometry_name})...", comm=comm)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)

    elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    all_hex_nodes = []
    for element_index, element_type in enumerate(elem_types):
        if element_type == 5:  # Gmsh type 5 = 8-node hexahedron
            all_hex_nodes.extend(elem_node_tags[element_index])

    if not all_hex_nodes:
        raise ValueError(
            "No hexahedra found. 3-D Winslow smoothing requires "
            "an 8-node hexahedral mesh."
        )

    tag_to_index = {
        int(tag): index
        for index, tag in enumerate(node_tags)
    }
    hexes = np.asarray(
        [tag_to_index[int(tag)] for tag in all_hex_nodes],
        dtype=np.int64,
    ).reshape(-1, 8)

    winslow_points = define_winslow_points_3d(
        gmsh=gmsh,
        points_3d=points_3d,
        tag_to_index=tag_to_index,
        length_x=length_x,
        length_y=length_y,
        depth_z=depth_z,
        padding_type=padding_type,
        padding_x=padding_x,
        padding_y=padding_y,
        padding_z=padding_z,
        water_interface=water_interface,
        tol=2.0,
    )

    move_all = winslow_points["move_all"]
    move_X_only = winslow_points["move_X_only"]
    move_Y_only = winslow_points["move_Y_only"]
    move_Z_only = winslow_points["move_Z_only"]
    movable_nodes = winslow_points["movable_nodes"]

    parallel_print("Applying 3D Winslow smoothing...", comm=comm)
    smoothed_points_3d = run_selected_winslow(
        points=points_3d,
        hexes=hexes,
        move_all=move_all,
        move_X_only=move_X_only,
        move_Y_only=move_Y_only,
        move_Z_only=move_Z_only,
        ef_segy=ef_segy3,
        length_x=length_x,
        length_y=length_y,
        depth_z=depth_z,
        winslow_iterations=winslow_iterations,
        winslow_omega=winslow_omega,
        selected_winslow=selected_winslow,
        comm=comm,
        parallel_print=parallel_print,
    )

    parallel_print("Updating Winslow-smoothed nodes back into Gmsh...", comm=comm)
    for node_index, node_tag in enumerate(node_tags):
        if node_index in movable_nodes:
            gmsh.model.mesh.setNode(
                int(node_tag),
                smoothed_points_3d[node_index].tolist(),
                [],
            )
