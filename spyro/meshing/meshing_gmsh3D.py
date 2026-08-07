from pathlib import Path

import numpy as np

from .meshing_winslow3D import winslow_smooth_3d55


def report_quality(gmsh, dim=3, quality_type=2):
    """
    dim: 2 (surface) or 3 (volume)
    quality_type controls the metric Gmsh uses, e.g.:
      0=gamma (vol/sum_face/max_edge), 1=eta (vol^(2/3)/sum_edge^2), 2=rho (min_edge/max_edge)
    """
    gmsh.option.setNumber("Mesh.QualityType", quality_type)  # choose metric

    # Get elements
    _elem_types, elem_tags, _elem_node_tags = gmsh.model.mesh.getElements(dim)

    # Collect element tags
    all_tags = []
    for tags in elem_tags:
        all_tags.extend(tags.tolist() if hasattr(tags, "tolist") else list(tags))

    if not all_tags:
        print(f"[quality] No elements found for dim={dim}")
        return

    # Element quality
    q = gmsh.model.mesh.getElementQualities(all_tags)

    q = np.asarray(q, dtype=float)
    print(
        f"[quality] count={q.size}  min={q.min():.6g}  p1={np.percentile(q, 1):.6g}  "
        f"p5={np.percentile(q, 5):.6g}  median={np.median(q):.6g}  "
        f"p95={np.percentile(q, 95):.6g}  max={q.max():.6g}  mean={q.mean():.6g}"
    )


def create_super_ellipsoid_volume(
    gmsh, a=1.0, b=1.0, c=1.0, n=2.0, xc=0.0, yc=0.0, zc=0.0
):
    """
    Create a 3D super ellipsoid volume using OpenCASCADE B-spline surfaces.

    Parameters:
    -----------
    a, b, c : float
        Semi-axes lengths in x, y, z directions
    n : float
        Exponent parameter (n=2 gives regular ellipsoid, n>2 gives more cubic shape, n<2 gives more pointed shape)
    xc, yc, zc : float
        Center coordinates of the ellipsoid

    Returns:
    --------
    volume_tag : int
        OpenCASCADE volume tag
    """

    def super_ellipsoid_point(u, v, a, b, c, n, xc, yc, zc):
        """
        Calculate a point on the super ellipsoid surface.
        u: longitude parameter [0, 2π]
        v: latitude parameter [-π/2, π/2]
        xc, yc, zc: center coordinates
        """

        # Signed power
        def sign_power(x, p):
            if abs(x) < 1e-10:
                return 0.0
            return np.sign(x) * (np.abs(x) ** p)

        cos_v = np.cos(v)
        sin_v = np.sin(v)
        cos_u = np.cos(u)
        sin_u = np.sin(u)

        # Local coordinates
        x = a * sign_power(cos_v, 2.0 / n) * sign_power(cos_u, 2.0 / n)
        y = b * sign_power(cos_v, 2.0 / n) * sign_power(sin_u, 2.0 / n)
        z = c * sign_power(sin_v, 2.0 / n)

        # Translate to center
        x += xc
        y += yc
        z += zc

        return x, y, z

    volume_tag = None

    try:
        volume_tag = create_properly_closed_surface(
            gmsh, a, b, c, n, xc, yc, zc, super_ellipsoid_point
        )
        if volume_tag:
            print("Successfully created volume using closed B-spline surface")
    except Exception as exc:
        raise RuntimeError(
            "B-spline surface creation failed."
        ) from exc

    return volume_tag


def create_properly_closed_surface(gmsh, a, b, c, n, xc, yc, zc, point_func):
    """
    Create a closed B-spline surface by ensuring u-direction closure and including poles.
    """
    u_res = 60  # Points in longitude
    v_res = 60  # Points in latitude (including poles)

    # Point grid
    point_tags = []

    # Parametric grid
    for j in range(v_res):
        for i in range(u_res + 1):  # +1 to include closure point at u=2π
            # Close longitude
            if i == u_res:
                u = 0.0  # Close the loop
            else:
                u = 2 * np.pi * i / u_res

            # Latitude
            v = np.pi * (j / (v_res - 1) - 0.5)  # From -π/2 to π/2

            # Poles
            if j == 0:  # South pole
                x, y, z = point_func(0, -np.pi / 2, a, b, c, n, xc, yc, zc)
            elif j == v_res - 1:  # North pole
                x, y, z = point_func(0, np.pi / 2, a, b, c, n, xc, yc, zc)
            else:
                x, y, z = point_func(u, v, a, b, c, n, xc, yc, zc)

            point_tag = gmsh.model.occ.addPoint(x, y, z)
            point_tags.append(point_tag)

    try:
        # B-spline surface
        surface_tag = gmsh.model.occ.addBSplineSurface(
            pointTags=point_tags,
            numPointsU=u_res + 1,  # Include closure point
            tag=-1,
            degreeU=min(3, u_res),
            degreeV=min(3, v_res - 1),
        )

        gmsh.model.occ.synchronize()

        # Create volume
        surface_loop = gmsh.model.occ.addSurfaceLoop([surface_tag])
        volume_tag = gmsh.model.occ.addVolume([surface_loop])
        gmsh.model.occ.synchronize()

        # Z cut
        z_cut = 0.0
        if volume_tag is not None and z_cut is not None:
            print(f"Applying z-cut at z = {z_cut}")

            # Cutting box
            domain_size = 2 * max(a, b, c)
            cutting_box = gmsh.model.occ.addBox(
                xc - domain_size,
                yc - domain_size,
                z_cut,
                2 * domain_size,
                2 * domain_size,
                domain_size,
            )

            gmsh.model.occ.synchronize()

            # Remove upper part
            result = gmsh.model.occ.cut(
                [(3, volume_tag)],
                [(3, cutting_box)],
                removeObject=True,
                removeTool=True,
            )

            if result[0]:
                volume_tag = result[0][0][1]
                print("Z-cut applied successfully")
            else:
                print("Warning: Z-cut removed entire volume")
                volume_tag = None

        return volume_tag

    except Exception as exc:
        raise RuntimeError(
            "B-spline surface creation failed."
        ) from exc


report_quality3D = report_quality
create_super_ellipsoid_volume3D = create_super_ellipsoid_volume
create_properly_closed_surface3D = create_properly_closed_surface


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
):
    """Build the complete 3-D Gmsh geometry and physical groups.

    Six geometry combinations are supported:

    1. no padding, no water delimitation;
    2. rectangular padding, no water delimitation;
    3. hyperelliptical padding, no water delimitation;
    4. no padding, water delimitation;
    5. rectangular padding, water delimitation;
    6. hyperelliptical padding, water delimitation.

    Returns
    -------
    dict
        Geometry and volume tags needed by mesh generation and smoothing.
    """
    del minElementSize

    source_padding_type = (
        "elliptical" if padding_type == "hyperelliptical" else padding_type
    )
    padding_type = source_padding_type

    if padding_type not in (None, "rectangular", "elliptical"):
        raise ValueError(
            "padding_type must be None, 'rectangular', "
            "'elliptical', or 'hyperelliptical'."
        )

    # Hyperelliptical padding
    if padding_type == "elliptical" and structured_mesh:
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

    domainX = float(length_x)
    domainY = float(length_y)
    domainZ = abs(float(depth_z))
    padX = float(padding_x)
    padY = float(padding_y)
    padZ = float(padding_z)
    ellipse_n = float(hyper_n)

    ORDER = axes_order_sort
    DTYPE_BE = np.dtype(dtype).newbyteorder(">" if byte_order == "big" else "<").str

    ellipseLx = padX
    ellipseLy = padY
    ellipseLz = padZ
    box_xmin = 0.0
    box_xmax = domainX
    box_ymin = 0.0
    box_ymax = domainY
    box_zmin = -domainZ
    box_zmax = 0.0
    ellipse_a = domainX / 2.0 + ellipseLx
    ellipse_b = domainY / 2.0 + ellipseLy
    ellipse_c = domainZ / 2.0 + ellipseLz
    xc = domainX / 2.0
    yc = domainY / 2.0
    zc = -domainZ / 2.0

    z_min, z_max = float(depth_z), 0.0
    x_min, x_max = 0.0, float(length_x)
    y_min, y_max = 0.0, float(length_y)

    if not water_interface and padding_type is None:
        # No padding
        print(
            "Generating undelimited water+subsurface rectangular domain (no padding)..."
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
        # Rectangular padding
        print(
            "Generating undelimited water+subsurface domain with rectangular padding..."
        )
        occ = gmsh.model.occ

        if min(domainX, domainY, domainZ) <= 0.0:
            raise ValueError("length_x, length_y, and depth_z must be positive.")
        if min(padX, padY, padZ) < 0.0:
            raise ValueError("padding_x, padding_y, and padding_z cannot be negative.")

        def add_padding_box(x0, y0, z0, lx, ly, lz):
            """Create a non-degenerate rectangular padding block."""
            if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
                return None
            return occ.addBox(
                float(x0),
                float(y0),
                float(z0),
                float(lx),
                float(ly),
                float(lz),
            )

        # water triangle and the subsurface.
        vol_subsoil = occ.addBox(
            0.0,
            0.0,
            -domainZ,
            domainX,
            domainY,
            domainZ,
        )
        vol_water = None

        vol_soil_plusX = add_padding_box(
            domainX,
            0.0,
            -domainZ,
            padX,
            domainY,
            domainZ,
        )
        vol_soil_minusX = add_padding_box(
            -padX,
            0.0,
            -domainZ,
            padX,
            domainY,
            domainZ,
        )
        vol_soil_plusY = add_padding_box(
            0.0,
            domainY,
            -domainZ,
            domainX,
            padY,
            domainZ,
        )
        vol_soil_minusY = add_padding_box(
            0.0,
            -padY,
            -domainZ,
            domainX,
            padY,
            domainZ,
        )

        vol_soil_diag_NE = add_padding_box(
            domainX,
            domainY,
            -domainZ,
            padX,
            padY,
            domainZ,
        )
        vol_soil_diag_NW = add_padding_box(
            -padX,
            domainY,
            -domainZ,
            padX,
            padY,
            domainZ,
        )
        vol_soil_diag_SW = add_padding_box(
            -padX,
            -padY,
            -domainZ,
            padX,
            padY,
            domainZ,
        )
        vol_soil_diag_SE = add_padding_box(
            domainX,
            -padY,
            -domainZ,
            padX,
            padY,
            domainZ,
        )

        bottom_z = -domainZ - padZ

        vol_soil_bottom_ext = add_padding_box(
            0.0,
            0.0,
            bottom_z,
            domainX,
            domainY,
            padZ,
        )
        vol_soil_plusX_bottom_ext = add_padding_box(
            domainX,
            0.0,
            bottom_z,
            padX,
            domainY,
            padZ,
        )
        vol_soil_minusX_bottom_ext = add_padding_box(
            -padX,
            0.0,
            bottom_z,
            padX,
            domainY,
            padZ,
        )
        vol_soil_plusY_bottom_ext = add_padding_box(
            0.0,
            domainY,
            bottom_z,
            domainX,
            padY,
            padZ,
        )
        vol_soil_minusY_bottom_ext = add_padding_box(
            0.0,
            -padY,
            bottom_z,
            domainX,
            padY,
            padZ,
        )
        vol_soil_NE_bottom_diag_ext = add_padding_box(
            domainX,
            domainY,
            bottom_z,
            padX,
            padY,
            padZ,
        )
        vol_soil_NW_bottom_diag_ext = add_padding_box(
            -padX,
            domainY,
            bottom_z,
            padX,
            padY,
            padZ,
        )
        vol_soil_SW_bottom_diag_ext = add_padding_box(
            -padX,
            -padY,
            bottom_z,
            padX,
            padY,
            padZ,
        )
        vol_soil_SE_bottom_diag_ext = add_padding_box(
            domainX,
            -padY,
            bottom_z,
            padX,
            padY,
            padZ,
        )

        side_padding_volumes = [
            vol_soil_plusX,
            vol_soil_minusX,
            vol_soil_plusY,
            vol_soil_minusY,
            vol_soil_diag_NE,
            vol_soil_diag_NW,
            vol_soil_diag_SW,
            vol_soil_diag_SE,
        ]
        bottom_padding_volumes = [
            vol_soil_bottom_ext,
            vol_soil_plusX_bottom_ext,
            vol_soil_minusX_bottom_ext,
            vol_soil_plusY_bottom_ext,
            vol_soil_minusY_bottom_ext,
            vol_soil_NE_bottom_diag_ext,
            vol_soil_NW_bottom_diag_ext,
            vol_soil_SW_bottom_diag_ext,
            vol_soil_SE_bottom_diag_ext,
        ]
        pad_vols = [
            tag
            for tag in side_padding_volumes + bottom_padding_volumes
            if tag is not None
        ]

        if pad_vols:
            input_volume_tags = [vol_subsoil] + pad_vols
            _, rectangular_fragment_map = occ.fragment(
                [(3, tag) for tag in input_volume_tags],
                [],
                removeObject=True,
                removeTool=False,
            )
            occ.synchronize()

            core_fragments = sorted(
                {tag for dim, tag in rectangular_fragment_map[0] if dim == 3}
            )
            padding_fragments = sorted(
                {
                    tag
                    for mapping in rectangular_fragment_map[1:]
                    for dim, tag in mapping
                    if dim == 3
                }
                - set(core_fragments)
            )

            if not core_fragments:
                raise RuntimeError(
                    "The central water+subsurface rectangle was lost during "
                    "the rectangular padding fragment operation."
                )

            vol_subsoil = core_fragments[0]
            pad_vols = padding_fragments
        else:
            occ.synchronize()
            core_fragments = [vol_subsoil]

        # subsurface through the velocity model.
        gmsh.model.addPhysicalGroup(3, core_fragments, name="SubsurfaceAndWater")
        if pad_vols:
            gmsh.model.addPhysicalGroup(3, pad_vols, name="Padding")

    elif not water_interface and padding_type == "elliptical":
        # Hyperelliptical padding
        print(
            "Generating undelimited water+subsurface domain with "
            "hyperelliptical padding..."
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
        ellipsoid_tag = create_super_ellipsoid_volume(
            gmsh=gmsh,
            a=ellipse_a,
            b=ellipse_b,
            c=ellipse_c,
            n=ellipse_n,
            xc=xc,
            yc=yc,
            zc=zc,
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

        def boundary_faces_of_volume(vol_tag):
            boundary = gmsh.model.getBoundary(
                [(3, vol_tag)],
                oriented=False,
                recursive=False,
                combined=True,
            )
            return [tag for dim, tag in boundary if dim == 2]

        ellipsoid_faces = set()
        for volume_tag in padding_fragments:
            ellipsoid_faces.update(boundary_faces_of_volume(volume_tag))

        for surface_tag in ellipsoid_faces:
            gmsh.model.mesh.setAlgorithm(2, int(surface_tag), 1)

    else:
        # Cases 4-6: water delimitation enabled.

        # diagonal, and bottom volumes;

        print("Generating geometrically delimited water and subsurface volumes...")

        bin_path = fname
        value = water_search_value
        tolerance = 1.0
        x_chunk = 64
        y_chunk = None

        try:
            nz_tot, nx_tot, ny_tot = int(nz), int(nx), int(ny)
        except Exception as e:
            raise RuntimeError("nz, nx, and ny must be valid integer arguments.") from e

        try:
            _dz, _dx, _dy = float(dz), float(dx), float(dy)
        except Exception as e:
            raise RuntimeError("dz, dx, and dy must be valid spacing arguments.") from e

        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary not found: {bin_path}")

        expected = nz_tot * nx_tot * ny_tot
        mm = np.memmap(bin_path, dtype=np.dtype(DTYPE_BE), mode="r")
        if mm.size != expected:
            raise ValueError(
                f"File size mismatch for {bin_path}:\n"
                f"  got {mm.size} floats, expected {expected} for shape (nz,nx,ny)=({nz_tot},{nx_tot},{ny_tot})"
            )

        cube_zxy = mm.reshape((nz_tot, nx_tot, ny_tot), order=ORDER)

        z_top = float(max(z_min, z_max))
        z_bot = float(min(z_min, z_max))
        x_low, x_high = sorted([float(x_min), float(x_max)])
        y_low, y_high = sorted([float(y_min), float(y_max)])

        Lx = (nx_tot - 1) * _dx
        Ly = (ny_tot - 1) * _dy
        Lz = (nz_tot - 1) * _dz

        x_low = np.clip(x_low, 0.0, Lx)
        x_high = np.clip(x_high, 0.0, Lx)
        y_low = np.clip(y_low, 0.0, Ly)
        y_high = np.clip(y_high, 0.0, Ly)
        z_top = np.clip(z_top, -Lz, 0.0)
        z_bot = np.clip(z_bot, -Lz, 0.0)

        ix_min = int(np.rint(x_low / _dx))
        ix_max = int(np.rint(x_high / _dx))
        iy_min = int(np.rint(y_low / _dy))
        iy_max = int(np.rint(y_high / _dy))

        ix_min, ix_max = (
            max(0, min(ix_min, nx_tot - 1)),
            max(0, min(ix_max, nx_tot - 1)),
        )
        iy_min, iy_max = (
            max(0, min(iy_min, ny_tot - 1)),
            max(0, min(iy_max, ny_tot - 1)),
        )
        if ix_max < ix_min:
            ix_min, ix_max = ix_max, ix_min
        if iy_max < iy_min:
            iy_min, iy_max = iy_max, iy_min

        iz_top = int(np.rint(-z_top / _dz))
        iz_bot = int(np.rint(-z_bot / _dz))
        iz_top = max(0, min(iz_top, nz_tot - 1))
        iz_bot = max(0, min(iz_bot, nz_tot - 1))
        if iz_bot < iz_top:
            iz_top, iz_bot = iz_bot, iz_top

        Xs = _dx * np.arange(ix_min, ix_max + 1, dtype=np.float64)
        Ys = _dy * np.arange(iy_min, iy_max + 1, dtype=np.float64)
        Nx = ix_max - ix_min
        Ny = iy_max - iy_min

        x_chunk = int(max(1, x_chunk))
        if y_chunk is None:
            y_chunk = Ny + 1
        y_chunk = int(max(1, y_chunk))

        Z_bottom = np.empty((Nx + 1, Ny + 1), dtype=np.float64)

        target = float(value)
        tol = float(tolerance)

        for xs in range(0, Nx + 1, x_chunk):
            xe = min(xs + x_chunk, Nx + 1)
            ix_tile = ix_min + np.arange(xs, xe, dtype=int)

            for ys in range(0, Ny + 1, y_chunk):
                ye = min(ys + y_chunk, Ny + 1)
                iy_tile = iy_min + np.arange(ys, ye, dtype=int)

                block = cube_zxy[iz_top: iz_bot + 1, :, :][:, ix_tile, :][
                    :, :, iy_tile
                ]
                block = np.asarray(block, dtype=np.float32)

                in_water = np.abs(block - target) <= tol
                non_water = ~in_water

                any_non_water = np.any(non_water, axis=0)
                first_non_idx = np.argmax(non_water, axis=0).astype(np.int32)

                first_non_idx = np.where(
                    any_non_water, first_non_idx, (iz_bot - iz_top)
                ).astype(np.int32)

                k_global = iz_top + first_non_idx
                z_phys = -k_global.astype(np.float64) * _dz

                Z_bottom[xs:xe, ys:ye] = z_phys

        occ = gmsh.model.occ

        gmsh.option.setNumber("Geometry.Tolerance", 1e-16)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

        pB = [[None] * (Ny + 1) for _ in range(Nx + 1)]
        for i, x in enumerate(Xs):
            for j, y in enumerate(Ys):
                pB[i][j] = occ.addPoint(float(x), float(y), float(Z_bottom[i, j]))

        south_pts = [pB[i][0] for i in range(Nx + 1)]
        [pB[Nx][j] for j in range(Ny + 1)]
        north_pts = [pB[i][Ny] for i in range(Nx, -1, -1)]
        west_pts = [pB[0][j] for j in range(Ny, -1, -1)]

        ctrl = [pB[i][j] for j in range(Ny + 1) for i in range(Nx + 1)]

        degreeU = min(3, Nx)
        degreeV = min(3, Ny)

        s_bottom = occ.addBSplineSurface(
            ctrl,
            Nx + 1,
            Ny + 1,
            degreeU,
            degreeV,
            [],
            [],
            [],
            [],
            [],
            [],
        )

        occ.synchronize()

        gmsh.model.occ.synchronize()
        x_low, x_high = float(Xs[0]), float(Xs[-1])
        y_low, y_high = float(Ys[0]), float(Ys[-1])
        eps = 1e-9 * max(abs(x_high - x_low), abs(y_high - y_low), 1.0)

        _, curves = gmsh.model.getAdjacencies(2, s_bottom)
        curves = list(dict.fromkeys(curves))

        lB_S = lB_E = lB_N = lB_W = None
        for e in curves:
            _, pts = gmsh.model.getAdjacencies(1, e)
            (x0, y0, _) = occ.getCenterOfMass(0, pts[0])
            (x1, y1, _) = occ.getCenterOfMass(0, pts[1])
            (cx, cy, _) = occ.getCenterOfMass(1, e)
            if abs(cy - y_low) < eps:
                lB_S = e if x1 >= x0 else -e
            elif abs(cy - y_high) < eps:
                lB_N = e if x1 <= x0 else -e
            elif abs(cx - x_high) < eps:
                lB_E = e if y1 >= y0 else -e
            else:
                lB_W = e if y1 <= y0 else -e

        occ.addCurveLoop([lB_S, lB_E, lB_N, lB_W])
        gmsh.model.occ.synchronize()

        occ.synchronize()

        pT_SW = occ.addPoint(x_low, y_low, z_top)
        pT_SE = occ.addPoint(x_high, y_low, z_top)
        pT_NE = occ.addPoint(x_high, y_high, z_top)
        pT_NW = occ.addPoint(x_low, y_high, z_top)

        lT_S = occ.addLine(pT_SW, pT_SE)
        lT_E = occ.addLine(pT_SE, pT_NE)
        lT_N = occ.addLine(pT_NE, pT_NW)
        lT_W = occ.addLine(pT_NW, pT_SW)

        wire_top = occ.addWire([lT_S, lT_E, lT_N, lT_W])
        s_top = occ.addBSplineFilling(wire_top)

        v_SW = occ.addLine(pT_SW, south_pts[0])
        v_SE = occ.addLine(pT_SE, south_pts[-1])
        v_NE = occ.addLine(pT_NE, north_pts[0])
        v_NW = occ.addLine(pT_NW, west_pts[0])

        wire_S = occ.addWire([lT_S, v_SE, -lB_S, -v_SW])
        s_side_S = occ.addBSplineFilling(wire_S)

        wire_E = occ.addWire([lT_E, v_NE, -lB_E, -v_SE])
        s_side_E = occ.addBSplineFilling(wire_E)

        wire_N = occ.addWire([lT_N, v_NW, -lB_N, -v_NE])
        s_side_N = occ.addBSplineFilling(wire_N)

        wire_W = occ.addWire([lT_W, v_SW, -lB_W, -v_NW])
        s_side_W = occ.addBSplineFilling(wire_W)

        pBot_SW = occ.addPoint(x_low, y_low, z_min)
        pBot_SE = occ.addPoint(x_high, y_low, z_min)
        pBot_NE = occ.addPoint(x_high, y_high, z_min)
        pBot_NW = occ.addPoint(x_low, y_high, z_min)

        lBot_S = occ.addLine(pBot_SW, pBot_SE)
        lBot_E = occ.addLine(pBot_SE, pBot_NE)
        lBot_N = occ.addLine(pBot_NE, pBot_NW)
        lBot_W = occ.addLine(pBot_NW, pBot_SW)

        wire_botflat = occ.addWire([lBot_S, lBot_E, lBot_N, lBot_W])
        s_bot_flat = occ.addBSplineFilling(wire_botflat)

        v_SW2 = occ.addLine(south_pts[0], pBot_SW)
        v_SE2 = occ.addLine(south_pts[-1], pBot_SE)
        v_NE2 = occ.addLine(north_pts[0], pBot_NE)
        v_NW2 = occ.addLine(west_pts[0], pBot_NW)

        wire_S2 = occ.addWire([lB_S, v_SE2, -lBot_S, -v_SW2])
        s_side_S2 = occ.addBSplineFilling(wire_S2)

        wire_E2 = occ.addWire([lB_E, v_NE2, -lBot_E, -v_SE2])
        s_side_E2 = occ.addBSplineFilling(wire_E2)

        wire_N2 = occ.addWire([lB_N, v_NW2, -lBot_N, -v_NE2])
        s_side_N2 = occ.addBSplineFilling(wire_N2)

        wire_W2 = occ.addWire([lB_W, v_SW2, -lBot_W, -v_NW2])
        s_side_W2 = occ.addBSplineFilling(wire_W2)

        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        surfaces_subsoil = [
            s_bottom,
            s_bot_flat,
            s_side_S2,
            s_side_E2,
            s_side_N2,
            s_side_W2,
        ]
        sl_subsoil = occ.addSurfaceLoop(surfaces_subsoil)
        vol_subsoil = occ.addVolume([sl_subsoil])

        surfaces_water = [s_top, s_bottom, s_side_S, s_side_E, s_side_N, s_side_W]
        sl_water = occ.addSurfaceLoop(surfaces_water, sewing=True)
        vol_water = occ.addVolume([sl_water])

        occ.synchronize()
        if padding_type is None:

            gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="Subsurface")
            gmsh.model.addPhysicalGroup(3, [vol_water], name="Water")
        if padding_type == "elliptical":
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
            ellipsoid_tag = create_super_ellipsoid_volume(
                gmsh,
                a=ellipse_a,
                b=ellipse_b,
                c=ellipse_c,
                n=ellipse_n,
                xc=xc,
                yc=yc,
                zc=zc,
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

            def boundary_faces_of_volume(vol_tag):
                # Unique 2D faces bounding this volume (tags only)
                bnd = gmsh.model.getBoundary(
                    [(3, vol_tag)], oriented=False, recursive=False, combined=True
                )  # unique
                return [t for (d, t) in bnd if d == 2]

            ellipsoid_faces = []
            for v in ellipsoid_volumes:
                ellipsoid_faces.extend(boundary_faces_of_volume(v))
            ellipsoid_faces = set(ellipsoid_faces)
            algo_id = 1.0
            for surf_tag in ellipsoid_faces:
                gmsh.model.mesh.setAlgorithm(2, int(surf_tag), int(algo_id))

        if padding_type == "rectangular":
            # +X WATER EXTENSION
            occ = gmsh.model.occ
            occ.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, vol_water)
            eps = 1e-9 * max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin), 1.0)
            _, surf_tags = gmsh.model.getAdjacencies(3, vol_water)
            east_surfs = []
            for s in surf_tags:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(sxmin - xmax) < eps and abs(sxmax - xmax) < eps:
                    east_surfs.append(s)
            if not east_surfs:
                raise RuntimeError(
                    "No +X wall surface found on vol_water (no surface with x-min==x-max==xmax)."
                )
            east_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[4]
                    - gmsh.model.getBoundingBox(2, s)[1]
                ),
                reverse=True,
            )
            sE = east_surfs[0]
            _, sE_curves = gmsh.model.getAdjacencies(2, sE)
            seen = set()
            curves = []
            for c in sE_curves:
                if c not in seen:
                    seen.add(c)
                    curves.append(c)
            c_top = c_bottom = c_south = c_north = None
            for c in curves:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmax) < eps and abs(czmax - zmax) < eps:
                    c_top = c
                elif abs(cymin - ymin) < eps and abs(cymax - ymin) < eps:
                    c_south = c
                elif abs(cymin - ymax) < eps and abs(cymax - ymax) < eps:
                    c_north = c
                else:
                    c_bottom = c
            if any(v is None for v in (c_top, c_bottom, c_south, c_north)):
                raise RuntimeError(
                    f"Failed to classify east wall edges: top={c_top}, bottom={c_bottom}, south={c_south}, north={c_north}"
                )
            copies = occ.copy([(1, c_top), (1, c_bottom), (1, c_south), (1, c_north)])
            occ.synchronize()
            occ.translate(copies, padX, 0.0, 0.0)
            occ.synchronize()
            c_top_new = copies[0][1]
            c_bottom_new = copies[1][1]
            c_south_new = copies[2][1]
            c_north_new = copies[3][1]
            _, top_pts_old = gmsh.model.getAdjacencies(1, c_top)
            _, top_pts_new = gmsh.model.getAdjacencies(1, c_top_new)
            top_pts_old = list(top_pts_old)
            top_pts_new = list(top_pts_new)
            if len(top_pts_old) < 2 or len(top_pts_new) < 2:
                raise RuntimeError("Top edge does not have at least 2 endpoints.")
            top_pts_old = sorted(
                top_pts_old, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            top_pts_new = sorted(
                top_pts_new, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_top_SE_old, pt_top_NE_old = top_pts_old[0], top_pts_old[-1]
            pt_top_SE_new, pt_top_NE_new = top_pts_new[0], top_pts_new[-1]
            _, bot_pts_old = gmsh.model.getAdjacencies(1, c_bottom)
            _, bot_pts_new = gmsh.model.getAdjacencies(1, c_bottom_new)
            bot_pts_old = list(bot_pts_old)
            bot_pts_new = list(bot_pts_new)
            if len(bot_pts_old) < 2 or len(bot_pts_new) < 2:
                raise RuntimeError(
                    "Bottom rim edge does not have at least 2 endpoints."
                )
            bot_pts_old = sorted(
                bot_pts_old, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            bot_pts_new = sorted(
                bot_pts_new, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_bot_S_old, pt_bot_N_old = bot_pts_old[0], bot_pts_old[-1]
            pt_bot_S_new, pt_bot_N_new = bot_pts_new[0], bot_pts_new[-1]
            ln_top_south = occ.addLine(pt_top_SE_old, pt_top_SE_new)
            ln_top_north = occ.addLine(pt_top_NE_old, pt_top_NE_new)
            ln_bot_south = occ.addLine(pt_bot_S_old, pt_bot_S_new)
            ln_bot_north = occ.addLine(pt_bot_N_old, pt_bot_N_new)
            occ.synchronize()
            wire_E_new = occ.addWire(
                [c_top_new, c_north_new, -c_bottom_new, -c_south_new]
            )
            sE_new = occ.addBSplineFilling(wire_E_new)
            wire_roof = occ.addWire([c_top, ln_top_north, -c_top_new, -ln_top_south])
            s_roof = occ.addBSplineFilling(wire_roof)
            wire_bottom_ext = occ.addWire(
                [c_bottom, ln_bot_north, -c_bottom_new, -ln_bot_south]
            )
            s_bottom_ext = occ.addBSplineFilling(wire_bottom_ext)
            wire_south = occ.addWire(
                [c_south, ln_top_south, -c_south_new, -ln_bot_south]
            )
            s_south_ext = occ.addBSplineFilling(wire_south)
            wire_north = occ.addWire(
                [c_north, ln_top_north, -c_north_new, -ln_bot_north]
            )
            s_north_ext = occ.addBSplineFilling(wire_north)
            occ.synchronize()
            sl_ext = occ.addSurfaceLoop(
                [sE, sE_new, s_roof, s_bottom_ext, s_south_ext, s_north_ext]
            )
            vol_water_plusX = occ.addVolume([sl_ext])
            occ.synchronize()

            # +X SOIL EXTENSION
            occ = gmsh.model.occ
            occ.synchronize()
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            eps_s = 1e-9 * max(
                abs(xmax_s - xmin_s), abs(ymax_s - ymin_s), abs(zmax_s - zmin_s), 1.0
            )
            _, soil_surfs = gmsh.model.getAdjacencies(3, vol_subsoil)
            east_soil_surfs = []
            for s in soil_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(sxmin - xmax_s) < eps_s and abs(sxmax - xmax_s) < eps_s:
                    east_soil_surfs.append(s)
            if not east_soil_surfs:
                raise RuntimeError("No +X wall surface found on vol_subsoil.")
            east_soil_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[5]
                    - gmsh.model.getBoundingBox(2, s)[2]
                ),
                reverse=True,
            )
            sE_soil = east_soil_surfs[0]
            _, sE_soil_curves = gmsh.model.getAdjacencies(2, sE_soil)
            seen = set()
            soil_curves = []
            for c in sE_soil_curves:
                if c not in seen:
                    seen.add(c)
                    soil_curves.append(c)
            c_top_rim_soil = c_bottom_flat_old = c_south_low_old = c_north_low_old = (
                None
            )
            for c in soil_curves:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmin_s) < eps_s and abs(czmax - zmin_s) < eps_s:
                    c_bottom_flat_old = c
                elif abs(cymin - ymin_s) < eps_s and abs(cymax - ymin_s) < eps_s:
                    c_south_low_old = c
                elif abs(cymin - ymax_s) < eps_s and abs(cymax - ymax_s) < eps_s:
                    c_north_low_old = c
                else:
                    c_top_rim_soil = c
            if any(
                v is None
                for v in (
                    c_top_rim_soil,
                    c_bottom_flat_old,
                    c_south_low_old,
                    c_north_low_old,
                )
            ):
                raise RuntimeError("Failed to classify soil +X wall curves.")
            copies_floor = occ.copy([(1, c_bottom_flat_old)])
            occ.synchronize()
            occ.translate(copies_floor, padX, 0.0, 0.0)
            occ.synchronize()
            c_bottom_flat_new = copies_floor[0][1]
            _, botflat_pts_old = gmsh.model.getAdjacencies(1, c_bottom_flat_old)
            _, botflat_pts_new = gmsh.model.getAdjacencies(1, c_bottom_flat_new)
            botflat_pts_old = list(botflat_pts_old)
            botflat_pts_new = list(botflat_pts_new)
            if len(botflat_pts_old) < 2 or len(botflat_pts_new) < 2:
                raise RuntimeError("Soil bottom line endpoints are incomplete.")
            botflat_pts_old = sorted(
                botflat_pts_old, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            botflat_pts_new = sorted(
                botflat_pts_new, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_floor_S_old, pt_floor_N_old = botflat_pts_old[0], botflat_pts_old[-1]
            pt_floor_S_new, pt_floor_N_new = botflat_pts_new[0], botflat_pts_new[-1]
            ln_south_far = occ.addLine(pt_floor_S_new, pt_bot_S_new)
            ln_north_far = occ.addLine(pt_floor_N_new, pt_bot_N_new)
            ln_floor_south = occ.addLine(pt_floor_S_old, pt_floor_S_new)
            ln_floor_north = occ.addLine(pt_floor_N_old, pt_floor_N_new)
            occ.synchronize()
            wire_floor = occ.addWire(
                [c_bottom_flat_old, ln_floor_north, -c_bottom_flat_new, -ln_floor_south]
            )
            s_floor_ext = occ.addBSplineFilling(wire_floor)
            wire_west_low = occ.addWire(
                [c_top_rim_soil, c_north_low_old, -c_bottom_flat_old, -c_south_low_old]
            )
            s_west_low = occ.addBSplineFilling(wire_west_low)
            wire_east_low = occ.addWire(
                [c_bottom_new, ln_north_far, -c_bottom_flat_new, -ln_south_far]
            )
            s_east_low = occ.addBSplineFilling(wire_east_low)
            wire_south_low = occ.addWire(
                [c_south_low_old, ln_bot_south, -ln_south_far, -ln_floor_south]
            )
            s_south_low = occ.addBSplineFilling(wire_south_low)
            wire_north_low = occ.addWire(
                [c_north_low_old, ln_bot_north, -ln_north_far, -ln_floor_north]
            )
            s_north_low = occ.addBSplineFilling(wire_north_low)
            occ.synchronize()
            sl_soil_ext = occ.addSurfaceLoop(
                [
                    s_bottom_ext,
                    s_floor_ext,
                    s_west_low,
                    s_east_low,
                    s_south_low,
                    s_north_low,
                ]
            )
            vol_soil_plusX = occ.addVolume([sl_soil_ext])
            occ.synchronize()

            # -X WATER & SOIL EXTENSIONS
            occ = gmsh.model.occ
            occ.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, vol_water)
            eps = 1e-9 * max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin), 1.0)
            _, water_surfs = gmsh.model.getAdjacencies(3, vol_water)
            west_surfs = []
            for s in water_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(sxmin - xmin) < eps and abs(sxmax - xmin) < eps:
                    west_surfs.append(s)
            if not west_surfs:
                raise RuntimeError("No -X wall surface found on vol_water.")
            west_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[4]
                    - gmsh.model.getBoundingBox(2, s)[1]
                ),
                reverse=True,
            )
            sW = west_surfs[0]
            _, sW_curves = gmsh.model.getAdjacencies(2, sW)
            seen = set()
            curvesW = []
            for c in sW_curves:
                if c not in seen:
                    seen.add(c)
                    curvesW.append(c)
            c_top_W = c_bottom_W = c_south_W = c_north_W = None
            for c in curvesW:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmax) < eps and abs(czmax - zmax) < eps:
                    c_top_W = c
                elif abs(cymin - ymin) < eps and abs(cymax - ymin) < eps:
                    c_south_W = c
                elif abs(cymin - ymax) < eps and abs(cymax - ymax) < eps:
                    c_north_W = c
                else:
                    c_bottom_W = c
            if any(v is None for v in (c_top_W, c_bottom_W, c_south_W, c_north_W)):
                raise RuntimeError(
                    "Failed to classify west (-X) wall curves of water volume."
                )
            copies_W = occ.copy(
                [(1, c_top_W), (1, c_bottom_W), (1, c_south_W), (1, c_north_W)]
            )
            occ.synchronize()
            occ.translate(copies_W, -padX, 0.0, 0.0)
            occ.synchronize()
            c_top_new_W = copies_W[0][1]
            c_bottom_new_W = copies_W[1][1]
            c_south_new_W = copies_W[2][1]
            c_north_new_W = copies_W[3][1]
            _, top_pts_old_W = gmsh.model.getAdjacencies(1, c_top_W)
            _, top_pts_new_W = gmsh.model.getAdjacencies(1, c_top_new_W)
            top_pts_old_W = sorted(
                top_pts_old_W, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            top_pts_new_W = sorted(
                top_pts_new_W, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_top_SW_old, pt_top_NW_old = top_pts_old_W[0], top_pts_old_W[-1]
            pt_top_SW_new, pt_top_NW_new = top_pts_new_W[0], top_pts_new_W[-1]
            _, bot_pts_old_W = gmsh.model.getAdjacencies(1, c_bottom_W)
            _, bot_pts_new_W = gmsh.model.getAdjacencies(1, c_bottom_new_W)
            bot_pts_old_W = sorted(
                bot_pts_old_W, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            bot_pts_new_W = sorted(
                bot_pts_new_W, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_bot_S_old_W, pt_bot_N_old_W = bot_pts_old_W[0], bot_pts_old_W[-1]
            pt_bot_S_new_W, pt_bot_N_new_W = bot_pts_new_W[0], bot_pts_new_W[-1]
            ln_top_south_W = occ.addLine(pt_top_SW_old, pt_top_SW_new)
            ln_top_north_W = occ.addLine(pt_top_NW_old, pt_top_NW_new)
            ln_bot_south_W = occ.addLine(pt_bot_S_old_W, pt_bot_S_new_W)
            ln_bot_north_W = occ.addLine(pt_bot_N_old_W, pt_bot_N_new_W)
            occ.synchronize()
            wire_W_new = occ.addWire(
                [c_top_new_W, c_north_new_W, -c_bottom_new_W, -c_south_new_W]
            )
            sW_new = occ.addBSplineFilling(wire_W_new)
            wire_roof_W = occ.addWire(
                [c_top_W, ln_top_north_W, -c_top_new_W, -ln_top_south_W]
            )
            s_roof_W = occ.addBSplineFilling(wire_roof_W)
            wire_bottom_ext_W = occ.addWire(
                [c_bottom_W, ln_bot_north_W, -c_bottom_new_W, -ln_bot_south_W]
            )
            s_bottom_ext_W = occ.addBSplineFilling(wire_bottom_ext_W)
            wire_south_W = occ.addWire(
                [c_south_W, ln_top_south_W, -c_south_new_W, -ln_bot_south_W]
            )
            s_south_ext_W = occ.addBSplineFilling(wire_south_W)
            wire_north_W = occ.addWire(
                [c_north_W, ln_top_north_W, -c_north_new_W, -ln_bot_north_W]
            )
            s_north_ext_W = occ.addBSplineFilling(wire_north_W)
            occ.synchronize()
            sl_water_ext_W = occ.addSurfaceLoop(
                [sW, sW_new, s_roof_W, s_bottom_ext_W, s_south_ext_W, s_north_ext_W]
            )
            vol_water_minusX = occ.addVolume([sl_water_ext_W])
            occ.synchronize()

            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            eps_s = 1e-9 * max(
                abs(xmax_s - xmin_s), abs(ymax_s - ymin_s), abs(zmax_s - zmin_s), 1.0
            )
            _, soil_surfs = gmsh.model.getAdjacencies(3, vol_subsoil)
            west_soil_surfs = []
            for s in soil_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(sxmin - xmin_s) < eps_s and abs(sxmax - xmin_s) < eps_s:
                    west_soil_surfs.append(s)
            if not west_soil_surfs:
                raise RuntimeError("No -X wall surface found on vol_subsoil.")
            west_soil_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[5]
                    - gmsh.model.getBoundingBox(2, s)[2]
                ),
                reverse=True,
            )
            sW_soil = west_soil_surfs[0]
            _, sW_soil_curves = gmsh.model.getAdjacencies(2, sW_soil)
            seen = set()
            soil_curves_W = []
            for c in sW_soil_curves:
                if c not in seen:
                    seen.add(c)
                    soil_curves_W.append(c)
            c_top_rim_soil_W = c_bottom_flat_old_W = c_south_low_old_W = (
                c_north_low_old_W
            ) = None
            for c in soil_curves_W:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmin_s) < eps_s and abs(czmax - zmin_s) < eps_s:
                    c_bottom_flat_old_W = c
                elif abs(cymin - ymin_s) < eps_s and abs(cymax - ymin_s) < eps_s:
                    c_south_low_old_W = c
                elif abs(cymin - ymax_s) < eps_s and abs(cymax - ymax_s) < eps_s:
                    c_north_low_old_W = c
                else:
                    c_top_rim_soil_W = c
            if any(
                v is None
                for v in (
                    c_top_rim_soil_W,
                    c_bottom_flat_old_W,
                    c_south_low_old_W,
                    c_north_low_old_W,
                )
            ):
                raise RuntimeError("Failed to classify soil -X wall curves.")
            copies_floor_W = occ.copy([(1, c_bottom_flat_old_W)])
            occ.synchronize()
            occ.translate(copies_floor_W, -padX, 0.0, 0.0)
            occ.synchronize()
            c_bottom_flat_new_W = copies_floor_W[0][1]
            _, botflat_pts_old_W = gmsh.model.getAdjacencies(1, c_bottom_flat_old_W)
            _, botflat_pts_new_W = gmsh.model.getAdjacencies(1, c_bottom_flat_new_W)
            botflat_pts_old_W = sorted(
                botflat_pts_old_W,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[1],
            )
            botflat_pts_new_W = sorted(
                botflat_pts_new_W,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[1],
            )
            pt_floor_S_old_W, pt_floor_N_old_W = (
                botflat_pts_old_W[0],
                botflat_pts_old_W[-1],
            )
            pt_floor_S_new_W, pt_floor_N_new_W = (
                botflat_pts_new_W[0],
                botflat_pts_new_W[-1],
            )
            ln_south_far_W = occ.addLine(pt_floor_S_new_W, pt_bot_S_new_W)
            ln_north_far_W = occ.addLine(pt_floor_N_new_W, pt_bot_N_new_W)
            ln_floor_south_W = occ.addLine(pt_floor_S_old_W, pt_floor_S_new_W)
            ln_floor_north_W = occ.addLine(pt_floor_N_old_W, pt_floor_N_new_W)
            occ.synchronize()
            wire_floor_W = occ.addWire(
                [
                    c_bottom_flat_old_W,
                    ln_floor_north_W,
                    -c_bottom_flat_new_W,
                    -ln_floor_south_W,
                ]
            )
            s_floor_ext_W = occ.addBSplineFilling(wire_floor_W)
            wire_east_low_W = occ.addWire(
                [
                    c_top_rim_soil_W,
                    c_north_low_old_W,
                    -c_bottom_flat_old_W,
                    -c_south_low_old_W,
                ]
            )
            s_east_low_W = occ.addBSplineFilling(wire_east_low_W)
            wire_west_low_W = occ.addWire(
                [c_bottom_new_W, ln_north_far_W, -c_bottom_flat_new_W, -ln_south_far_W]
            )
            s_west_low_W = occ.addBSplineFilling(wire_west_low_W)
            wire_south_low_W = occ.addWire(
                [c_south_low_old_W, ln_bot_south_W, -ln_south_far_W, -ln_floor_south_W]
            )
            s_south_low_W = occ.addBSplineFilling(wire_south_low_W)
            wire_north_low_W = occ.addWire(
                [c_north_low_old_W, ln_bot_north_W, -ln_north_far_W, -ln_floor_north_W]
            )
            s_north_low_W = occ.addBSplineFilling(wire_north_low_W)
            occ.synchronize()
            sl_soil_ext_W = occ.addSurfaceLoop(
                [
                    s_bottom_ext_W,
                    s_floor_ext_W,
                    s_east_low_W,
                    s_west_low_W,
                    s_south_low_W,
                    s_north_low_W,
                ]
            )
            vol_soil_minusX = occ.addVolume([sl_soil_ext_W])
            occ.synchronize()

            # -Y WATER & SOIL EXTENSIONS
            occ = gmsh.model.occ
            occ.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, vol_water)
            eps = 1e-9 * max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin), 1.0)
            _, water_surfs = gmsh.model.getAdjacencies(3, vol_water)
            south_surfs = []
            for s in water_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(symin - ymin) < eps and abs(symax - ymin) < eps:
                    south_surfs.append(s)
            if not south_surfs:
                raise RuntimeError("No -Y wall surface found on vol_water.")
            south_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[3]
                    - gmsh.model.getBoundingBox(2, s)[0]
                ),
                reverse=True,
            )
            sS = south_surfs[0]
            _, sS_curves = gmsh.model.getAdjacencies(2, sS)
            seen = set()
            curvesS = []
            for c in sS_curves:
                if c not in seen:
                    seen.add(c)
                    curvesS.append(c)
            c_top_S = c_bottom_S = c_west_S = c_east_S = None
            for c in curvesS:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmax) < eps and abs(czmax - zmax) < eps:
                    c_top_S = c
                elif abs(cxmin - xmin) < eps and abs(cxmax - xmin) < eps:
                    c_west_S = c
                elif abs(cxmin - xmax) < eps and abs(cxmax - xmax) < eps:
                    c_east_S = c
                else:
                    c_bottom_S = c
            if any(v is None for v in (c_top_S, c_bottom_S, c_west_S, c_east_S)):
                raise RuntimeError(
                    "Failed to classify south (-Y) wall curves of water volume."
                )
            copies_S = occ.copy(
                [(1, c_top_S), (1, c_bottom_S), (1, c_west_S), (1, c_east_S)]
            )
            occ.synchronize()
            occ.translate(copies_S, 0.0, -padY, 0.0)
            occ.synchronize()
            c_top_new_S = copies_S[0][1]
            c_bottom_new_S = copies_S[1][1]
            c_west_new_S = copies_S[2][1]
            c_east_new_S = copies_S[3][1]
            _, top_pts_old_S = gmsh.model.getAdjacencies(1, c_top_S)
            _, top_pts_new_S = gmsh.model.getAdjacencies(1, c_top_new_S)
            top_pts_old_S = sorted(
                top_pts_old_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            top_pts_new_S = sorted(
                top_pts_new_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_top_SW_old, pt_top_SE_old = top_pts_old_S[0], top_pts_old_S[-1]
            pt_top_SW_new, pt_top_SE_new = top_pts_new_S[0], top_pts_new_S[-1]
            _, bot_pts_old_S = gmsh.model.getAdjacencies(1, c_bottom_S)
            _, bot_pts_new_S = gmsh.model.getAdjacencies(1, c_bottom_new_S)
            bot_pts_old_S = sorted(
                bot_pts_old_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            bot_pts_new_S = sorted(
                bot_pts_new_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_bot_W_old_S, pt_bot_E_old_S = bot_pts_old_S[0], bot_pts_old_S[-1]
            pt_bot_W_new_S, pt_bot_E_new_S = bot_pts_new_S[0], bot_pts_new_S[-1]
            ln_top_west_S = occ.addLine(pt_top_SW_old, pt_top_SW_new)
            ln_top_east_S = occ.addLine(pt_top_SE_old, pt_top_SE_new)
            ln_bot_west_S = occ.addLine(pt_bot_W_old_S, pt_bot_W_new_S)
            ln_bot_east_S = occ.addLine(pt_bot_E_old_S, pt_bot_E_new_S)
            occ.synchronize()
            wire_S_new = occ.addWire(
                [c_top_new_S, c_east_new_S, -c_bottom_new_S, -c_west_new_S]
            )
            sS_new = occ.addBSplineFilling(wire_S_new)
            wire_roof_S = occ.addWire(
                [c_top_S, ln_top_east_S, -c_top_new_S, -ln_top_west_S]
            )
            s_roof_S = occ.addBSplineFilling(wire_roof_S)
            wire_bottom_ext_S = occ.addWire(
                [c_bottom_S, ln_bot_east_S, -c_bottom_new_S, -ln_bot_west_S]
            )
            s_bottom_ext_S = occ.addBSplineFilling(wire_bottom_ext_S)
            wire_west_S = occ.addWire(
                [c_west_S, ln_top_west_S, -c_west_new_S, -ln_bot_west_S]
            )
            s_west_ext_S = occ.addBSplineFilling(wire_west_S)
            wire_east_S = occ.addWire(
                [c_east_S, ln_top_east_S, -c_east_new_S, -ln_bot_east_S]
            )
            s_east_ext_S = occ.addBSplineFilling(wire_east_S)
            occ.synchronize()
            sl_water_ext_S = occ.addSurfaceLoop(
                [sS, sS_new, s_roof_S, s_bottom_ext_S, s_west_ext_S, s_east_ext_S]
            )
            vol_water_minusY = occ.addVolume([sl_water_ext_S])
            occ.synchronize()

            # SOIL -Y extension (reuse s_bottom_ext_S)
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            eps_s = 1e-9 * max(
                abs(xmax_s - xmin_s), abs(ymax_s - ymin_s), abs(zmax_s - zmin_s), 1.0
            )
            _, soil_surfs = gmsh.model.getAdjacencies(3, vol_subsoil)
            south_soil_surfs = []
            for s in soil_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(symin - ymin_s) < eps_s and abs(symax - ymin_s) < eps_s:
                    south_soil_surfs.append(s)
            if not south_soil_surfs:
                raise RuntimeError("No -Y wall surface found on vol_subsoil.")
            south_soil_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[3]
                    - gmsh.model.getBoundingBox(2, s)[0]
                ),
                reverse=True,
            )
            sS_soil = south_soil_surfs[0]
            _, sS_soil_curves = gmsh.model.getAdjacencies(2, sS_soil)
            seen = set()
            soil_curves_S = []
            for c in sS_soil_curves:
                if c not in seen:
                    seen.add(c)
                    soil_curves_S.append(c)
            c_top_rim_soil_S = c_bottom_flat_old_S = c_west_low_old_S = (
                c_east_low_old_S
            ) = None
            for c in soil_curves_S:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmin_s) < eps_s and abs(czmax - zmin_s) < eps_s:
                    c_bottom_flat_old_S = c
                elif abs(cxmin - xmin_s) < eps_s and abs(cxmax - xmin_s) < eps_s:
                    c_west_low_old_S = c
                elif abs(cxmin - xmax_s) < eps_s and abs(cxmax - xmax_s) < eps_s:
                    c_east_low_old_S = c
                else:
                    c_top_rim_soil_S = c
            if any(
                v is None
                for v in (
                    c_top_rim_soil_S,
                    c_bottom_flat_old_S,
                    c_west_low_old_S,
                    c_east_low_old_S,
                )
            ):
                raise RuntimeError("Failed to classify soil -Y wall curves.")
            copies_floor_S = occ.copy([(1, c_bottom_flat_old_S)])
            occ.synchronize()
            occ.translate(copies_floor_S, 0.0, -padY, 0.0)
            occ.synchronize()
            c_bottom_flat_new_S = copies_floor_S[0][1]
            _, botflat_pts_old_S = gmsh.model.getAdjacencies(1, c_bottom_flat_old_S)
            _, botflat_pts_new_S = gmsh.model.getAdjacencies(1, c_bottom_flat_new_S)
            botflat_pts_old_S = sorted(
                botflat_pts_old_S,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[0],
            )
            botflat_pts_new_S = sorted(
                botflat_pts_new_S,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[0],
            )
            pt_floor_W_old_S, pt_floor_E_old_S = (
                botflat_pts_old_S[0],
                botflat_pts_old_S[-1],
            )
            pt_floor_W_new_S, pt_floor_E_new_S = (
                botflat_pts_new_S[0],
                botflat_pts_new_S[-1],
            )
            ln_west_far_S = occ.addLine(pt_floor_W_new_S, pt_bot_W_new_S)
            ln_east_far_S = occ.addLine(pt_floor_E_new_S, pt_bot_E_new_S)
            ln_floor_west_S = occ.addLine(pt_floor_W_old_S, pt_floor_W_new_S)
            ln_floor_east_S = occ.addLine(pt_floor_E_old_S, pt_floor_E_new_S)
            occ.synchronize()
            wire_floor_S = occ.addWire(
                [
                    c_bottom_flat_old_S,
                    ln_floor_east_S,
                    -c_bottom_flat_new_S,
                    -ln_floor_west_S,
                ]
            )
            s_floor_ext_S = occ.addBSplineFilling(wire_floor_S)
            wire_north_low_S = occ.addWire(
                [
                    c_top_rim_soil_S,
                    c_east_low_old_S,
                    -c_bottom_flat_old_S,
                    -c_west_low_old_S,
                ]
            )
            s_north_low_S = occ.addBSplineFilling(wire_north_low_S)
            wire_south_low_S = occ.addWire(
                [c_bottom_new_S, ln_east_far_S, -c_bottom_flat_new_S, -ln_west_far_S]
            )
            s_south_low_S = occ.addBSplineFilling(wire_south_low_S)
            wire_west_low_S = occ.addWire(
                [c_west_low_old_S, ln_bot_west_S, -ln_west_far_S, -ln_floor_west_S]
            )
            s_west_low_S = occ.addBSplineFilling(wire_west_low_S)
            wire_east_low_S = occ.addWire(
                [c_east_low_old_S, ln_bot_east_S, -ln_east_far_S, -ln_floor_east_S]
            )
            s_east_low_S = occ.addBSplineFilling(wire_east_low_S)
            occ.synchronize()
            sl_soil_ext_S = occ.addSurfaceLoop(
                [
                    s_bottom_ext_S,
                    s_floor_ext_S,
                    s_north_low_S,
                    s_south_low_S,
                    s_west_low_S,
                    s_east_low_S,
                ]
            )
            vol_soil_minusY = occ.addVolume([sl_soil_ext_S])
            occ.synchronize()

            # WATER +Y extension
            occ = gmsh.model.occ
            occ.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, vol_water)
            eps = 1e-9 * max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin), 1.0)
            _, water_surfs = gmsh.model.getAdjacencies(3, vol_water)
            north_surfs = []
            for s in water_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(symin - ymax) < eps and abs(symax - ymax) < eps:
                    north_surfs.append(s)
            if not north_surfs:
                raise RuntimeError("No +Y wall surface found on vol_water.")
            north_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[3]
                    - gmsh.model.getBoundingBox(2, s)[0]
                ),
                reverse=True,
            )
            sN = north_surfs[0]
            _, sN_curves = gmsh.model.getAdjacencies(2, sN)
            seen = set()
            curvesN = []
            for c in sN_curves:
                if c not in seen:
                    seen.add(c)
                    curvesN.append(c)
            c_top_N = c_bottom_N = c_west_N = c_east_N = None
            for c in curvesN:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmax) < eps and abs(czmax - zmax) < eps:
                    c_top_N = c
                elif abs(cxmin - xmin) < eps and abs(cxmax - xmin) < eps:
                    c_west_N = c
                elif abs(cxmin - xmax) < eps and abs(cxmax - xmax) < eps:
                    c_east_N = c
                else:
                    c_bottom_N = c
            if any(v is None for v in (c_top_N, c_bottom_N, c_west_N, c_east_N)):
                raise RuntimeError(
                    "Failed to classify north (+Y) wall curves of water volume."
                )
            copies_N = occ.copy(
                [(1, c_top_N), (1, c_bottom_N), (1, c_west_N), (1, c_east_N)]
            )
            occ.synchronize()
            occ.translate(copies_N, 0.0, padY, 0.0)
            occ.synchronize()
            c_top_new_N = copies_N[0][1]
            c_bottom_new_N = copies_N[1][1]
            c_west_new_N = copies_N[2][1]
            c_east_new_N = copies_N[3][1]
            _, top_pts_old_N = gmsh.model.getAdjacencies(1, c_top_N)
            _, top_pts_new_N = gmsh.model.getAdjacencies(1, c_top_new_N)
            top_pts_old_N = sorted(
                top_pts_old_N, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            top_pts_new_N = sorted(
                top_pts_new_N, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_top_NW_old, pt_top_NE_old = top_pts_old_N[0], top_pts_old_N[-1]
            pt_top_NW_new, pt_top_NE_new = top_pts_new_N[0], top_pts_new_N[-1]
            _, bot_pts_old_N = gmsh.model.getAdjacencies(1, c_bottom_N)
            _, bot_pts_new_N = gmsh.model.getAdjacencies(1, c_bottom_new_N)
            bot_pts_old_N = sorted(
                bot_pts_old_N, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            bot_pts_new_N = sorted(
                bot_pts_new_N, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_bot_W_old_N, pt_bot_E_old_N = bot_pts_old_N[0], bot_pts_old_N[-1]
            pt_bot_W_new_N, pt_bot_E_new_N = bot_pts_new_N[0], bot_pts_new_N[-1]
            ln_top_west_N = occ.addLine(pt_top_NW_old, pt_top_NW_new)
            ln_top_east_N = occ.addLine(pt_top_NE_old, pt_top_NE_new)
            ln_bot_west_N = occ.addLine(pt_bot_W_old_N, pt_bot_W_new_N)
            ln_bot_east_N = occ.addLine(pt_bot_E_old_N, pt_bot_E_new_N)
            occ.synchronize()
            wire_N_new = occ.addWire(
                [c_top_new_N, c_east_new_N, -c_bottom_new_N, -c_west_new_N]
            )
            sN_new = occ.addBSplineFilling(wire_N_new)
            wire_roof_N = occ.addWire(
                [c_top_N, ln_top_east_N, -c_top_new_N, -ln_top_west_N]
            )
            s_roof_N = occ.addBSplineFilling(wire_roof_N)
            wire_bottom_ext_N = occ.addWire(
                [c_bottom_N, ln_bot_east_N, -c_bottom_new_N, -ln_bot_west_N]
            )
            s_bottom_ext_N = occ.addBSplineFilling(wire_bottom_ext_N)
            wire_west_N = occ.addWire(
                [c_west_N, ln_top_west_N, -c_west_new_N, -ln_bot_west_N]
            )
            s_west_ext_N = occ.addBSplineFilling(wire_west_N)
            wire_east_N = occ.addWire(
                [c_east_N, ln_top_east_N, -c_east_new_N, -ln_bot_east_N]
            )
            s_east_ext_N = occ.addBSplineFilling(wire_east_N)
            occ.synchronize()
            sl_water_ext_N = occ.addSurfaceLoop(
                [sN, sN_new, s_roof_N, s_bottom_ext_N, s_west_ext_N, s_east_ext_N]
            )
            vol_water_plusY = occ.addVolume([sl_water_ext_N])
            occ.synchronize()

            # SOIL +Y extension (reuse s_bottom_ext_N)
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            eps_s = 1e-9 * max(
                abs(xmax_s - xmin_s), abs(ymax_s - ymin_s), abs(zmax_s - zmin_s), 1.0
            )
            _, soil_surfs = gmsh.model.getAdjacencies(3, vol_subsoil)
            north_soil_surfs = []
            for s in soil_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(symin - ymax_s) < eps_s and abs(symax - ymax_s) < eps_s:
                    north_soil_surfs.append(s)
            if not north_soil_surfs:
                raise RuntimeError("No +Y wall surface found on vol_subsoil.")
            north_soil_surfs.sort(
                key=lambda s: (
                    gmsh.model.getBoundingBox(2, s)[3]
                    - gmsh.model.getBoundingBox(2, s)[0]
                ),
                reverse=True,
            )
            sN_soil = north_soil_surfs[0]
            _, sN_soil_curves = gmsh.model.getAdjacencies(2, sN_soil)
            seen = set()
            soil_curves_N = []
            for c in sN_soil_curves:
                if c not in seen:
                    seen.add(c)
                    soil_curves_N.append(c)
            c_top_rim_soil_N = c_bottom_flat_old_N = c_west_low_old_N = (
                c_east_low_old_N
            ) = None
            for c in soil_curves_N:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(czmin - zmin_s) < eps_s and abs(czmax - zmin_s) < eps_s:
                    c_bottom_flat_old_N = c
                elif abs(cxmin - xmin_s) < eps_s and abs(cxmax - xmin_s) < eps_s:
                    c_west_low_old_N = c
                elif abs(cxmin - xmax_s) < eps_s and abs(cxmax - xmax_s) < eps_s:
                    c_east_low_old_N = c
                else:
                    c_top_rim_soil_N = c
            if any(
                v is None
                for v in (
                    c_top_rim_soil_N,
                    c_bottom_flat_old_N,
                    c_west_low_old_N,
                    c_east_low_old_N,
                )
            ):
                raise RuntimeError("Failed to classify soil +Y wall curves.")
            copies_floor_N = occ.copy([(1, c_bottom_flat_old_N)])
            occ.synchronize()
            occ.translate(copies_floor_N, 0.0, padY, 0.0)
            occ.synchronize()
            c_bottom_flat_new_N = copies_floor_N[0][1]
            _, botflat_pts_old_N = gmsh.model.getAdjacencies(1, c_bottom_flat_old_N)
            _, botflat_pts_new_N = gmsh.model.getAdjacencies(1, c_bottom_flat_new_N)
            botflat_pts_old_N = sorted(
                botflat_pts_old_N,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[0],
            )
            botflat_pts_new_N = sorted(
                botflat_pts_new_N,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[0],
            )
            pt_floor_W_old_N, pt_floor_E_old_N = (
                botflat_pts_old_N[0],
                botflat_pts_old_N[-1],
            )
            pt_floor_W_new_N, pt_floor_E_new_N = (
                botflat_pts_new_N[0],
                botflat_pts_new_N[-1],
            )
            ln_west_far_N = occ.addLine(pt_floor_W_new_N, pt_bot_W_new_N)
            ln_east_far_N = occ.addLine(pt_floor_E_new_N, pt_bot_E_new_N)
            ln_floor_west_N = occ.addLine(pt_floor_W_old_N, pt_floor_W_new_N)
            ln_floor_east_N = occ.addLine(pt_floor_E_old_N, pt_floor_E_new_N)
            occ.synchronize()
            wire_floor_N = occ.addWire(
                [
                    c_bottom_flat_old_N,
                    ln_floor_east_N,
                    -c_bottom_flat_new_N,
                    -ln_floor_west_N,
                ]
            )
            s_floor_ext_N = occ.addBSplineFilling(wire_floor_N)
            wire_south_low_N = occ.addWire(
                [
                    c_top_rim_soil_N,
                    c_east_low_old_N,
                    -c_bottom_flat_old_N,
                    -c_west_low_old_N,
                ]
            )
            s_south_low_N = occ.addBSplineFilling(wire_south_low_N)
            wire_north_low_N = occ.addWire(
                [c_bottom_new_N, ln_east_far_N, -c_bottom_flat_new_N, -ln_west_far_N]
            )
            s_north_low_N = occ.addBSplineFilling(wire_north_low_N)
            wire_west_low_N = occ.addWire(
                [c_west_low_old_N, ln_bot_west_N, -ln_west_far_N, -ln_floor_west_N]
            )
            s_west_low_N = occ.addBSplineFilling(wire_west_low_N)
            wire_east_low_N = occ.addWire(
                [c_east_low_old_N, ln_bot_east_N, -ln_east_far_N, -ln_floor_east_N]
            )
            s_east_low_N = occ.addBSplineFilling(wire_east_low_N)
            occ.synchronize()
            sl_soil_ext_N = occ.addSurfaceLoop(
                [
                    s_bottom_ext_N,
                    s_floor_ext_N,
                    s_south_low_N,
                    s_north_low_N,
                    s_west_low_N,
                    s_east_low_N,
                ]
            )
            vol_soil_plusY = occ.addVolume([sl_soil_ext_N])
            occ.synchronize()

            # WATER corner +X,+Y (NE)
            occ = gmsh.model.occ
            occ.synchronize()
            copies_corner_NE = occ.copy([(1, c_east_N)])
            occ.synchronize()
            occ.translate(copies_corner_NE, padX, padY, 0.0)
            occ.synchronize()
            c_corner_NE_new = copies_corner_NE[0][1]
            _, corner_new_pts = gmsh.model.getAdjacencies(1, c_corner_NE_new)
            corner_new_pts = sorted(
                corner_new_pts, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(corner_new_pts) < 2:
                raise RuntimeError("Far NE corner edge is missing endpoints.")
            pt_bot_NE_far = corner_new_pts[0]
            pt_top_NE_far = corner_new_pts[1]
            _, pts_E_new = gmsh.model.getAdjacencies(1, c_north_new)
            pts_E_new = sorted(
                pts_E_new, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_E_new) < 2:
                raise RuntimeError("c_north_new edge is missing endpoints.")
            pt_bot_NE_E = pts_E_new[0]
            pt_top_NE_E = pts_E_new[1]
            _, pts_N_new = gmsh.model.getAdjacencies(1, c_east_new_N)
            pts_N_new = sorted(
                pts_N_new, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_N_new) < 2:
                raise RuntimeError("c_east_new_N edge is missing endpoints.")
            pt_bot_NE_N = pts_N_new[0]
            pt_top_NE_N = pts_N_new[1]
            ln_top_far_east = occ.addLine(pt_top_NE_E, pt_top_NE_far)
            ln_top_far_north = occ.addLine(pt_top_NE_N, pt_top_NE_far)
            ln_bot_far_east = occ.addLine(pt_bot_NE_E, pt_bot_NE_far)
            ln_bot_far_north = occ.addLine(pt_bot_NE_N, pt_bot_NE_far)
            occ.synchronize()
            wire_top_diag = occ.addWire(
                [ln_top_north, ln_top_far_east, -ln_top_far_north, -ln_top_east_N]
            )
            s_top_diag = occ.addBSplineFilling(wire_top_diag)
            wire_bot_diag = occ.addWire(
                [ln_bot_north, ln_bot_far_east, -ln_bot_far_north, -ln_bot_east_N]
            )
            s_bot_diag = occ.addBSplineFilling(wire_bot_diag)
            wire_east_diag = occ.addWire(
                [c_north_new, ln_top_far_east, -c_corner_NE_new, -ln_bot_far_east]
            )
            s_east_diag = occ.addBSplineFilling(wire_east_diag)
            wire_north_diag = occ.addWire(
                [c_east_new_N, ln_top_far_north, -c_corner_NE_new, -ln_bot_far_north]
            )
            s_north_diag = occ.addBSplineFilling(wire_north_diag)
            occ.synchronize()
            sl_diag_NE = occ.addSurfaceLoop(
                [
                    s_east_ext_N,
                    s_north_ext,
                    s_top_diag,
                    s_bot_diag,
                    s_east_diag,
                    s_north_diag,
                ]
            )
            vol_water_diag_NE = occ.addVolume([sl_diag_NE])
            occ.synchronize()

            # SOIL corner +X,+Y (NE)
            occ = gmsh.model.occ
            occ.synchronize()
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            _, corner_new_pts = gmsh.model.getAdjacencies(1, c_corner_NE_new)
            corner_new_pts = sorted(
                corner_new_pts, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(corner_new_pts) < 2:
                raise RuntimeError(
                    "NE soil corner: far NE interface vertical has no endpoints."
                )
            pt_bot_NE_far = corner_new_pts[0]
            _, botflat_pts_E = gmsh.model.getAdjacencies(1, c_bottom_flat_new)
            botflat_pts_E = sorted(
                botflat_pts_E, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_floor_N_new = botflat_pts_E[-1]
            _, botflat_pts_N = gmsh.model.getAdjacencies(1, c_bottom_flat_new_N)
            botflat_pts_N = sorted(
                botflat_pts_N, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_floor_E_new_N = botflat_pts_N[-1]
            x_far = max(
                gmsh.model.getBoundingBox(0, pt_floor_N_new)[0],
                gmsh.model.getBoundingBox(0, pt_floor_E_new_N)[0],
            )
            y_far = max(
                gmsh.model.getBoundingBox(0, pt_floor_N_new)[1],
                gmsh.model.getBoundingBox(0, pt_floor_E_new_N)[1],
            )
            pt_floor_NE_far = occ.addPoint(x_far, y_far, zmin_s)
            ln_corner_floor_NE = occ.addLine(pt_floor_NE_far, pt_bot_NE_far)
            ln_floor_far_east = occ.addLine(pt_floor_N_new, pt_floor_NE_far)
            ln_floor_far_north = occ.addLine(pt_floor_E_new_N, pt_floor_NE_far)
            occ.synchronize()
            wire_east_far_low = occ.addWire(
                [ln_north_far, ln_bot_far_east, -ln_corner_floor_NE, -ln_floor_far_east]
            )
            s_east_far_low = occ.addBSplineFilling(wire_east_far_low)
            wire_north_far_low = occ.addWire(
                [
                    ln_east_far_N,
                    ln_bot_far_north,
                    -ln_corner_floor_NE,
                    -ln_floor_far_north,
                ]
            )
            s_north_far_low = occ.addBSplineFilling(wire_north_far_low)
            wire_bottom_diag_low = occ.addWire(
                [
                    ln_floor_north,
                    ln_floor_far_east,
                    -ln_floor_far_north,
                    -ln_floor_east_N,
                ]
            )
            s_bottom_diag_low = occ.addBSplineFilling(wire_bottom_diag_low)
            occ.synchronize()
            sl_soil_diag_NE = occ.addSurfaceLoop(
                [
                    s_north_low,
                    s_east_low_N,
                    s_east_far_low,
                    s_north_far_low,
                    s_bottom_diag_low,
                    s_bot_diag,
                ]
            )
            vol_soil_diag_NE = occ.addVolume([sl_soil_diag_NE])
            occ.synchronize()

            # WATER corner -X,+Y (NW)
            occ = gmsh.model.occ
            occ.synchronize()
            copies_corner_NW = occ.copy([(1, c_west_N)])
            occ.synchronize()
            occ.translate(copies_corner_NW, -padX, padY, 0.0)
            occ.synchronize()
            c_corner_NW_new = copies_corner_NW[0][1]
            _, corner_new_pts_NW = gmsh.model.getAdjacencies(1, c_corner_NW_new)
            corner_new_pts_NW = sorted(
                corner_new_pts_NW,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[2],
            )
            if len(corner_new_pts_NW) < 2:
                raise RuntimeError("Far NW corner edge is missing endpoints.")
            pt_bot_NW_far = corner_new_pts_NW[0]
            pt_top_NW_far = corner_new_pts_NW[1]
            _, pts_W_new = gmsh.model.getAdjacencies(1, c_north_new_W)
            pts_W_new = sorted(
                pts_W_new, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_W_new) < 2:
                raise RuntimeError("c_north_new_W edge is missing endpoints.")
            pt_bot_NW_W = pts_W_new[0]
            pt_top_NW_W = pts_W_new[1]
            _, pts_N_new = gmsh.model.getAdjacencies(1, c_west_new_N)
            pts_N_new = sorted(
                pts_N_new, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_N_new) < 2:
                raise RuntimeError("c_west_new_N edge is missing endpoints.")
            pt_bot_NW_N = pts_N_new[0]
            pt_top_NW_N = pts_N_new[1]
            ln_top_far_west = occ.addLine(pt_top_NW_W, pt_top_NW_far)
            ln_top_far_north_W = occ.addLine(pt_top_NW_N, pt_top_NW_far)
            ln_bot_far_west = occ.addLine(pt_bot_NW_W, pt_bot_NW_far)
            ln_bot_far_north_W = occ.addLine(pt_bot_NW_N, pt_bot_NW_far)
            occ.synchronize()
            wire_top_diag_NW = occ.addWire(
                [ln_top_north_W, ln_top_far_west, -ln_top_far_north_W, -ln_top_west_N]
            )
            s_top_diag_NW = occ.addBSplineFilling(wire_top_diag_NW)
            wire_bot_diag_NW = occ.addWire(
                [ln_bot_north_W, ln_bot_far_west, -ln_bot_far_north_W, -ln_bot_west_N]
            )
            s_bot_diag_NW = occ.addBSplineFilling(wire_bot_diag_NW)
            wire_west_diag = occ.addWire(
                [c_north_new_W, ln_top_far_west, -c_corner_NW_new, -ln_bot_far_west]
            )
            s_west_diag = occ.addBSplineFilling(wire_west_diag)
            wire_north_diag_NW = occ.addWire(
                [
                    c_west_new_N,
                    ln_top_far_north_W,
                    -c_corner_NW_new,
                    -ln_bot_far_north_W,
                ]
            )
            s_north_diag_NW = occ.addBSplineFilling(wire_north_diag_NW)
            occ.synchronize()
            sl_diag_NW = occ.addSurfaceLoop(
                [
                    s_west_ext_N,
                    s_north_ext_W,
                    s_top_diag_NW,
                    s_bot_diag_NW,
                    s_west_diag,
                    s_north_diag_NW,
                ]
            )
            vol_water_diag_NW = occ.addVolume([sl_diag_NW])
            occ.synchronize()

            # SOIL corner -X,+Y (NW)
            occ = gmsh.model.occ
            occ.synchronize()
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            _, corner_new_pts_NW = gmsh.model.getAdjacencies(1, c_corner_NW_new)
            corner_new_pts_NW = sorted(
                corner_new_pts_NW,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[2],
            )
            if len(corner_new_pts_NW) < 2:
                raise RuntimeError(
                    "NW soil corner: far NW interface vertical has no endpoints."
                )
            pt_bot_NW_far = corner_new_pts_NW[0]
            _, botflat_pts_W = gmsh.model.getAdjacencies(1, c_bottom_flat_new_W)
            botflat_pts_W = sorted(
                botflat_pts_W, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_floor_N_new_W = botflat_pts_W[-1]
            _, botflat_pts_N = gmsh.model.getAdjacencies(1, c_bottom_flat_new_N)
            botflat_pts_N = sorted(
                botflat_pts_N, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_floor_W_new_N = botflat_pts_N[0]
            x_far_W = min(
                gmsh.model.getBoundingBox(0, pt_floor_N_new_W)[0],
                gmsh.model.getBoundingBox(0, pt_floor_W_new_N)[0],
            )
            y_far_N = max(
                gmsh.model.getBoundingBox(0, pt_floor_N_new_W)[1],
                gmsh.model.getBoundingBox(0, pt_floor_W_new_N)[1],
            )
            pt_floor_NW_far = occ.addPoint(x_far_W, y_far_N, zmin_s)
            ln_corner_floor_NW = occ.addLine(pt_floor_NW_far, pt_bot_NW_far)
            ln_floor_far_west = occ.addLine(pt_floor_N_new_W, pt_floor_NW_far)
            ln_floor_far_north_W = occ.addLine(pt_floor_W_new_N, pt_floor_NW_far)
            occ.synchronize()
            wire_west_far_low = occ.addWire(
                [
                    ln_north_far_W,
                    ln_bot_far_west,
                    -ln_corner_floor_NW,
                    -ln_floor_far_west,
                ]
            )
            s_west_far_low = occ.addBSplineFilling(wire_west_far_low)
            wire_north_far_low_W = occ.addWire(
                [
                    ln_west_far_N,
                    ln_bot_far_north_W,
                    -ln_corner_floor_NW,
                    -ln_floor_far_north_W,
                ]
            )
            s_north_far_low_W = occ.addBSplineFilling(wire_north_far_low_W)
            wire_bottom_diag_low_NW = occ.addWire(
                [
                    ln_floor_north_W,
                    ln_floor_far_west,
                    -ln_floor_far_north_W,
                    -ln_floor_west_N,
                ]
            )
            s_bottom_diag_low_NW = occ.addBSplineFilling(wire_bottom_diag_low_NW)
            occ.synchronize()
            sl_soil_diag_NW = occ.addSurfaceLoop(
                [
                    s_north_low_W,
                    s_west_low_N,
                    s_west_far_low,
                    s_north_far_low_W,
                    s_bottom_diag_low_NW,
                    s_bot_diag_NW,
                ]
            )
            vol_soil_diag_NW = occ.addVolume([sl_soil_diag_NW])
            occ.synchronize()

            # WATER corner -X,-Y (SW)
            occ = gmsh.model.occ
            occ.synchronize()
            copies_corner_SW = occ.copy([(1, c_south_W)])
            occ.synchronize()
            occ.translate(copies_corner_SW, -padX, -padY, 0.0)
            occ.synchronize()
            c_corner_SW_new = copies_corner_SW[0][1]
            _, corner_new_pts_SW = gmsh.model.getAdjacencies(1, c_corner_SW_new)
            corner_new_pts_SW = sorted(
                corner_new_pts_SW,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[2],
            )
            if len(corner_new_pts_SW) < 2:
                raise RuntimeError("Far SW corner edge is missing endpoints.")
            pt_bot_SW_far = corner_new_pts_SW[0]
            pt_top_SW_far = corner_new_pts_SW[1]
            _, pts_W_new_S = gmsh.model.getAdjacencies(1, c_south_new_W)
            pts_W_new_S = sorted(
                pts_W_new_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_W_new_S) < 2:
                raise RuntimeError("c_south_new_W edge is missing endpoints.")
            pt_bot_SW_W = pts_W_new_S[0]
            pt_top_SW_W = pts_W_new_S[1]
            _, pts_S_new_W = gmsh.model.getAdjacencies(1, c_west_new_S)
            pts_S_new_W = sorted(
                pts_S_new_W, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_S_new_W) < 2:
                raise RuntimeError("c_west_new_S edge is missing endpoints.")
            pt_bot_SW_S = pts_S_new_W[0]
            pt_top_SW_S = pts_S_new_W[1]
            ln_top_far_west_S = occ.addLine(pt_top_SW_W, pt_top_SW_far)
            ln_top_far_south_W = occ.addLine(pt_top_SW_S, pt_top_SW_far)
            ln_bot_far_west_S = occ.addLine(pt_bot_SW_W, pt_bot_SW_far)
            ln_bot_far_south_W = occ.addLine(pt_bot_SW_S, pt_bot_SW_far)
            occ.synchronize()
            wire_top_diag_SW = occ.addWire(
                [ln_top_south_W, ln_top_far_west_S, -ln_top_far_south_W, -ln_top_west_S]
            )
            s_top_diag_SW = occ.addBSplineFilling(wire_top_diag_SW)
            wire_bot_diag_SW = occ.addWire(
                [ln_bot_south_W, ln_bot_far_west_S, -ln_bot_far_south_W, -ln_bot_west_S]
            )
            s_bot_diag_SW = occ.addBSplineFilling(wire_bot_diag_SW)
            wire_west_diag_SW = occ.addWire(
                [c_south_new_W, ln_top_far_west_S, -c_corner_SW_new, -ln_bot_far_west_S]
            )
            s_west_diag_SW = occ.addBSplineFilling(wire_west_diag_SW)
            wire_south_diag_SW = occ.addWire(
                [
                    c_west_new_S,
                    ln_top_far_south_W,
                    -c_corner_SW_new,
                    -ln_bot_far_south_W,
                ]
            )
            s_south_diag_SW = occ.addBSplineFilling(wire_south_diag_SW)
            occ.synchronize()
            sl_diag_SW = occ.addSurfaceLoop(
                [
                    s_west_ext_S,
                    s_south_ext_W,
                    s_top_diag_SW,
                    s_bot_diag_SW,
                    s_west_diag_SW,
                    s_south_diag_SW,
                ]
            )
            vol_water_diag_SW = occ.addVolume([sl_diag_SW])
            occ.synchronize()

            # SOIL corner -X,-Y (SW)
            occ = gmsh.model.occ
            occ.synchronize()
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            _, corner_new_pts_SW = gmsh.model.getAdjacencies(1, c_corner_SW_new)
            corner_new_pts_SW = sorted(
                corner_new_pts_SW,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[2],
            )
            if len(corner_new_pts_SW) < 2:
                raise RuntimeError(
                    "SW soil corner: far SW interface vertical has no endpoints."
                )
            pt_bot_SW_far = corner_new_pts_SW[0]
            _, botflat_pts_W_S = gmsh.model.getAdjacencies(1, c_bottom_flat_new_W)
            botflat_pts_W_S = sorted(
                botflat_pts_W_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_floor_S_new_W = botflat_pts_W_S[0]
            _, botflat_pts_S_W = gmsh.model.getAdjacencies(1, c_bottom_flat_new_S)
            botflat_pts_S_W = sorted(
                botflat_pts_S_W, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_floor_W_new_S = botflat_pts_S_W[0]
            x_far_W = min(
                gmsh.model.getBoundingBox(0, pt_floor_S_new_W)[0],
                gmsh.model.getBoundingBox(0, pt_floor_W_new_S)[0],
            )
            y_far_S = min(
                gmsh.model.getBoundingBox(0, pt_floor_S_new_W)[1],
                gmsh.model.getBoundingBox(0, pt_floor_W_new_S)[1],
            )
            pt_floor_SW_far = occ.addPoint(x_far_W, y_far_S, zmin_s)
            ln_corner_floor_SW = occ.addLine(pt_floor_SW_far, pt_bot_SW_far)
            ln_floor_far_west_S = occ.addLine(pt_floor_S_new_W, pt_floor_SW_far)
            ln_floor_far_south_W = occ.addLine(pt_floor_W_new_S, pt_floor_SW_far)
            occ.synchronize()
            wire_west_far_low_SW = occ.addWire(
                [
                    ln_south_far_W,
                    ln_bot_far_west_S,
                    -ln_corner_floor_SW,
                    -ln_floor_far_west_S,
                ]
            )
            s_west_far_low_SW = occ.addBSplineFilling(wire_west_far_low_SW)
            wire_south_far_low_SW = occ.addWire(
                [
                    ln_west_far_S,
                    ln_bot_far_south_W,
                    -ln_corner_floor_SW,
                    -ln_floor_far_south_W,
                ]
            )
            s_south_far_low_SW = occ.addBSplineFilling(wire_south_far_low_SW)
            wire_bottom_diag_low_SW = occ.addWire(
                [
                    ln_floor_south_W,
                    ln_floor_far_west_S,
                    -ln_floor_far_south_W,
                    -ln_floor_west_S,
                ]
            )
            s_bottom_diag_low_SW = occ.addBSplineFilling(wire_bottom_diag_low_SW)
            occ.synchronize()
            sl_soil_diag_SW = occ.addSurfaceLoop(
                [
                    s_south_low_W,
                    s_west_low_S,
                    s_west_far_low_SW,
                    s_south_far_low_SW,
                    s_bottom_diag_low_SW,
                    s_bot_diag_SW,
                ]
            )
            vol_soil_diag_SW = occ.addVolume([sl_soil_diag_SW])
            occ.synchronize()

            # Build the +X, -Y (SE) WATER corner volume
            occ = gmsh.model.occ
            occ.synchronize()
            print("Building +X, -Y (SE) water corner volume...")
            copies_corner_SE = occ.copy([(1, c_south)])
            occ.synchronize()
            occ.translate(copies_corner_SE, padX, -padY, 0.0)
            occ.synchronize()
            c_corner_SE_new = copies_corner_SE[0][1]
            _, corner_new_pts_SE = gmsh.model.getAdjacencies(1, c_corner_SE_new)
            corner_new_pts_SE = sorted(
                corner_new_pts_SE,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[2],
            )
            if len(corner_new_pts_SE) < 2:
                raise RuntimeError("Far SE corner edge is missing endpoints.")
            pt_bot_SE_far = corner_new_pts_SE[0]
            pt_top_SE_far = corner_new_pts_SE[1]
            _, pts_E_new_S = gmsh.model.getAdjacencies(1, c_south_new)
            pts_E_new_S = sorted(
                pts_E_new_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_E_new_S) < 2:
                raise RuntimeError("c_south_new edge is missing endpoints.")
            pt_bot_SE_E = pts_E_new_S[0]
            pt_top_SE_E = pts_E_new_S[1]
            _, pts_S_new_E = gmsh.model.getAdjacencies(1, c_east_new_S)
            pts_S_new_E = sorted(
                pts_S_new_E, key=lambda p: gmsh.model.getBoundingBox(0, p)[2]
            )
            if len(pts_S_new_E) < 2:
                raise RuntimeError("c_east_new_S edge is missing endpoints.")
            pt_bot_SE_S = pts_S_new_E[0]
            pt_top_SE_S = pts_S_new_E[1]
            ln_top_far_east_S = occ.addLine(pt_top_SE_E, pt_top_SE_far)
            ln_top_far_south_E = occ.addLine(pt_top_SE_S, pt_top_SE_far)
            ln_bot_far_east_S = occ.addLine(pt_bot_SE_E, pt_bot_SE_far)
            ln_bot_far_south_E = occ.addLine(pt_bot_SE_S, pt_bot_SE_far)
            occ.synchronize()
            wire_top_diag_SE = occ.addWire(
                [ln_top_south, ln_top_far_east_S, -ln_top_far_south_E, -ln_top_east_S]
            )
            s_top_diag_SE = occ.addBSplineFilling(wire_top_diag_SE)
            wire_bot_diag_SE = occ.addWire(
                [ln_bot_south, ln_bot_far_east_S, -ln_bot_far_south_E, -ln_bot_east_S]
            )
            s_bot_diag_SE = occ.addBSplineFilling(wire_bot_diag_SE)
            wire_east_diag_SE = occ.addWire(
                [c_south_new, ln_top_far_east_S, -c_corner_SE_new, -ln_bot_far_east_S]
            )
            s_east_diag_SE = occ.addBSplineFilling(wire_east_diag_SE)
            wire_south_diag_SE = occ.addWire(
                [
                    c_east_new_S,
                    ln_top_far_south_E,
                    -c_corner_SE_new,
                    -ln_bot_far_south_E,
                ]
            )
            s_south_diag_SE = occ.addBSplineFilling(wire_south_diag_SE)
            occ.synchronize()
            sl_diag_SE = occ.addSurfaceLoop(
                [
                    s_east_ext_S,
                    s_south_ext,
                    s_top_diag_SE,
                    s_bot_diag_SE,
                    s_east_diag_SE,
                    s_south_diag_SE,
                ]
            )
            vol_water_diag_SE = occ.addVolume([sl_diag_SE])
            occ.synchronize()
            print(f"Created +X, -Y corner water volume: {vol_water_diag_SE}")

            occ = gmsh.model.occ
            occ.synchronize()
            print("Building +X,-Y (SE) SOIL corner volume...")
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            _, corner_new_pts_SE = gmsh.model.getAdjacencies(1, c_corner_SE_new)
            corner_new_pts_SE = sorted(
                corner_new_pts_SE,
                key=lambda p: gmsh.model.getBoundingBox(0, p)[2],
            )
            if len(corner_new_pts_SE) < 2:
                raise RuntimeError(
                    "SE soil corner: far SE interface vertical has no endpoints."
                )
            pt_bot_SE_far = corner_new_pts_SE[0]
            _, botflat_pts_E_S = gmsh.model.getAdjacencies(1, c_bottom_flat_new)
            botflat_pts_E_S = sorted(
                botflat_pts_E_S, key=lambda p: gmsh.model.getBoundingBox(0, p)[1]
            )
            pt_floor_S_new = botflat_pts_E_S[0]
            _, botflat_pts_S_E = gmsh.model.getAdjacencies(1, c_bottom_flat_new_S)
            botflat_pts_S_E = sorted(
                botflat_pts_S_E, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_floor_E_new_S = botflat_pts_S_E[-1]
            x_far_E = max(
                gmsh.model.getBoundingBox(0, pt_floor_S_new)[0],
                gmsh.model.getBoundingBox(0, pt_floor_E_new_S)[0],
            )
            y_far_S = min(
                gmsh.model.getBoundingBox(0, pt_floor_S_new)[1],
                gmsh.model.getBoundingBox(0, pt_floor_E_new_S)[1],
            )
            pt_floor_SE_far = occ.addPoint(x_far_E, y_far_S, zmin_s)
            ln_corner_floor_SE = occ.addLine(pt_floor_SE_far, pt_bot_SE_far)
            ln_floor_far_east_S = occ.addLine(pt_floor_S_new, pt_floor_SE_far)
            ln_floor_far_south_E = occ.addLine(pt_floor_E_new_S, pt_floor_SE_far)
            occ.synchronize()
            wire_east_far_low_SE = occ.addWire(
                [
                    ln_south_far,
                    ln_bot_far_east_S,
                    -ln_corner_floor_SE,
                    -ln_floor_far_east_S,
                ]
            )
            s_east_far_low_SE = occ.addBSplineFilling(wire_east_far_low_SE)
            wire_south_far_low_SE = occ.addWire(
                [
                    ln_east_far_S,
                    ln_bot_far_south_E,
                    -ln_corner_floor_SE,
                    -ln_floor_far_south_E,
                ]
            )
            s_south_far_low_SE = occ.addBSplineFilling(wire_south_far_low_SE)
            wire_bottom_diag_low_SE = occ.addWire(
                [
                    ln_floor_south,
                    ln_floor_far_east_S,
                    -ln_floor_far_south_E,
                    -ln_floor_east_S,
                ]
            )
            s_bottom_diag_low_SE = occ.addBSplineFilling(wire_bottom_diag_low_SE)
            occ.synchronize()
            sl_soil_diag_SE = occ.addSurfaceLoop(
                [
                    s_south_low,
                    s_east_low_S,
                    s_east_far_low_SE,
                    s_south_far_low_SE,
                    s_bottom_diag_low_SE,
                    s_bot_diag_SE,
                ]
            )
            vol_soil_diag_SE = occ.addVolume([sl_soil_diag_SE])
            occ.synchronize()
            print(
                f"Created +X,-Y SOIL corner volume (bottom diagonal): {vol_soil_diag_SE}"
            )

            occ = gmsh.model.occ
            occ.synchronize()
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(
                3, vol_subsoil
            )
            eps_s = 1e-9 * max(
                abs(xmax_s - xmin_s), abs(ymax_s - ymin_s), abs(zmax_s - zmin_s), 1.0
            )
            _, soil_surfs = gmsh.model.getAdjacencies(3, vol_subsoil)
            bottom_faces = []
            for s in soil_surfs:
                sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(
                    2, s
                )
                if abs(szmin - zmin_s) < eps_s and abs(szmax - zmin_s) < eps_s:
                    bottom_faces.append(s)
            if not bottom_faces:
                raise RuntimeError(
                    "Could not find a flat bottom face (z = zmin) on the subsoil volume."
                )
            bottom_faces.sort(
                key=lambda s: (
                    (
                        gmsh.model.getBoundingBox(2, s)[3]
                        - gmsh.model.getBoundingBox(2, s)[0]
                    )
                    * (
                        gmsh.model.getBoundingBox(2, s)[4]
                        - gmsh.model.getBoundingBox(2, s)[1]
                    )
                ),
                reverse=True,
            )
            s_bottom_top = bottom_faces[0]
            _, face_curves = gmsh.model.getAdjacencies(2, s_bottom_top)
            face_curves = list(dict.fromkeys(face_curves))
            c_south_old = c_east_old = c_north_old = c_west_old = None
            for c in face_curves:
                cxmin, cymin, czmin, cxmax, cymax, czmax = gmsh.model.getBoundingBox(
                    1, c
                )
                if abs(cymin - ymin_s) < eps_s and abs(cymax - ymin_s) < eps_s:
                    c_south_old = c
                elif abs(cymin - ymax_s) < eps_s and abs(cymax - ymax_s) < eps_s:
                    c_north_old = c
                elif abs(cxmin - xmin_s) < eps_s and abs(cxmax - xmin_s) < eps_s:
                    c_west_old = c
                elif abs(cxmin - xmax_s) < eps_s and abs(cxmax - xmax_s) < eps_s:
                    c_east_old = c
            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify bottom-face boundary curves (south/east/north/west)."
                )
            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()
            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]
            _, south_old_pts = gmsh.model.getAdjacencies(1, c_south_old)
            _, south_new_pts = gmsh.model.getAdjacencies(1, c_south_new)
            south_old_pts = sorted(
                south_old_pts, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            south_new_pts = sorted(
                south_new_pts, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_SW_old, pt_SE_old = south_old_pts[0], south_old_pts[-1]
            pt_SW_new, pt_SE_new = south_new_pts[0], south_new_pts[-1]
            _, north_old_pts = gmsh.model.getAdjacencies(1, c_north_old)
            _, north_new_pts = gmsh.model.getAdjacencies(1, c_north_new)
            north_old_pts = sorted(
                north_old_pts, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            north_new_pts = sorted(
                north_new_pts, key=lambda p: gmsh.model.getBoundingBox(0, p)[0]
            )
            pt_NW_old, pt_NE_old = north_old_pts[0], north_old_pts[-1]
            pt_NW_new, pt_NE_new = north_new_pts[0], north_new_pts[-1]
            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()
            wire_southZ = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ = occ.addBSplineFilling(wire_southZ)
            wire_eastZ = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ = occ.addBSplineFilling(wire_eastZ)
            wire_northZ = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ = occ.addBSplineFilling(wire_northZ)
            wire_westZ = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ = occ.addBSplineFilling(wire_westZ)
            wire_bottom_new = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_new = occ.addBSplineFilling(wire_bottom_new)
            occ.synchronize()
            sl_bottom_ext = occ.addSurfaceLoop(
                [s_bottom_top, s_southZ, s_eastZ, s_northZ, s_westZ, s_bottom_new]
            )
            vol_soil_bottom_ext = occ.addVolume([sl_bottom_ext])
            occ.synchronize()
            print(
                f"Created bottom cube extension (soil) vol tag = {vol_soil_bottom_ext}, thickness padZ = {padZ}"
            )

            # +X BOTTOM EXTENSION (downward by padZ)
            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-12):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError("No flat bottom face found on the volume.")
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_plusX_bottom = s_floor_ext
            except NameError:
                try:
                    s_top_plusX_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_plusX, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Please provide s_floor_ext or ensure vol_soil_plusX exists to auto-detect its bottom face."
                    )
            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_plusX_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)
            _, top_curves = gmsh.model.getAdjacencies(2, s_top_plusX_bottom)
            top_curves = list(dict.fromkeys(top_curves))
            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e
            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify +X floor face edges (south/east/north/west)."
                )
            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()
            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, key_axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if key_axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            pt_SW_old, pt_SE_old = (
                _sorted_pts(c_south_old, "x")[0],
                _sorted_pts(c_south_old, "x")[-1],
            )
            pt_SW_new, pt_SE_new = (
                _sorted_pts(c_south_new, "x")[0],
                _sorted_pts(c_south_new, "x")[-1],
            )
            pt_NW_old, pt_NE_old = (
                _sorted_pts(c_north_old, "x")[0],
                _sorted_pts(c_north_old, "x")[-1],
            )
            pt_NW_new, pt_NE_new = (
                _sorted_pts(c_north_new, "x")[0],
                _sorted_pts(c_north_new, "x")[-1],
            )
            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()
            wire_southZ = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_X = occ.addBSplineFilling(wire_southZ)
            wire_eastZ = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_X = occ.addBSplineFilling(wire_eastZ)
            wire_northZ = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_X = occ.addBSplineFilling(wire_northZ)
            wire_westZ = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_X = occ.addBSplineFilling(wire_westZ)
            wire_bottom_new_X = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_new_X = occ.addBSplineFilling(wire_bottom_new_X)
            occ.synchronize()
            sl_plusX_bottom = occ.addSurfaceLoop(
                [
                    s_top_plusX_bottom,
                    s_southZ_X,
                    s_eastZ_X,
                    s_northZ_X,
                    s_westZ_X,
                    s_bottom_new_X,
                ]
            )
            vol_soil_plusX_bottom_ext = occ.addVolume([sl_plusX_bottom])
            occ.synchronize()
            print(
                f"[+X bottom] Created bottom extension under +X extension: vol={vol_soil_plusX_bottom_ext}, padZ={padZ}"
            )

            # -X BOTTOM EXTENSION (downward by padZ)
            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-12):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError("No flat bottom face found on the volume.")
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_minusX_bottom = s_floor_ext_W
            except NameError:
                try:
                    s_top_minusX_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_minusX, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Provide s_floor_ext_W or ensure vol_soil_minusX exists to auto-detect its bottom face."
                    )
            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_minusX_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)
            _, top_curves = gmsh.model.getAdjacencies(2, s_top_minusX_bottom)
            top_curves = list(dict.fromkeys(top_curves))
            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e
            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify -X floor face edges (south/east/north/west)."
                )
            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()
            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            pt_SW_old, pt_SE_old = (
                _sorted_pts(c_south_old, "x")[0],
                _sorted_pts(c_south_old, "x")[-1],
            )
            pt_SW_new, pt_SE_new = (
                _sorted_pts(c_south_new, "x")[0],
                _sorted_pts(c_south_new, "x")[-1],
            )
            pt_NW_old, pt_NE_old = (
                _sorted_pts(c_north_old, "x")[0],
                _sorted_pts(c_north_old, "x")[-1],
            )
            pt_NW_new, pt_NE_new = (
                _sorted_pts(c_north_new, "x")[0],
                _sorted_pts(c_north_new, "x")[-1],
            )
            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()
            wire_southZ = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_W = occ.addBSplineFilling(wire_southZ)
            wire_eastZ = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_W = occ.addBSplineFilling(wire_eastZ)
            wire_northZ = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_W = occ.addBSplineFilling(wire_northZ)
            wire_westZ = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_W = occ.addBSplineFilling(wire_westZ)
            wire_bottom_new_W = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_new_W = occ.addBSplineFilling(wire_bottom_new_W)
            occ.synchronize()
            sl_minusX_bottom = occ.addSurfaceLoop(
                [
                    s_top_minusX_bottom,
                    s_southZ_W,
                    s_eastZ_W,
                    s_northZ_W,
                    s_westZ_W,
                    s_bottom_new_W,
                ]
            )
            vol_soil_minusX_bottom_ext = occ.addVolume([sl_minusX_bottom])
            occ.synchronize()
            print(
                f"[-X bottom] Created bottom extension under -X extension: vol={vol_soil_minusX_bottom_ext}, padZ={padZ}"
            )

            # +Y BOTTOM EXTENSION (downward by padZ)
            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-12):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError("No flat bottom face found on the volume.")
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_plusY_bottom = s_floor_ext_N
            except NameError:
                try:
                    s_top_plusY_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_plusY, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Provide s_floor_ext_N or ensure vol_soil_plusY exists to auto-detect its bottom face."
                    )
            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_plusY_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)
            _, top_curves = gmsh.model.getAdjacencies(2, s_top_plusY_bottom)
            top_curves = list(dict.fromkeys(top_curves))
            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e
            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify +Y floor face edges (south/east/north/west)."
                )
            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()
            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            _s = _sorted_pts(c_south_old, "x")
            _sn = _sorted_pts(c_south_new, "x")
            pt_SW_old, pt_SE_old = _s[0], _s[-1]
            pt_SW_new, pt_SE_new = _sn[0], _sn[-1]
            _n = _sorted_pts(c_north_old, "x")
            _nn = _sorted_pts(c_north_new, "x")
            pt_NW_old, pt_NE_old = _n[0], _n[-1]
            pt_NW_new, pt_NE_new = _nn[0], _nn[-1]
            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()
            wire_southZ = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_N = occ.addBSplineFilling(wire_southZ)
            wire_eastZ = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_N = occ.addBSplineFilling(wire_eastZ)
            wire_northZ = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_N = occ.addBSplineFilling(wire_northZ)
            wire_westZ = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_N = occ.addBSplineFilling(wire_westZ)
            wire_bottom_new_N = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_new_N = occ.addBSplineFilling(wire_bottom_new_N)
            occ.synchronize()
            sl_plusY_bottom = occ.addSurfaceLoop(
                [
                    s_top_plusY_bottom,
                    s_southZ_N,
                    s_eastZ_N,
                    s_northZ_N,
                    s_westZ_N,
                    s_bottom_new_N,
                ]
            )
            vol_soil_plusY_bottom_ext = occ.addVolume([sl_plusY_bottom])
            occ.synchronize()
            print(
                f"[+Y bottom] Created bottom extension under +Y extension: vol={vol_soil_plusY_bottom_ext}, padZ={padZ}"
            )

            # -Y BOTTOM EXTENSION (downward by padZ)
            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-12):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError("No flat bottom face found on the volume.")
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_minusY_bottom = s_floor_ext_S
            except NameError:
                try:
                    s_top_minusY_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_minusY, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Provide s_floor_ext_S or ensure vol_soil_minusY exists to auto-detect its bottom face."
                    )
            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_minusY_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)
            _, top_curves = gmsh.model.getAdjacencies(2, s_top_minusY_bottom)
            top_curves = list(dict.fromkeys(top_curves))
            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e
            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify -Y floor face edges (south/east/north/west)."
                )
            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()
            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            _s = _sorted_pts(c_south_old, "x")
            _sn = _sorted_pts(c_south_new, "x")
            pt_SW_old, pt_SE_old = _s[0], _s[-1]
            pt_SW_new, pt_SE_new = _sn[0], _sn[-1]
            _n = _sorted_pts(c_north_old, "x")
            _nn = _sorted_pts(c_north_new, "x")
            pt_NW_old, pt_NE_old = _n[0], _n[-1]
            pt_NW_new, pt_NE_new = _nn[0], _nn[-1]
            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()
            wire_southZ = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_S = occ.addBSplineFilling(wire_southZ)
            wire_eastZ = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_S = occ.addBSplineFilling(wire_eastZ)
            wire_northZ = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_S = occ.addBSplineFilling(wire_northZ)
            wire_westZ = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_S = occ.addBSplineFilling(wire_westZ)
            wire_bottom_new_S = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_new_S = occ.addBSplineFilling(wire_bottom_new_S)
            occ.synchronize()
            sl_minusY_bottom = occ.addSurfaceLoop(
                [
                    s_top_minusY_bottom,
                    s_southZ_S,
                    s_eastZ_S,
                    s_northZ_S,
                    s_westZ_S,
                    s_bottom_new_S,
                ]
            )
            vol_soil_minusY_bottom_ext = occ.addVolume([sl_minusY_bottom])
            occ.synchronize()
            print(
                f"[-Y bottom] Created bottom extension under -Y extension: vol={vol_soil_minusY_bottom_ext}, padZ={padZ}"
            )

            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-9):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError(
                        "No flat bottom face found on provided diagonal volume."
                    )
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_NE_diag_bottom = s_bottom_diag_low
            except NameError:
                try:
                    s_top_NE_diag_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_diag_NE, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Provide s_bottom_diag_low or ensure vol_soil_diag_NE exists to auto-detect its bottom face."
                    )
            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_NE_diag_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)
            _, top_curves = gmsh.model.getAdjacencies(2, s_top_NE_diag_bottom)
            top_curves = list(dict.fromkeys(top_curves))
            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e
            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify NE diagonal bottom edges (south/east/north/west)."
                )
            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()
            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            _s = _sorted_pts(c_south_old, "x")
            _sn = _sorted_pts(c_south_new, "x")
            pt_SW_old, pt_SE_old = _s[0], _s[-1]
            pt_SW_new, pt_SE_new = _sn[0], _sn[-1]
            _n = _sorted_pts(c_north_old, "x")
            _nn = _sorted_pts(c_north_new, "x")
            pt_NW_old, pt_NE_old = _n[0], _n[-1]
            pt_NW_new, pt_NE_new = _nn[0], _nn[-1]
            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()
            wire_southZ_NE = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_NE = occ.addBSplineFilling(wire_southZ_NE)
            wire_eastZ_NE = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_NE = occ.addBSplineFilling(wire_eastZ_NE)
            wire_northZ_NE = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_NE = occ.addBSplineFilling(wire_northZ_NE)
            wire_westZ_NE = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_NE = occ.addBSplineFilling(wire_westZ_NE)
            wire_bottom_NE_new = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_NE_new = occ.addBSplineFilling(wire_bottom_NE_new)
            occ.synchronize()
            sl_NE_bottom_diag = occ.addSurfaceLoop(
                [
                    s_top_NE_diag_bottom,
                    s_southZ_NE,
                    s_eastZ_NE,
                    s_northZ_NE,
                    s_westZ_NE,
                    s_bottom_NE_new,
                ]
            )
            vol_soil_NE_bottom_diag_ext = occ.addVolume([sl_NE_bottom_diag])
            occ.synchronize()
            print(
                f"[+X+Y bottom diagonal] Created NE diagonal bottom extension: vol={vol_soil_NE_bottom_diag_ext}, padZ={padZ}"
            )

            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-9):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError(
                        "No flat bottom face found on provided SE diagonal volume."
                    )
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_SE_diag_bottom = s_bottom_diag_low_SE
            except NameError:
                try:
                    s_top_SE_diag_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_diag_SE, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Provide s_bottom_diag_low_SE or ensure vol_soil_diag_SE exists to auto-detect its bottom face."
                    )

            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_SE_diag_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)

            _, top_curves = gmsh.model.getAdjacencies(2, s_top_SE_diag_bottom)
            top_curves = list(dict.fromkeys(top_curves))

            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e

            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify SE diagonal bottom edges (south/east/north/west)."
                )

            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()

            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            _s = _sorted_pts(c_south_old, "x")
            _sn = _sorted_pts(c_south_new, "x")
            pt_SW_old, pt_SE_old = _s[0], _s[-1]
            pt_SW_new, pt_SE_new = _sn[0], _sn[-1]

            _n = _sorted_pts(c_north_old, "x")
            _nn = _sorted_pts(c_north_new, "x")
            pt_NW_old, pt_NE_old = _n[0], _n[-1]
            pt_NW_new, pt_NE_new = _nn[0], _nn[-1]

            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()

            wire_southZ_SE = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_SE = occ.addBSplineFilling(wire_southZ_SE)

            wire_eastZ_SE = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_SE = occ.addBSplineFilling(wire_eastZ_SE)

            wire_northZ_SE = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_SE = occ.addBSplineFilling(wire_northZ_SE)

            wire_westZ_SE = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_SE = occ.addBSplineFilling(wire_westZ_SE)

            wire_bottom_SE_new = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_SE_new = occ.addBSplineFilling(wire_bottom_SE_new)

            occ.synchronize()

            sl_SE_bottom_diag = occ.addSurfaceLoop(
                [
                    s_top_SE_diag_bottom,
                    s_southZ_SE,
                    s_eastZ_SE,
                    s_northZ_SE,
                    s_westZ_SE,
                    s_bottom_SE_new,
                ]
            )
            vol_soil_SE_bottom_diag_ext = occ.addVolume([sl_SE_bottom_diag])
            occ.synchronize()

            print(
                f"[+X-Y bottom diagonal] Created SE diagonal bottom extension: vol={vol_soil_SE_bottom_diag_ext}, padZ={padZ}"
            )

            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-9):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError(
                        "No flat bottom face found on provided SW diagonal volume."
                    )
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_SW_diag_bottom = s_bottom_diag_low_SW
            except NameError:
                try:
                    s_top_SW_diag_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_diag_SW, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Provide s_bottom_diag_low_SW or ensure vol_soil_diag_SW exists to auto-detect its bottom face."
                    )

            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_SW_diag_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)

            _, top_curves = gmsh.model.getAdjacencies(2, s_top_SW_diag_bottom)
            top_curves = list(dict.fromkeys(top_curves))

            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e

            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify SW diagonal bottom edges (south/east/north/west)."
                )

            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()

            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            _s = _sorted_pts(c_south_old, "x")
            _sn = _sorted_pts(c_south_new, "x")
            pt_SW_old, pt_SE_old = _s[0], _s[-1]
            pt_SW_new, pt_SE_new = _sn[0], _sn[-1]

            _n = _sorted_pts(c_north_old, "x")
            _nn = _sorted_pts(c_north_new, "x")
            pt_NW_old, pt_NE_old = _n[0], _n[-1]
            pt_NW_new, pt_NE_new = _nn[0], _nn[-1]

            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()

            wire_southZ_SW = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_SW = occ.addBSplineFilling(wire_southZ_SW)

            wire_eastZ_SW = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_SW = occ.addBSplineFilling(wire_eastZ_SW)

            wire_northZ_SW = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_SW = occ.addBSplineFilling(wire_northZ_SW)

            wire_westZ_SW = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_SW = occ.addBSplineFilling(wire_westZ_SW)

            wire_bottom_SW_new = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_SW_new = occ.addBSplineFilling(wire_bottom_SW_new)

            occ.synchronize()

            sl_SW_bottom_diag = occ.addSurfaceLoop(
                [
                    s_top_SW_diag_bottom,
                    s_southZ_SW,
                    s_eastZ_SW,
                    s_northZ_SW,
                    s_westZ_SW,
                    s_bottom_SW_new,
                ]
            )
            vol_soil_SW_bottom_diag_ext = occ.addVolume([sl_SW_bottom_diag])
            occ.synchronize()

            print(
                f"[-X-Y bottom diagonal] Created SW diagonal bottom extension: vol={vol_soil_SW_bottom_diag_ext}, padZ={padZ}"
            )

            occ = gmsh.model.occ
            occ.synchronize()

            def _pick_flat_bottom_face_from_volume(vol_tag, eps=1e-9):
                _xmin, _ymin, zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(
                    3, vol_tag
                )
                _, faces = gmsh.model.getAdjacencies(3, vol_tag)
                cand = []
                for s in faces:
                    _sx0, _sy0, sz0, _sx1, _sy1, sz1 = gmsh.model.getBoundingBox(2, s)
                    if abs(sz0 - zmin) < eps and abs(sz1 - zmin) < eps:
                        cand.append(s)
                if not cand:
                    raise RuntimeError(
                        "No flat bottom face found on provided NW diagonal volume."
                    )
                cand.sort(
                    key=lambda s: (
                        (
                            gmsh.model.getBoundingBox(2, s)[3]
                            - gmsh.model.getBoundingBox(2, s)[0]
                        )
                        * (
                            gmsh.model.getBoundingBox(2, s)[4]
                            - gmsh.model.getBoundingBox(2, s)[1]
                        )
                    ),
                    reverse=True,
                )
                return cand[0]

            try:
                s_top_NW_diag_bottom = s_bottom_diag_low_NW
            except NameError:
                try:
                    s_top_NW_diag_bottom = _pick_flat_bottom_face_from_volume(
                        vol_soil_diag_NW, eps=1e-9
                    )
                except NameError:
                    raise RuntimeError(
                        "Provide s_bottom_diag_low_NW or ensure vol_soil_diag_NW exists to auto-detect its bottom face."
                    )

            sx0, sy0, _sz0, sx1, sy1, _sz1 = gmsh.model.getBoundingBox(
                2, s_top_NW_diag_bottom
            )
            eps_xy = 1e-9 * max(abs(sx1 - sx0), abs(sy1 - sy0), 1.0)

            _, top_curves = gmsh.model.getAdjacencies(2, s_top_NW_diag_bottom)
            top_curves = list(dict.fromkeys(top_curves))

            c_south_old = c_east_old = c_north_old = c_west_old = None
            for e in top_curves:
                ex0, ey0, _ez0, ex1, ey1, _ez1 = gmsh.model.getBoundingBox(1, e)
                if abs(ey0 - sy0) < eps_xy and abs(ey1 - sy0) < eps_xy:
                    c_south_old = e
                elif abs(ey0 - sy1) < eps_xy and abs(ey1 - sy1) < eps_xy:
                    c_north_old = e
                elif abs(ex0 - sx0) < eps_xy and abs(ex1 - sx0) < eps_xy:
                    c_west_old = e
                elif abs(ex0 - sx1) < eps_xy and abs(ex1 - sx1) < eps_xy:
                    c_east_old = e

            if any(
                v is None for v in (c_south_old, c_east_old, c_north_old, c_west_old)
            ):
                raise RuntimeError(
                    "Failed to classify NW diagonal bottom edges (south/east/north/west)."
                )

            copiesZ = occ.copy(
                [(1, c_south_old), (1, c_east_old), (1, c_north_old), (1, c_west_old)]
            )
            occ.synchronize()
            occ.translate(copiesZ, 0.0, 0.0, -padZ)
            occ.synchronize()

            c_south_new, c_east_new, c_north_new, c_west_new = [
                copiesZ[i][1] for i in range(4)
            ]

            def _sorted_pts(edge, axis):
                _, pts = gmsh.model.getAdjacencies(1, edge)
                pts = list(pts)
                if axis == "x":
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[0])
                else:
                    pts.sort(key=lambda p: gmsh.model.getBoundingBox(0, p)[1])
                return pts

            _s = _sorted_pts(c_south_old, "x")
            _sn = _sorted_pts(c_south_new, "x")
            pt_SW_old, pt_SE_old = _s[0], _s[-1]
            pt_SW_new, pt_SE_new = _sn[0], _sn[-1]

            _n = _sorted_pts(c_north_old, "x")
            _nn = _sorted_pts(c_north_new, "x")
            pt_NW_old, pt_NE_old = _n[0], _n[-1]
            pt_NW_new, pt_NE_new = _nn[0], _nn[-1]

            ln_SW = occ.addLine(pt_SW_old, pt_SW_new)
            ln_SE = occ.addLine(pt_SE_old, pt_SE_new)
            ln_NE = occ.addLine(pt_NE_old, pt_NE_new)
            ln_NW = occ.addLine(pt_NW_old, pt_NW_new)
            occ.synchronize()

            wire_southZ_NW = occ.addWire([c_south_old, ln_SE, -c_south_new, -ln_SW])
            s_southZ_NW = occ.addBSplineFilling(wire_southZ_NW)

            wire_eastZ_NW = occ.addWire([c_east_old, ln_NE, -c_east_new, -ln_SE])
            s_eastZ_NW = occ.addBSplineFilling(wire_eastZ_NW)

            wire_northZ_NW = occ.addWire([c_north_old, ln_NW, -c_north_new, -ln_NE])
            s_northZ_NW = occ.addBSplineFilling(wire_northZ_NW)

            wire_westZ_NW = occ.addWire([c_west_old, ln_SW, -c_west_new, -ln_NW])
            s_westZ_NW = occ.addBSplineFilling(wire_westZ_NW)

            wire_bottom_NW_new = occ.addWire(
                [c_south_new, c_east_new, c_north_new, c_west_new]
            )
            s_bottom_NW_new = occ.addBSplineFilling(wire_bottom_NW_new)

            occ.synchronize()

            sl_NW_bottom_diag = occ.addSurfaceLoop(
                [
                    s_top_NW_diag_bottom,
                    s_southZ_NW,
                    s_eastZ_NW,
                    s_northZ_NW,
                    s_westZ_NW,
                    s_bottom_NW_new,
                ]
            )
            vol_soil_NW_bottom_diag_ext = occ.addVolume([sl_NW_bottom_diag])
            occ.synchronize()

            gmsh.model.occ.removeAllDuplicates()

            occ.synchronize()
            print(
                f"[-X+Y bottom diagonal] Created NW diagonal bottom extension: vol={vol_soil_NW_bottom_diag_ext}, padZ={padZ}"
            )

            print(
                f"[-X+Y bottom diagonal] Created NW diagonal bottom extension: vol={vol_soil_NW_bottom_diag_ext}, padZ={padZ}"
            )

            # Physical groups
            water_vols = [
                vol_water,
                vol_water_plusX,
                vol_water_minusX,
                vol_water_plusY,
                vol_water_minusY,
                vol_water_diag_NE,
                vol_water_diag_NW,
                vol_water_diag_SW,
                vol_water_diag_SE,
            ]
            soil_vols = [
                vol_subsoil,
                vol_soil_plusX,
                vol_soil_minusX,
                vol_soil_plusY,
                vol_soil_minusY,
                vol_soil_diag_NE,
                vol_soil_diag_NW,
                vol_soil_diag_SW,
                vol_soil_diag_SE,
                vol_soil_bottom_ext,
                vol_soil_plusX_bottom_ext,
                vol_soil_minusX_bottom_ext,
                vol_soil_plusY_bottom_ext,
                vol_soil_minusY_bottom_ext,
                vol_soil_NE_bottom_diag_ext,
                vol_soil_SE_bottom_diag_ext,
                vol_soil_SW_bottom_diag_ext,
                vol_soil_NW_bottom_diag_ext,
            ]
            pad_vols = [
                vol_water_plusX,
                vol_water_minusX,
                vol_water_plusY,
                vol_water_minusY,
                vol_water_diag_NE,
                vol_water_diag_NW,
                vol_water_diag_SW,
                vol_water_diag_SE,
                vol_soil_plusX,
                vol_soil_minusX,
                vol_soil_plusY,
                vol_soil_minusY,
                vol_soil_diag_NE,
                vol_soil_diag_NW,
                vol_soil_diag_SW,
                vol_soil_diag_SE,
                vol_soil_bottom_ext,
                vol_soil_plusX_bottom_ext,
                vol_soil_minusX_bottom_ext,
                vol_soil_plusY_bottom_ext,
                vol_soil_minusY_bottom_ext,
                vol_soil_NE_bottom_diag_ext,
                vol_soil_SE_bottom_diag_ext,
                vol_soil_SW_bottom_diag_ext,
                vol_soil_NW_bottom_diag_ext,
            ]

            gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="Subsurface")
            gmsh.model.addPhysicalGroup(3, [vol_water], name="Water")
            gmsh.model.addPhysicalGroup(3, pad_vols, name="Padding")
            gmsh.model.addPhysicalGroup(3, water_vols, name="Water_with_padding")
            gmsh.model.addPhysicalGroup(3, soil_vols, name="Subsurface_with_padding")

    gmsh.model.occ.synchronize()
    return {
        "vol_subsoil": vol_subsoil,
        "vol_water": vol_water,
        "padding_type3D": source_padding_type,
        "water_interface": bool(water_interface),
        "water_delimited": bool(water_interface),
        "contains_water_from_velocity_model": True,
        "structured_mesh_supported": source_padding_type != "elliptical",
        "winslow_supported": source_padding_type != "elliptical",
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
):
    """Install the exact serial callback or parallel PostView sizing field."""
    ef_segy2 = ef_segy3
    domainX = float(length_x)
    domainY = float(length_y)
    domainZ = abs(float(depth_z))
    padX = float(padding_x)
    padY = float(padding_y)
    padZ = float(padding_z)
    ellipse_n = float(hyper_n)
    padding_type = "elliptical" if padding_type == "hyperelliptical" else padding_type

    if structured_mesh is False:
        if parallel is False:
            callback_calls = 0

            def checked_mesh_size(coords, context):
                values = np.asarray(
                    ef_segy2(coords),
                    dtype=float,
                ).reshape(-1)

                if values.size != 1:
                    raise ValueError(
                        "The 3-D mesh sizing callback expected one value, "
                        f"received {values.size} in {context}."
                    )

                value = float(values[0])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        "The 3-D mesh sizing callback produced an invalid "
                        f"element size {value} in {context} at coordinate "
                        f"(z, x, y)={np.asarray(coords)[0].tolist()}. "
                        "Gmsh requires a finite strictly positive size."
                    )
                return value

            def mesh_size_callback3D(dim, tag, x, y, z, lc):
                nonlocal callback_calls
                callback_calls += 1

                if extend_segy:
                    # Edge-extended sizing

                    coords = np.array([[z, x, y]], dtype=float)
                    return checked_mesh_size(
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
                            coords_proj,
                            "projected padding boundary",
                        )

                        dx = abs(x - x_proj)
                        dy = abs(y - y_proj)
                        dz = abs(z - z_proj)

                        tx = dx / padX if padX > 0 else 0.0
                        ty = (
                            dy / padY if padY > 0 else 0.0
                        )
                        tz = dz / padZ if padZ > 0 else 0.0

                        if padding_type == "elliptical":
                            t = (tx**ellipse_n + ty**ellipse_n + tz**ellipse_n) ** (
                                1.0 / ellipse_n
                            )
                        else:
                            t = max(tx, ty, tz)

                        t = min(t, 1.0)  # Clamp between 0.0 and 1.0
                        max_padding_size = h_padding
                        graded_size = base_size + t * (max_padding_size - base_size)

                        return float(graded_size)

            # ########################################
            gmsh.model.mesh.setSizeCallback(mesh_size_callback3D)

        if parallel:
            print("Computing Multi-Resolution Sizing Point Cloud...")

            # --- 1. DENSE CORE POINTS ---

            x_core = np.linspace(bbox[2], bbox[3], nx)
            y_core = np.linspace(bbox[4], bbox[5], ny)
            z_core = np.linspace(bbox[0], bbox[1], nz)

            XX_c, YY_c, ZZ_c = np.meshgrid(x_core, y_core, z_core, indexing="ij")
            coords_c = np.column_stack((ZZ_c.ravel(), XX_c.ravel(), YY_c.ravel()))

            # Core sizing
            size_core = ef_segy2(coords_c).ravel()

            sp_data_core = np.empty((XX_c.size, 4), dtype=np.float64)
            sp_data_core[:, 0] = XX_c.ravel()
            sp_data_core[:, 1] = YY_c.ravel()
            sp_data_core[:, 2] = ZZ_c.ravel()
            sp_data_core[:, 3] = size_core

            del XX_c, YY_c, ZZ_c, coords_c, size_core

            # --- 2. SPARSE PADDING POINTS ---
            xmin_pad = 0.0 - padX
            xmax_pad = domainX + padX
            ymin_pad = 0.0 - padY
            ymax_pad = domainY + padY
            zmin_pad = -domainZ - padZ
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

                # Math for padding expansion
                tx = (
                    np.abs(XX_out - XX_proj) / padX
                    if padX > 0
                    else np.zeros_like(XX_out)
                )
                ty = (
                    np.abs(YY_out - YY_proj) / padY
                    if padY > 0
                    else np.zeros_like(YY_out)
                )
                tz = (
                    np.abs(ZZ_out - ZZ_proj) / padZ
                    if padZ > 0
                    else np.zeros_like(ZZ_out)
                )

                if padding_type == "elliptical":
                    e_n = ellipse_n
                    t = (tx**e_n + ty**e_n + tz**e_n) ** (1.0 / e_n)
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

            # --- 3. COMBINE INTO A SINGLE ARRAY ---

            sp_data_combined = np.vstack((sp_data_core, sp_data_pad))
            total_points = len(sp_data_combined)

            print("Formatting array for C++ memory transfer...")

            sp_list = sp_data_combined.ravel().tolist()

            del sp_data_core, sp_data_pad, sp_data_combined

            print(f"Loading {total_points} total points into single Gmsh PostView...")
            view_tag = gmsh.view.add("background_size")

            # Load the combined cloud as Scalar Points (SP)
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
    parallel_print=print,
):
    """Apply the supplied structured 3-D node constraints and smoothing."""
    if not apply_winslow:
        parallel_print("Skipping 3D Winslow smoothing.", comm=comm)
        return
    del comm, geom_params, hyper_n
    del n_samples, n_traces_x, n_traces_y
    del parallel_print
    structured_mesh = True
    ef_segy2 = ef_segy3
    domainX = float(length_x)
    domainY = float(length_y)
    domainZ = abs(float(depth_z))
    padX = float(padding_x)
    padY = float(padding_y)
    padZ = float(padding_z)
    padding_type = "elliptical" if padding_type == "hyperelliptical" else padding_type

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

    def sizing_function_xyz(X, Y, Z):
        """Evaluate structured Winslow sizing using nearest-edge extension."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=float)

        if X.shape != Y.shape or X.shape != Z.shape:
            raise ValueError(
                "Winslow sizing coordinates X, Y and Z must have matching shapes."
            )

        finite_coordinates = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        if not np.all(finite_coordinates):
            bad_count = int(
                finite_coordinates.size - np.count_nonzero(finite_coordinates)
            )
            raise FloatingPointError(
                "Winslow generated NaN or infinite coordinates before "
                f"sizing evaluation ({bad_count} invalid nodes). Reduce "
                "winslow_omega or inspect the structured topology."
            )

        X_edge = np.clip(X.reshape(-1), 0.0, domainX)
        Y_edge = np.clip(Y.reshape(-1), 0.0, domainY)
        Z_edge = np.clip(Z.reshape(-1), -domainZ, 0.0)

        queries_zxy = np.column_stack(
            (
                Z_edge,
                X_edge,
                Y_edge,
            )
        )

        sizes = np.asarray(
            ef_segy2(queries_zxy),
            dtype=float,
        ).reshape(X.shape)

        if not np.all(np.isfinite(sizes)):
            invalid = np.flatnonzero(~np.isfinite(sizes))
            first = int(invalid[0])
            raise ValueError(
                "The structured edge-extended sizing function returned "
                "NaN or infinity. First projected query "
                f"(z, x, y)={queries_zxy[first].tolist()}. Check the "
                "velocity binary metadata and sizing-function construction."
            )

        if np.any(sizes <= 0.0):
            minimum = float(np.min(sizes))
            raise ValueError(
                "The structured Winslow sizing function must be positive; "
                f"minimum value is {minimum}."
            )

        return sizes

    def run_selected_winslow(
        *,
        points,
        hexes,
        segy_grid_x,
        segy_grid_y,
        segy_grid_z,
        segy_grid_vals,
        move_all,
        move_X_only,
        move_Y_only,
        move_Z_only,
    ):
        """Run the selected smoother and verify actual node displacement."""
        movable_nodes = (
            set(move_all) | set(move_X_only) | set(move_Y_only) | set(move_Z_only)
        )

        if not movable_nodes:
            raise RuntimeError(
                "Winslow smoothing has no movable nodes. Check the Water "
                "physical group and rectangular boundary classification."
            )

        movable_indices = np.asarray(
            sorted(movable_nodes),
            dtype=np.int64,
        )

        initial_sizes = sizing_function_xyz(
            points[:, 0],
            points[:, 1],
            points[:, 2],
        )
        movable_sizes = initial_sizes[movable_indices]

        size_min = float(np.min(movable_sizes))
        size_max = float(np.max(movable_sizes))
        size_mean = float(np.mean(movable_sizes))
        size_std = float(np.std(movable_sizes))
        size_span = size_max - size_min

        print(
            "Winslow sizing on movable nodes | "
            f"min={size_min:.9g}, max={size_max:.9g}, "
            f"mean={size_mean:.9g}, std={size_std:.9g}"
        )
        print(
            "Winslow movable-node counts | "
            f"all={len(move_all)}, X={len(move_X_only)}, "
            f"Y={len(move_Y_only)}, Z={len(move_Z_only)}, "
            f"union={len(movable_nodes)}"
        )

        relative_span = size_span / max(abs(size_mean), 1.0)
        if relative_span <= 1.0e-10:
            print(
                "WARNING: the sizing field is effectively constant on the "
                "movable nodes. A uniform structured mesh will remain "
                "unchanged. Reduce hmin_segy; the supplied reference uses "
                "a 250 m initial mesh and no 500 m sizing floor."
            )

        movement_arguments = {
            "points": points,
            "hexes": hexes,
            "move_all": move_all,
            "move_X_only": move_X_only,
            "move_Y_only": move_Y_only,
            "move_Z_only": move_Z_only,
            "iterations": winslow_iterations,
            "omega": winslow_omega,
        }

        if selected_winslow == "numba":
            print("Using winslow_smooth_3d55 (Numba 3D Winslow implementation).")
            smoothed = winslow_smooth_3d55(
                sizing_fn=sizing_function_xyz,
                **movement_arguments,
            )

        displacement = np.linalg.norm(
            np.asarray(smoothed, dtype=float) - np.asarray(points, dtype=float),
            axis=1,
        )
        movable_displacement = displacement[movable_indices]

        moved_count = int(np.count_nonzero(movable_displacement > 1.0e-10))
        max_displacement = float(np.max(movable_displacement))
        mean_displacement = float(np.mean(movable_displacement))

        print(
            "Winslow displacement | "
            f"moved={moved_count}/{len(movable_indices)}, "
            f"max={max_displacement:.9g}, "
            f"mean={mean_displacement:.9g}"
        )

        if moved_count == 0:
            print(
                "WARNING: winslow_smooth_3d55 completed but no movable node "
                "changed position. Inspect the sizing statistics above."
            )

        return smoothed

    if not water_interface:  # noqa: SIM102
        if structured_mesh:
            if padding_type is None:
                print(
                    "Extracting nodes for 3D smoothing (No Water Interface, No Padding)..."
                )
                # Extract nodes
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)

                # Extract hexahedra
                (
                    elem_types,
                    _elem_tags,
                    elem_node_tags,
                ) = gmsh.model.mesh.getElements(dim=3)

                all_hex_nodes = []
                for i, t_type in enumerate(elem_types):
                    if t_type == 5:  # Type 5 = 8-node Hexahedron in Gmsh
                        all_hex_nodes.extend(elem_node_tags[i])

                if not all_hex_nodes:
                    raise ValueError(
                        "No hexahedrons found! 3D Winslow requires a hexahedral mesh."
                    )

                # Map node tags
                tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
                hexes = np.asarray(
                    [tag_to_index[tag] for tag in all_hex_nodes], dtype=np.int64
                ).reshape(-1, 8)

                locked = set()
                move_X_only = set()
                move_Y_only = set()
                move_Z_only = set()
                move_all = set()

                # 1. No Water Points to Extract
                water_nodes = set()

                # 2. Categorize remaining points
                tol = 2.0

                for i, pt in enumerate(points_3d):
                    x, y, z = pt

                    on_xmin = abs(x) < tol
                    on_xmax = abs(x - domainX) < tol
                    on_ymin = abs(y) < tol
                    on_ymax = abs(y - domainY) < tol
                    on_zmin = abs(z - (-domainZ)) < tol
                    on_zmax = abs(z) < tol  # Zmax is 0.0

                    on_x_plane = on_xmin or on_xmax
                    on_y_plane = on_ymin or on_ymax
                    on_z_plane = on_zmin or on_zmax

                    # Plane count
                    contact_count = sum([on_x_plane, on_y_plane, on_z_plane])

                    # Corners
                    if contact_count == 3:
                        locked.add(i)
                        continue

                    # Edges
                    if contact_count == 2:
                        if on_x_plane and on_y_plane:
                            move_Z_only.add(i)  # Vertical pillars
                        elif on_x_plane and on_z_plane:
                            move_Y_only.add(i)  # Bottom/Top edges along Y
                        elif on_y_plane and on_z_plane:
                            move_X_only.add(i)  # Bottom/Top edges along X
                        continue

                    if contact_count == 1:
                        if on_x_plane:
                            move_Y_only.add(i)
                            move_Z_only.add(i)
                        elif on_y_plane:
                            move_X_only.add(i)
                            move_Z_only.add(i)
                        elif on_z_plane:
                            move_X_only.add(i)
                            move_Y_only.add(i)
                        continue

                    # Interior nodes
                    move_all.add(i)

                print(f"Nodes Breakdown | Total: {len(points_3d)}")
                print(
                    f"Locked (Corners): {len(locked)} | Interior (Move All): {len(move_all)}"
                )
                print("Preparing 3D sizing function grid for Numba...")

                # Grid resolution

                nx_grid, ny_grid, nz_grid = 201, 201, 401

                segy_grid_x = np.linspace(0.0, domainX, nx_grid)
                segy_grid_y = np.linspace(0.0, domainY, ny_grid)
                segy_grid_z = np.linspace(0.0, -domainZ, nz_grid)

                X_grid, Y_grid, Z_grid = np.meshgrid(
                    segy_grid_x, segy_grid_y, segy_grid_z, indexing="ij"
                )

                pts_for_eval = np.column_stack(
                    (Z_grid.flatten(), X_grid.flatten(), Y_grid.flatten())
                )
                sizes_flat = ef_segy2(pts_for_eval)

                # Reshape sizing values
                segy_grid_vals = sizes_flat.reshape((nx_grid, ny_grid, nz_grid))

                # Winslow smoothing
                print("Applying 3D Winslow smoothing...")
                print("Applying 3D Winslow smoothing to padded domain...")
                smoothed_points_3d = run_selected_winslow(
                    points=points_3d,
                    hexes=hexes,
                    segy_grid_x=segy_grid_x,
                    segy_grid_y=segy_grid_y,
                    segy_grid_z=segy_grid_z,
                    segy_grid_vals=segy_grid_vals,
                    move_all=move_all,
                    move_X_only=move_X_only,
                    move_Y_only=move_Y_only,
                    move_Z_only=move_Z_only,
                )

                print("Updating nodes back into Gmsh...")
                # Update nodes
                movable_nodes = move_all | move_X_only | move_Y_only | move_Z_only
                for i, tag in enumerate(node_tags):
                    if i in movable_nodes:
                        gmsh.model.mesh.setNode(
                            int(tag), smoothed_points_3d[i].tolist(), []
                        )

            elif padding_type == "rectangular":
                print(
                    "Extracting nodes for 3D smoothing "
                    "(No Water Interface, Rectangular Padding)..."
                )

                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                points_3d = np.asarray(
                    node_coords,
                    dtype=float,
                ).reshape(-1, 3)

                # Extract 8-node hexahedra. Winslow operates on the

                (
                    elem_types,
                    _elem_tags,
                    elem_node_tags,
                ) = gmsh.model.mesh.getElements(dim=3)

                all_hex_nodes = []
                for element_index, element_type in enumerate(elem_types):
                    if element_type == 5:
                        all_hex_nodes.extend(
                            elem_node_tags[element_index]
                        )

                if not all_hex_nodes:
                    raise ValueError(
                        "No hexahedra found. 3-D Winslow smoothing "
                        "requires an 8-node hexahedral mesh."
                    )

                tag_to_index = {
                    tag: index
                    for index, tag in enumerate(node_tags)
                }
                hexes = np.asarray(
                    [
                        tag_to_index[tag]
                        for tag in all_hex_nodes
                    ],
                    dtype=np.int64,
                ).reshape(-1, 8)

                locked = set()
                move_X_only = set()
                move_Y_only = set()
                move_Z_only = set()
                move_all = set()

                xmin_pad = -padX
                xmax_pad = domainX + padX
                ymin_pad = -padY
                ymax_pad = domainY + padY
                zmin_pad = -domainZ - padZ
                zmax_pad = 0.0

                # water volume.
                tol = 2.0

                for node_index, point in enumerate(points_3d):
                    x_coord, y_coord, z_coord = point

                    on_x_plane = (
                        abs(x_coord - xmin_pad) < tol
                        or abs(x_coord - xmax_pad) < tol
                        or abs(x_coord) < tol
                        or abs(x_coord - domainX) < tol
                    )
                    on_y_plane = (
                        abs(y_coord - ymin_pad) < tol
                        or abs(y_coord - ymax_pad) < tol
                        or abs(y_coord) < tol
                        or abs(y_coord - domainY) < tol
                    )
                    on_z_plane = (
                        abs(z_coord - zmin_pad) < tol
                        or abs(z_coord + domainZ) < tol
                        or abs(z_coord - zmax_pad) < tol
                    )

                    contact_count = sum(
                        (
                            on_x_plane,
                            on_y_plane,
                            on_z_plane,
                        )
                    )

                    # Corners are fixed.
                    if contact_count == 3:
                        locked.add(node_index)
                        continue

                    # Edge constraints
                    if contact_count == 2:
                        if on_x_plane and on_y_plane:
                            move_Z_only.add(node_index)
                        elif on_x_plane and on_z_plane:
                            move_Y_only.add(node_index)
                        elif on_y_plane and on_z_plane:
                            move_X_only.add(node_index)
                        continue

                    # combines those masks correctly.
                    if contact_count == 1:
                        if on_x_plane:
                            move_Y_only.add(node_index)
                            move_Z_only.add(node_index)
                        elif on_y_plane:
                            move_X_only.add(node_index)
                            move_Z_only.add(node_index)
                        elif on_z_plane:
                            move_X_only.add(node_index)
                            move_Y_only.add(node_index)
                        continue

                    move_all.add(node_index)

                movable_nodes = (
                    move_all
                    | move_X_only
                    | move_Y_only
                    | move_Z_only
                )

                print(
                    "Winslow rectangular no-water topology | "
                    f"nodes={len(points_3d)}, "
                    f"hexes={len(hexes)}, "
                    f"locked={len(locked)}, "
                    f"movable={len(movable_nodes)}"
                )

                if not movable_nodes:
                    raise RuntimeError(
                        "No movable nodes were found for the no-water "
                        "rectangular Winslow branch."
                    )

                smoothed_points_3d = run_selected_winslow(
                    points=points_3d,
                    hexes=hexes,
                    segy_grid_x=None,
                    segy_grid_y=None,
                    segy_grid_z=None,
                    segy_grid_vals=None,
                    move_all=move_all,
                    move_X_only=move_X_only,
                    move_Y_only=move_Y_only,
                    move_Z_only=move_Z_only,
                )

                print("Updating Winslow-smoothed nodes back into Gmsh...")
                for node_index, node_tag in enumerate(node_tags):
                    if node_index in movable_nodes:
                        gmsh.model.mesh.setNode(
                            int(node_tag),
                            smoothed_points_3d[node_index].tolist(),
                            [],
                        )

    if water_interface:  # noqa: SIM102
        if structured_mesh:
            if padding_type is None:
                print("Extracting nodes for 3D smoothing...")
                # Extract nodes
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)

                # Extract hexahedra
                (
                    elem_types,
                    _elem_tags,
                    elem_node_tags,
                ) = gmsh.model.mesh.getElements(dim=3)

                all_hex_nodes = []
                for i, t_type in enumerate(elem_types):
                    if t_type == 5:  # Type 5 = 8-node Hexahedron in Gmsh
                        all_hex_nodes.extend(elem_node_tags[i])

                if not all_hex_nodes:
                    raise ValueError(
                        "No hexahedrons found! 3D Winslow requires a hexahedral mesh."
                    )

                # Map node tags
                tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
                hexes = np.asarray(
                    [tag_to_index[tag] for tag in all_hex_nodes], dtype=np.int64
                ).reshape(-1, 8)

                locked = set()
                move_X_only = set()
                move_Y_only = set()
                move_Z_only = set()
                move_all = set()

                # 1. Extract Water Points via Physical Group
                water_nodes = set()
                for dim, tag in gmsh.model.getPhysicalGroups(dim=3):
                    if gmsh.model.getPhysicalName(dim, tag) == "Water":
                        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                        for e in entities:

                            w_node_tags, _, _ = gmsh.model.mesh.getNodes(
                                dim, e, includeBoundary=True
                            )
                            water_nodes.update([tag_to_index[t] for t in w_node_tags])

                # 2. Categorize remaining points
                tol = 2.0

                for i, pt in enumerate(points_3d):

                    if i in water_nodes:
                        locked.add(i)
                        continue

                    x, y, z = pt
                    on_xmin = abs(x) < tol
                    on_xmax = abs(x - domainX) < tol
                    on_ymin = abs(y) < tol
                    on_ymax = abs(y - domainY) < tol
                    on_zmin = abs(z - (-domainZ)) < tol

                    on_x_plane = on_xmin or on_xmax
                    on_y_plane = on_ymin or on_ymax
                    on_z_plane = on_zmin

                    # Plane count
                    contact_count = sum([on_x_plane, on_y_plane, on_z_plane])

                    # Corners
                    if contact_count == 3:
                        locked.add(i)
                        continue

                    # Edges
                    if contact_count == 2:
                        if on_x_plane and on_y_plane:
                            move_Z_only.add(i)  # Vertical pillars
                        elif on_x_plane and on_z_plane:
                            move_Y_only.add(i)  # Bottom edges along Y
                        elif on_y_plane and on_z_plane:
                            move_X_only.add(i)  # Bottom edges along X
                        continue

                    if contact_count == 1:
                        if on_x_plane:
                            move_Y_only.add(i)
                            move_Z_only.add(i)
                        elif on_y_plane:
                            move_X_only.add(i)
                            move_Z_only.add(i)
                        elif on_z_plane:
                            move_X_only.add(i)
                            move_Y_only.add(i)
                        continue

                    # Interior nodes
                    move_all.add(i)

                print(f"Nodes Breakdown | Total: {len(points_3d)}")
                print(
                    f"Locked (Water + Corners): {len(locked)} | Interior (Move All): {len(move_all)}"
                )
                print("Preparing 3D sizing function grid for Numba...")

                # Grid resolution

                nx_grid, ny_grid, nz_grid = 201, 201, 401

                # Zmax is typically 0.0 (water surface)
                zmax_pad = 0.0

                segy_grid_x = np.linspace(0.0, domainX, nx_grid)
                segy_grid_y = np.linspace(0.0, domainY, ny_grid)
                segy_grid_z = np.linspace(0.0, -domainZ, nz_grid)

                X_grid, Y_grid, Z_grid = np.meshgrid(
                    segy_grid_x, segy_grid_y, segy_grid_z, indexing="ij"
                )

                pts_for_eval = np.column_stack(
                    (Z_grid.flatten(), X_grid.flatten(), Y_grid.flatten())
                )
                sizes_flat = ef_segy2(pts_for_eval)

                # Reshape sizing values
                segy_grid_vals = sizes_flat.reshape((nx_grid, ny_grid, nz_grid))

                # Winslow smoothing
                print("Applying 3D Winslow smoothing to padded domain...")
                smoothed_points_3d = run_selected_winslow(
                    points=points_3d,
                    hexes=hexes,
                    segy_grid_x=segy_grid_x,
                    segy_grid_y=segy_grid_y,
                    segy_grid_z=segy_grid_z,
                    segy_grid_vals=segy_grid_vals,
                    move_all=move_all,
                    move_X_only=move_X_only,
                    move_Y_only=move_Y_only,
                    move_Z_only=move_Z_only,
                )
                # Winslow smoothing
                print("Updating nodes back into Gmsh...")
                # Update nodes
                movable_nodes = move_all | move_X_only | move_Y_only | move_Z_only
                for i, tag in enumerate(node_tags):
                    if i in movable_nodes:
                        gmsh.model.mesh.setNode(
                            int(tag), smoothed_points_3d[i].tolist(), []
                        )

            elif padding_type == "rectangular":
                print("Extracting nodes for 3D smoothing (Rectangular Padding)...")
                # Extract nodes
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)

                # Extract hexahedra
                elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
                    dim=3
                )

                all_hex_nodes = []
                for i, t_type in enumerate(elem_types):
                    if t_type == 5:  # Type 5 = 8-node Hexahedron in Gmsh
                        all_hex_nodes.extend(elem_node_tags[i])

                if not all_hex_nodes:
                    raise ValueError(
                        "No hexahedrons found! 3D Winslow requires a hexahedral mesh."
                    )

                # Map node tags
                tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
                hexes = np.asarray(
                    [tag_to_index[tag] for tag in all_hex_nodes], dtype=np.int64
                ).reshape(-1, 8)

                locked = set()
                move_X_only = set()
                move_Y_only = set()
                move_Z_only = set()
                move_all = set()

                # 1. Extract Water Points via Physical Group
                water_nodes = set()
                for dim, tag in gmsh.model.getPhysicalGroups(dim=3):
                    if gmsh.model.getPhysicalName(dim, tag) == "Water_with_padding":
                        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                        for e in entities:

                            w_node_tags, _, _ = gmsh.model.mesh.getNodes(
                                dim, e, includeBoundary=True
                            )
                            water_nodes.update([tag_to_index[t] for t in w_node_tags])
                print(
                    "Water_with_padding node count: "
                    f"{len(water_nodes)}/{len(points_3d)}"
                )
                if len(water_nodes) == len(points_3d):
                    raise RuntimeError(
                        "Every mesh node was classified as Water_with_padding. "
                        "No subsurface nodes remain available for Winslow "
                        "smoothing."
                    )

                # 2. Categorize remaining points
                tol = 2.0

                # Padded bounds
                xmin_pad = -padX
                xmax_pad = domainX + padX
                ymin_pad = -padY
                ymax_pad = domainY + padY
                zmin_pad = -domainZ - padZ

                for i, pt in enumerate(points_3d):

                    if i in water_nodes:
                        locked.add(i)
                        continue

                    x, y, z = pt
                    on_x_plane = (
                        (abs(x - xmin_pad) < tol)
                        or (abs(x - xmax_pad) < tol)
                        or (abs(x) < tol)
                        or (abs(x - domainX) < tol)
                    )

                    on_y_plane = (
                        (abs(y - ymin_pad) < tol)
                        or (abs(y - ymax_pad) < tol)
                        or (abs(y) < tol)
                        or (abs(y - domainY) < tol)
                    )

                    on_z_plane = (abs(z - zmin_pad) < tol) or (
                        abs(z - (-domainZ)) < tol
                    )

                    # Plane count
                    contact_count = sum([on_x_plane, on_y_plane, on_z_plane])

                    # Corners
                    if contact_count == 3:
                        locked.add(i)
                        continue

                    if contact_count == 2:
                        if on_x_plane and on_y_plane:
                            move_Z_only.add(i)  # Vertical pillars
                        elif on_x_plane and on_z_plane:
                            move_Y_only.add(i)  # Bottom edges along Y
                        elif on_y_plane and on_z_plane:
                            move_X_only.add(i)  # Bottom edges along X
                        continue

                    if contact_count == 1:
                        if on_x_plane:
                            move_Y_only.add(i)
                            move_Z_only.add(i)
                        elif on_y_plane:
                            move_X_only.add(i)
                            move_Z_only.add(i)
                        elif on_z_plane:
                            move_X_only.add(i)
                            move_Y_only.add(i)
                        continue

                    move_all.add(i)

                print(f"Nodes Breakdown | Total: {len(points_3d)}")
                print(
                    f"Locked (Water + Corners): {len(locked)} | Interior (Move All): {len(move_all)}"
                )

                # Winslow smoothing
                print("Applying 3D Winslow smoothing to padded domain...")
                print("Preparing 3D sizing function grid for Numba...")

                # Grid resolution

                nx_grid, ny_grid, nz_grid = 201, 201, 401

                zmax_pad = 0.0

                segy_grid_x = np.linspace(xmin_pad, xmax_pad, nx_grid)
                segy_grid_y = np.linspace(ymin_pad, ymax_pad, ny_grid)
                segy_grid_z = np.linspace(zmin_pad, zmax_pad, nz_grid)
                X_grid, Y_grid, Z_grid = np.meshgrid(
                    segy_grid_x, segy_grid_y, segy_grid_z, indexing="ij"
                )

                pts_for_eval = np.column_stack(
                    (Z_grid.flatten(), X_grid.flatten(), Y_grid.flatten())
                )
                sizes_flat = ef_segy2(pts_for_eval)

                # Reshape sizing values
                segy_grid_vals = sizes_flat.reshape((nx_grid, ny_grid, nz_grid))

                # Winslow smoothing
                print("Applying 3D Winslow smoothing to padded domain...")
                smoothed_points_3d = run_selected_winslow(
                    points=points_3d,
                    hexes=hexes,
                    segy_grid_x=segy_grid_x,
                    segy_grid_y=segy_grid_y,
                    segy_grid_z=segy_grid_z,
                    segy_grid_vals=segy_grid_vals,
                    move_all=move_all,
                    move_X_only=move_X_only,
                    move_Y_only=move_Y_only,
                    move_Z_only=move_Z_only,
                )

                print("Updating nodes back into Gmsh...")
                # Update nodes
                movable_nodes = move_all | move_X_only | move_Y_only | move_Z_only
                for i, tag in enumerate(node_tags):
                    if i in movable_nodes:
                        gmsh.model.mesh.setNode(
                            int(tag), smoothed_points_3d[i].tolist(), []
                        )
