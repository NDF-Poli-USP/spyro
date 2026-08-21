from pathlib import Path

import numpy as np


def boundary_faces_of_volume(gmsh, vol_tag):
    """Return the unique surface tags bounding a Gmsh volume.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    vol_tag : int
        Tag of the three-dimensional Gmsh volume.

    Returns
    -------
    list of int
        Two-dimensional surface tags bounding the volume.
    """
    boundary = gmsh.model.getBoundary(
        [(3, int(vol_tag))],
        oriented=False,
        recursive=False,
        combined=True,
    )
    return [tag for dim, tag in boundary if dim == 2]


def checked_mesh_size(ef_segy, coords, context):
    """Evaluate and validate a three-dimensional mesh-size query.

    Parameters
    ----------
    ef_segy : callable
        Mesh-sizing function evaluated in ``(z, x, y)`` coordinates.
    coords : numpy.ndarray
        Coordinates passed to the sizing callback.
    context : str
        Description of the sizing-query context used in error messages.

    Returns
    -------
    float
        Validated positive mesh size.
    """
    values = np.asarray(ef_segy(coords), dtype=float).reshape(-1)

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


def _signed_power(x, p):
    """Evaluate a sign power used by the hyperellipsoid parameterization.

    Parameters
    ----------
    x : float
        Value whose signed power is evaluated.
    p : float
        Exponent applied while preserving the sign of the input.

    Returns
    -------
    float or numpy.ndarray
        Sign-preserving powered value.
    """
    if abs(x) < 1.0e-10:
        return 0.0
    return np.sign(x) * np.abs(x) ** p


def hyperellipsoid_point(u, v, a, b, c, n, xc, yc, zc):
    """Evaluate one Cartesian point on a centered hyperellipsoid.

    Parameters
    ----------
    u : float
        Azimuthal hyperellipsoid parameter.
    v : float
        Polar hyperellipsoid parameter.
    a : float
        Hyperellipsoid semi-axis in the x direction.
    b : float
        Hyperellipsoid semi-axis in the y direction.
    c : float
        Hyperellipsoid semi-axis in the z direction.
    n : float
        Hyperellipsoid shape exponent.
    xc : float
        Hyperellipsoid center coordinate in x.
    yc : float
        Hyperellipsoid center coordinate in y.
    zc : float
        Hyperellipsoid center coordinate in z.

    Returns
    -------
    tuple of float
        Cartesian ``(x, y, z)`` point on the hyperellipsoid.
    """
    cv, sv = np.cos(v), np.sin(v)
    cu, su = np.cos(u), np.sin(u)
    p = 2.0 / n

    return (
        xc + a * _signed_power(cv, p) * _signed_power(cu, p),
        yc + b * _signed_power(cv, p) * _signed_power(su, p),
        zc + c * _signed_power(sv, p),
    )


def create_closed_surface(
    gmsh,
    a,
    b,
    c,
    n,
    xc,
    yc,
    zc,
    point_func=hyperellipsoid_point,
    u_res=60,
    v_res=60,
    z_cut=0.0,
    *,
    comm,
    parallel_print,
):
    """Create a closed B-spline hyperellipsoid volume and optionally cut it at a z plane.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    a : float
        Hyperellipsoid semi-axis in the x direction.
    b : float
        Hyperellipsoid semi-axis in the y direction.
    c : float
        Hyperellipsoid semi-axis in the z direction.
    n : float
        Hyperellipsoid shape exponent.
    xc : float
        Hyperellipsoid center coordinate in x.
    yc : float
        Hyperellipsoid center coordinate in y.
    zc : float
        Hyperellipsoid center coordinate in z.
    point_func : callable
        Function used to evaluate hyperellipsoid surface points.
    u_res : int
        Number of azimuthal sampling intervals used by the B-spline surface.
    v_res : int
        Number of polar sampling points used by the B-spline surface.
    z_cut : float or None
        Z coordinate above which the generated volume is cut.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    int or None
        Generated Gmsh volume tag, or ``None`` if the z cut removes the volume.
    """
    occ = gmsh.model.occ
    point_tags = []

    for j in range(v_res):
        v = np.pi * (j / (v_res - 1) - 0.5)
        for i in range(u_res + 1):
            u = 0.0 if i == u_res else 2.0 * np.pi * i / u_res

            if j == 0:
                xyz = point_func(0.0, -np.pi / 2.0, a, b, c, n, xc, yc, zc)
            elif j == v_res - 1:
                xyz = point_func(0.0, np.pi / 2.0, a, b, c, n, xc, yc, zc)
            else:
                xyz = point_func(u, v, a, b, c, n, xc, yc, zc)

            point_tags.append(occ.addPoint(*xyz))

    try:
        surface_tag = occ.addBSplineSurface(
            pointTags=point_tags,
            numPointsU=u_res + 1,
            tag=-1,
            degreeU=min(3, u_res),
            degreeV=min(3, v_res - 1),
        )
        occ.synchronize()

        volume_tag = occ.addVolume([occ.addSurfaceLoop([surface_tag])])
        occ.synchronize()

        if volume_tag is not None and z_cut is not None:
            parallel_print(f"Applying z-cut at z = {z_cut}", comm=comm)
            domain_size = 2.0 * max(a, b, c)
            cutting_box = occ.addBox(
                xc - domain_size,
                yc - domain_size,
                z_cut,
                2.0 * domain_size,
                2.0 * domain_size,
                domain_size,
            )
            occ.synchronize()

            result = occ.cut(
                [(3, volume_tag)],
                [(3, cutting_box)],
                removeObject=True,
                removeTool=True,
            )

            if result[0]:
                volume_tag = result[0][0][1]
                parallel_print("Z-cut applied successfully", comm=comm)
            else:
                parallel_print("Warning: Z-cut removed entire volume", comm=comm)
                volume_tag = None

        return volume_tag

    except Exception as exc:
        raise RuntimeError("B-spline surface creation failed.") from exc


def create_hyperellipsoid_volume(
    gmsh,
    a=1.0,
    b=1.0,
    c=1.0,
    n=2.0,
    xc=0.0,
    yc=0.0,
    zc=0.0,
    *,
    comm,
    parallel_print,
):
    """Create a hyperellipsoid volume for three-dimensional padding.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    a : float
        Hyperellipsoid semi-axis in the x direction.
    b : float
        Hyperellipsoid semi-axis in the y direction.
    c : float
        Hyperellipsoid semi-axis in the z direction.
    n : float
        Hyperellipsoid shape exponent.
    xc : float
        Hyperellipsoid center coordinate in x.
    yc : float
        Hyperellipsoid center coordinate in y.
    zc : float
        Hyperellipsoid center coordinate in z.
    comm : mpi4py.MPI.Comm or None
        MPI communicator.
    parallel_print : callable
        Print function.

    Returns
    -------
    int or None
        Generated hyperellipsoid volume tag.
    """
    try:
        volume_tag = create_closed_surface(
            gmsh,
            a,
            b,
            c,
            n,
            xc,
            yc,
            zc,
            comm=comm,
            parallel_print=parallel_print,
        )
        if volume_tag:
            parallel_print("Successfully created volume using closed B-spline surface", comm=comm)
        return volume_tag
    except Exception as exc:
        raise RuntimeError("B-spline surface creation failed.") from exc


def _add_padding_box(occ, x0, y0, z0, lx, ly, lz):
    """Create a non-degenerate OpenCASCADE rectangular padding block.

    Parameters
    ----------
    occ : gmsh.model.occ
        Gmsh OpenCASCADE geometry interface.
    x0 : float
        Lower x coordinate of the box.
    y0 : float
        Lower y coordinate of the box.
    z0 : float
        Lower z coordinate of the box.
    lx : float
        Box extent in the x direction.
    ly : float
        Box extent in the y direction.
    lz : float
        Box extent in the z direction.

    Returns
    -------
    int or None
        Created volume tag, or ``None`` for a zero-thickness block.
    """
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        return None
    return occ.addBox(
        float(x0), float(y0), float(z0), float(lx), float(ly), float(lz)
    )


def create_rectangular_padding_boxes(
    gmsh, length_x, length_y, depth_z, padding_x, padding_y, padding_z
):
    """Create the central domain and rectangular side and bottom padding blocks.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.

    Returns
    -------
    tuple
        Central subsurface volume tag and list of padding-volume tags.
    """
    occ = gmsh.model.occ

    vol_subsoil = occ.addBox(
        0.0, 0.0, -abs(depth_z), length_x, length_y, abs(depth_z)
    )

    x_cells = [
        (-padding_x, padding_x),
        (0.0, length_x),
        (length_x, padding_x),
    ]
    y_cells = [
        (-padding_y, padding_y),
        (0.0, length_y),
        (length_y, padding_y),
    ]

    side_padding = [
        _add_padding_box(occ, x0, y0, -abs(depth_z), lx, ly, abs(depth_z))
        for ix, (x0, lx) in enumerate(x_cells)
        for iy, (y0, ly) in enumerate(y_cells)
        if (ix, iy) != (1, 1)
    ]

    bottom_z = -abs(depth_z) - padding_z
    bottom_padding = [
        _add_padding_box(occ, x0, y0, bottom_z, lx, ly, padding_z)
        for x0, lx in x_cells
        for y0, ly in y_cells
    ]

    pad_vols = [v for v in side_padding + bottom_padding if v is not None]
    return vol_subsoil, pad_vols


def generate_rectangular_padding_no_water(
    gmsh, length_x, length_y, depth_z, padding_x, padding_y, padding_z
):
    """Build the undelimited rectangular padded domain and its physical groups.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.

    Returns
    -------
    tuple
        Central subsurface volume tag and ``None`` for the water volume.
    """
    if min(length_x, length_y, abs(depth_z)) <= 0.0:
        raise ValueError("length_x, length_y, and depth_z must be positive.")
    if min(padding_x, padding_y, padding_z) < 0.0:
        raise ValueError("padding_x, padding_y, and padding_z cannot be negative.")

    occ = gmsh.model.occ
    vol_subsoil, pad_vols = create_rectangular_padding_boxes(
        gmsh, length_x, length_y, depth_z, padding_x, padding_y, padding_z
    )

    if pad_vols:
        _, fragment_map = occ.fragment(
            [(3, tag) for tag in [vol_subsoil] + pad_vols],
            [],
            removeObject=True,
            removeTool=False,
        )
        occ.synchronize()

        core_fragments = sorted({tag for dim, tag in fragment_map[0] if dim == 3})
        padding_fragments = sorted(
            {
                tag
                for mapping in fragment_map[1:]
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

    gmsh.model.addPhysicalGroup(3, core_fragments, name="SubsurfaceAndWater")
    if pad_vols:
        gmsh.model.addPhysicalGroup(3, pad_vols, name="Padding")

    return vol_subsoil, None


def _read_velocity_cube(fname, nz, nx, ny, dtype_string, order):
    """Memory-map a binary velocity model as a ``(z, x, y)`` cube.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the binary velocity model.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    dtype_string : str
        NumPy dtype string including the required byte order.
    order : {"C", "F"}
        Array reshape order used for the binary velocity model.

    Returns
    -------
    tuple
        Memory-mapped velocity cube and validated ``nz``, ``nx``, and ``ny`` sizes.
    """
    try:
        nz, nx, ny = int(nz), int(nx), int(ny)
    except Exception as exc:
        raise RuntimeError("nz, nx, and ny must be valid integer arguments.") from exc

    path = Path(fname)
    if not path.exists():
        raise FileNotFoundError(f"Binary not found: {path}")

    mm = np.memmap(path, dtype=np.dtype(dtype_string), mode="r")
    expected = nz * nx * ny
    if mm.size != expected:
        raise ValueError(
            f"File size mismatch for {path}:\n"
            f"  got {mm.size} floats, expected {expected} for "
            f"shape (nz,nx,ny)=({nz},{nx},{ny})"
        )

    return mm.reshape((nz, nx, ny), order=order), nz, nx, ny


def _water_interface_grid(
    cube_zxy,
    nz,
    nx,
    ny,
    dz,
    dx,
    dy,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    target,
    tolerance=1.0,
    x_chunk=64,
    y_chunk=None,
):
    """Reconstruct the water/subsurface interface from the velocity cube.

    Parameters
    ----------
    cube_zxy : numpy.ndarray
        Velocity cube ordered as ``(z, x, y)``.
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
    x_min : float
        Minimum x coordinate of the requested region.
    x_max : float
        Maximum x coordinate of the requested region.
    y_min : float
        Minimum y coordinate of the requested region.
    y_max : float
        Maximum y coordinate of the requested region.
    z_min : float
        Minimum z coordinate of the requested region.
    z_max : float
        Maximum z coordinate of the requested region.
    target : float
        Velocity value used to identify water cells.
    tolerance : float
        Absolute tolerance used when identifying the target velocity.
    x_chunk : int
        Number of x samples processed per interface-extraction chunk.
    y_chunk : int or None
        Number of y samples processed per chunk; ``None`` uses the full span.

    Returns
    -------
    tuple
        Interface x coordinates, y coordinates, bottom elevations, and top elevation.
    """
    try:
        dz, dx, dy = float(dz), float(dx), float(dy)
    except Exception as exc:
        raise RuntimeError("dz, dx, and dy must be valid spacing arguments.") from exc

    z_top, z_bot = float(max(z_min, z_max)), float(min(z_min, z_max))
    x_low, x_high = sorted((float(x_min), float(x_max)))
    y_low, y_high = sorted((float(y_min), float(y_max)))

    lx, ly, lz = (nx - 1) * dx, (ny - 1) * dy, (nz - 1) * dz
    x_low, x_high = np.clip([x_low, x_high], 0.0, lx)
    y_low, y_high = np.clip([y_low, y_high], 0.0, ly)
    z_top, z_bot = np.clip([z_top, z_bot], -lz, 0.0)

    ix_min, ix_max = int(np.rint(x_low / dx)), int(np.rint(x_high / dx))
    iy_min, iy_max = int(np.rint(y_low / dy)), int(np.rint(y_high / dy))
    ix_min, ix_max = sorted((max(0, min(ix_min, nx - 1)), max(0, min(ix_max, nx - 1))))
    iy_min, iy_max = sorted((max(0, min(iy_min, ny - 1)), max(0, min(iy_max, ny - 1))))

    iz_top = max(0, min(int(np.rint(-z_top / dz)), nz - 1))
    iz_bot = max(0, min(int(np.rint(-z_bot / dz)), nz - 1))
    iz_top, iz_bot = sorted((iz_top, iz_bot))

    xs = dx * np.arange(ix_min, ix_max + 1, dtype=np.float64)
    ys = dy * np.arange(iy_min, iy_max + 1, dtype=np.float64)
    nx_cells, ny_cells = ix_max - ix_min, iy_max - iy_min

    x_chunk = max(1, int(x_chunk))
    y_chunk = ny_cells + 1 if y_chunk is None else max(1, int(y_chunk))
    z_bottom = np.empty((nx_cells + 1, ny_cells + 1), dtype=np.float64)

    target, tolerance = float(target), float(tolerance)

    for i0 in range(0, nx_cells + 1, x_chunk):
        i1 = min(i0 + x_chunk, nx_cells + 1)
        ix_tile = ix_min + np.arange(i0, i1, dtype=int)

        for j0 in range(0, ny_cells + 1, y_chunk):
            j1 = min(j0 + y_chunk, ny_cells + 1)
            iy_tile = iy_min + np.arange(j0, j1, dtype=int)

            block = cube_zxy[iz_top: iz_bot + 1][:, ix_tile][:, :, iy_tile]
            block = np.asarray(block, dtype=np.float32)
            non_water = np.abs(block - target) > tolerance

            any_non_water = np.any(non_water, axis=0)
            first = np.argmax(non_water, axis=0).astype(np.int32)
            first = np.where(any_non_water, first, iz_bot - iz_top).astype(np.int32)
            z_bottom[i0:i1, j0:j1] = -(iz_top + first).astype(np.float64) * dz

    return xs, ys, z_bottom, float(z_top)


def _create_interface_bspline(gmsh, xs, ys, z_bottom):
    """Create the B-spline water/subsurface interface surface.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    xs : numpy.ndarray
        X coordinates of the reconstructed interface grid.
    ys : numpy.ndarray
        Y coordinates of the reconstructed interface grid.
    z_bottom : numpy.ndarray
        Reconstructed water/subsurface interface elevations.

    Returns
    -------
    tuple
        Interface surface tag and ordered point tags on the south, north, and west rims.
    """
    occ = gmsh.model.occ
    nx_cells, ny_cells = len(xs) - 1, len(ys) - 1

    points = [[None] * (ny_cells + 1) for _ in range(nx_cells + 1)]
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            points[i][j] = occ.addPoint(float(x), float(y), float(z_bottom[i, j]))

    south_pts = [points[i][0] for i in range(nx_cells + 1)]
    north_pts = [points[i][ny_cells] for i in range(nx_cells, -1, -1)]
    west_pts = [points[0][j] for j in range(ny_cells, -1, -1)]
    ctrl = [points[i][j] for j in range(ny_cells + 1) for i in range(nx_cells + 1)]

    surface = occ.addBSplineSurface(
        ctrl,
        nx_cells + 1,
        ny_cells + 1,
        min(3, nx_cells),
        min(3, ny_cells),
        [], [], [], [], [], [],
    )
    occ.synchronize()
    return surface, south_pts, north_pts, west_pts


def _oriented_interface_curves(gmsh, surface, x_low, x_high, y_low, y_high):
    """Identify and orient the four boundary curves of the interface surface.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    surface : int
        Gmsh surface tag.
    x_low : float
        Minimum x coordinate of the interface.
    x_high : float
        Maximum x coordinate of the interface.
    y_low : float
        Minimum y coordinate of the interface.
    y_high : float
        Maximum y coordinate of the interface.

    Returns
    -------
    tuple of int
        Oriented south, east, north, and west curve tags.
    """
    occ = gmsh.model.occ
    eps = 1.0e-9 * max(abs(x_high - x_low), abs(y_high - y_low), 1.0)
    _, curves = gmsh.model.getAdjacencies(2, surface)

    south = east = north = west = None
    for edge in dict.fromkeys(curves):
        _, pts = gmsh.model.getAdjacencies(1, edge)
        x0, y0, _ = occ.getCenterOfMass(0, pts[0])
        x1, y1, _ = occ.getCenterOfMass(0, pts[1])
        cx, cy, _ = occ.getCenterOfMass(1, edge)

        if abs(cy - y_low) < eps:
            south = edge if x1 >= x0 else -edge
        elif abs(cy - y_high) < eps:
            north = edge if x1 <= x0 else -edge
        elif abs(cx - x_high) < eps:
            east = edge if y1 >= y0 else -edge
        else:
            west = edge if y1 <= y0 else -edge

    if any(edge is None for edge in (south, east, north, west)):
        raise RuntimeError(
            "Could not classify the four B-spline water-interface boundary curves."
        )

    return south, east, north, west


def _build_water_subsoil_volumes(
    gmsh,
    s_bottom,
    south_pts,
    north_pts,
    west_pts,
    lB_S,
    lB_E,
    lB_N,
    lB_W,
    x_low,
    x_high,
    y_low,
    y_high,
    z_top,
    z_min,
):
    """Close the interface with top, bottom, and side surfaces to form water and subsoil volumes.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    s_bottom : int
        Gmsh tag of the water/subsurface interface surface.
    south_pts : sequence of int
        Point tags along the south edge of the interface.
    north_pts : sequence of int
        Point tags along the north edge of the interface.
    west_pts : sequence of int
        Point tags along the west edge of the interface.
    lB_S : int
        Oriented south boundary-curve tag.
    lB_E : int
        Oriented east boundary-curve tag.
    lB_N : int
        Oriented north boundary-curve tag.
    lB_W : int
        Oriented west boundary-curve tag.
    x_low : float
        Minimum x coordinate of the interface.
    x_high : float
        Maximum x coordinate of the interface.
    y_low : float
        Minimum y coordinate of the interface.
    y_high : float
        Maximum y coordinate of the interface.
    z_top : float
        Top elevation of the water volume.
    z_min : float
        Minimum z coordinate of the requested region.

    Returns
    -------
    tuple of int
        Subsurface and water volume tags.
    """
    occ = gmsh.model.occ

    # Top rectangle and water side walls.
    pT_SW = occ.addPoint(x_low, y_low, z_top)
    pT_SE = occ.addPoint(x_high, y_low, z_top)
    pT_NE = occ.addPoint(x_high, y_high, z_top)
    pT_NW = occ.addPoint(x_low, y_high, z_top)

    lT_S = occ.addLine(pT_SW, pT_SE)
    lT_E = occ.addLine(pT_SE, pT_NE)
    lT_N = occ.addLine(pT_NE, pT_NW)
    lT_W = occ.addLine(pT_NW, pT_SW)
    s_top = occ.addBSplineFilling(occ.addWire([lT_S, lT_E, lT_N, lT_W]))

    v_SW = occ.addLine(pT_SW, south_pts[0])
    v_SE = occ.addLine(pT_SE, south_pts[-1])
    v_NE = occ.addLine(pT_NE, north_pts[0])
    v_NW = occ.addLine(pT_NW, west_pts[0])

    s_side_S = occ.addBSplineFilling(occ.addWire([lT_S, v_SE, -lB_S, -v_SW]))
    s_side_E = occ.addBSplineFilling(occ.addWire([lT_E, v_NE, -lB_E, -v_SE]))
    s_side_N = occ.addBSplineFilling(occ.addWire([lT_N, v_NW, -lB_N, -v_NE]))
    s_side_W = occ.addBSplineFilling(occ.addWire([lT_W, v_SW, -lB_W, -v_NW]))

    # Flat domain bottom and lower side walls.
    pB_SW = occ.addPoint(x_low, y_low, z_min)
    pB_SE = occ.addPoint(x_high, y_low, z_min)
    pB_NE = occ.addPoint(x_high, y_high, z_min)
    pB_NW = occ.addPoint(x_low, y_high, z_min)

    lBot_S = occ.addLine(pB_SW, pB_SE)
    lBot_E = occ.addLine(pB_SE, pB_NE)
    lBot_N = occ.addLine(pB_NE, pB_NW)
    lBot_W = occ.addLine(pB_NW, pB_SW)
    s_bot_flat = occ.addBSplineFilling(
        occ.addWire([lBot_S, lBot_E, lBot_N, lBot_W])
    )

    v_SW2 = occ.addLine(south_pts[0], pB_SW)
    v_SE2 = occ.addLine(south_pts[-1], pB_SE)
    v_NE2 = occ.addLine(north_pts[0], pB_NE)
    v_NW2 = occ.addLine(west_pts[0], pB_NW)

    s_side_S2 = occ.addBSplineFilling(occ.addWire([lB_S, v_SE2, -lBot_S, -v_SW2]))
    s_side_E2 = occ.addBSplineFilling(occ.addWire([lB_E, v_NE2, -lBot_E, -v_SE2]))
    s_side_N2 = occ.addBSplineFilling(occ.addWire([lB_N, v_NW2, -lBot_N, -v_NE2]))
    s_side_W2 = occ.addBSplineFilling(occ.addWire([lB_W, v_SW2, -lBot_W, -v_NW2]))

    occ.removeAllDuplicates()
    occ.synchronize()

    vol_subsoil = occ.addVolume(
        [occ.addSurfaceLoop([s_bottom, s_bot_flat, s_side_S2, s_side_E2, s_side_N2, s_side_W2])]
    )
    vol_water = occ.addVolume(
        [occ.addSurfaceLoop([s_top, s_bottom, s_side_S, s_side_E, s_side_N, s_side_W], sewing=True)]
    )
    occ.synchronize()

    return vol_subsoil, vol_water


def generate_water_interface_volumes(
    gmsh,
    fname,
    water_search_value,
    nz,
    nx,
    ny,
    dz,
    dx,
    dy,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    byte_order="big",
    order="F",
    dtype="float32",
    tolerance=1.0,
    x_chunk=64,
    y_chunk=None,
    *,
    comm,
    parallel_print,
):
    """Generate geometrically delimited water and subsurface volumes from a velocity model.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    fname : str or pathlib.Path
        Path to the binary velocity model.
    water_search_value : float
        Velocity value used to identify the water layer.
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
    x_min : float
        Minimum x coordinate of the requested region.
    x_max : float
        Maximum x coordinate of the requested region.
    y_min : float
        Minimum y coordinate of the requested region.
    y_max : float
        Maximum y coordinate of the requested region.
    z_min : float
        Minimum z coordinate of the requested region.
    z_max : float
        Maximum z coordinate of the requested region.
    byte_order : {"big", "little"}
        Byte order of the velocity-model binary file.
    order : {"C", "F"}
        Array reshape order used for the binary velocity model.
    dtype : str or numpy.dtype
        Numeric data type stored in the velocity-model file.
    tolerance : float
        Absolute tolerance used when identifying the target velocity.
    x_chunk : int
        Number of x samples processed per interface-extraction chunk.
    y_chunk : int or None
        Number of y samples processed per chunk; ``None`` uses the full span.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    tuple of int
        Subsurface and water volume tags.
    """
    parallel_print("Generating geometrically delimited water and subsurface volumes...", comm=comm)

    dtype_string = np.dtype(dtype).newbyteorder(">" if byte_order == "big" else "<").str
    cube, nz, nx, ny = _read_velocity_cube(fname, nz, nx, ny, dtype_string, order)

    xs, ys, z_bottom, z_top = _water_interface_grid(
        cube,
        nz,
        nx,
        ny,
        dz,
        dx,
        dy,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        water_search_value,
        tolerance,
        x_chunk,
        y_chunk,
    )

    gmsh.option.setNumber("Geometry.Tolerance", 1.0e-16)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

    s_bottom, south_pts, north_pts, west_pts = _create_interface_bspline(
        gmsh, xs, ys, z_bottom
    )

    x_low, x_high = float(xs[0]), float(xs[-1])
    y_low, y_high = float(ys[0]), float(ys[-1])
    lB_S, lB_E, lB_N, lB_W = _oriented_interface_curves(
        gmsh, s_bottom, x_low, x_high, y_low, y_high
    )

    return _build_water_subsoil_volumes(
        gmsh,
        s_bottom,
        south_pts,
        north_pts,
        west_pts,
        lB_S,
        lB_E,
        lB_N,
        lB_W,
        x_low,
        x_high,
        y_low,
        y_high,
        z_top,
        float(z_min),
    )


def find_outer_volume_face(gmsh, volume, axis, side):
    """Find the outermost boundary face of a volume along a Cartesian axis.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    volume : int
        Gmsh volume tag.
    axis : int
        Cartesian axis index: 0 for x, 1 for y, or 2 for z.
    side : int
        Side selector; positive chooses the maximum side and negative the minimum.

    Returns
    -------
    int
        Tag of the selected outer boundary face.
    """
    occ = gmsh.model.occ
    faces = [
        tag
        for dim, tag in gmsh.model.getBoundary(
            [(3, int(volume))],
            oriented=False,
            recursive=False,
        )
        if dim == 2
    ]

    if not faces:
        raise RuntimeError(f"Volume {volume} has no boundary surfaces.")

    centers = {
        face: occ.getCenterOfMass(2, face)[axis]
        for face in faces
    }
    return (max if side > 0 else min)(faces, key=centers.get)


def sweep_volume_faces(gmsh, volumes, axis, side, distance):
    """Extrude selected outer faces and map each source volume to its generated volume.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    volumes : sequence of int
        Gmsh volume tags to process.
    axis : int
        Cartesian axis index: 0 for x, 1 for y, or 2 for z.
    side : int
        Side selector; positive chooses the maximum side and negative the minimum.
    distance : float
        Extrusion distance.

    Returns
    -------
    dict
        Mapping from source volume tags to generated swept-volume tags.
    """
    occ = gmsh.model.occ
    volumes = list(
        dict.fromkeys(
            int(volume)
            for volume in volumes
            if volume is not None
        )
    )

    if not volumes or distance <= 0.0:
        return {}

    source_faces = {
        volume: find_outer_volume_face(gmsh, volume, axis, side)
        for volume in volumes
    }

    direction = [0.0, 0.0, 0.0]
    direction[axis] = float(side) * float(distance)

    result = occ.extrude(
        [(2, face) for face in source_faces.values()],
        *direction,
    )
    occ.synchronize()

    generated_volumes = [tag for dim, tag in result if dim == 3]
    if not generated_volumes:
        raise RuntimeError(
            "OCC extrusion generated no volumes "
            f"(axis={axis}, side={side}, distance={distance})."
        )

    generated_boundaries = {
        volume: {
            tag
            for dim, tag in gmsh.model.getBoundary(
                [(3, volume)],
                oriented=False,
                recursive=False,
            )
            if dim == 2
        }
        for volume in generated_volumes
    }

    mapping = {}
    for source_volume, source_face in source_faces.items():
        matches = [
            generated_volume
            for generated_volume in generated_volumes
            if source_face in generated_boundaries[generated_volume]
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Could not uniquely identify swept volume.\n"
                f"  source volume = {source_volume}\n"
                f"  source face   = {source_face}\n"
                f"  axis          = {axis}\n"
                f"  side          = {side}\n"
                f"  matches       = {matches}"
            )

        mapping[source_volume] = matches[0]

    return mapping


def generate_structured_rectangular_padding_water(
    gmsh,
    vol_water,
    vol_subsoil,
    padding_x,
    padding_y,
    padding_z,
    *,
    comm,
    parallel_print,
):
    """Create conformal rectangular padding around delimited water and subsurface volumes.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    vol_water : int
        Tag of the central water volume.
    vol_subsoil : int
        Tag of the central subsurface volume.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    tuple
        Water-volume tags, soil-volume tags, and padding-volume tags.

    Notes
    -----
    Padding is generated by OpenCASCADE face sweeps rather than Boolean fragmentation so
    that the structured multiblock topology remains suitable for hexahedral meshing.
    """
    parallel_print("Generating structured rectangular padding by OCC sweeps...", comm=comm)

    occ = gmsh.model.occ
    occ.synchronize()

    vol_water = int(vol_water)
    vol_subsoil = int(vol_subsoil)

    water = {(0, 0): vol_water}
    soil = {(0, 0): vol_subsoil}

    # X padding
    if padding_x > 0.0:
        for side in (-1, +1):
            mapping = sweep_volume_faces(
                gmsh,
                [vol_water, vol_subsoil],
                axis=0,
                side=side,
                distance=padding_x,
            )
            water[(side, 0)] = mapping[vol_water]
            soil[(side, 0)] = mapping[vol_subsoil]

    # Y padding
    if padding_y > 0.0:
        water_row = sorted(water.items(), key=lambda item: item[0][0])
        soil_row = sorted(soil.items(), key=lambda item: item[0][0])
        sources = (
            [volume for _, volume in water_row]
            + [volume for _, volume in soil_row]
        )

        for side in (-1, +1):
            mapping = sweep_volume_faces(
                gmsh,
                sources,
                axis=1,
                side=side,
                distance=padding_y,
            )

            for (xposition, _), volume in water_row:
                water[(xposition, side)] = mapping[volume]
            for (xposition, _), volume in soil_row:
                soil[(xposition, side)] = mapping[volume]

    # Bottom padding
    soil_xy = [
        volume
        for _, volume in sorted(
            soil.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]

    if padding_z > 0.0:
        bottom_map = sweep_volume_faces(
            gmsh,
            soil_xy,
            axis=2,
            side=-1,
            distance=padding_z,
        )
        soil_bottom = [bottom_map[volume] for volume in soil_xy]
    else:
        soil_bottom = []

    water_vols = [
        volume
        for _, volume in sorted(
            water.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]
    soil_vols = soil_xy + soil_bottom

    pad_vols = (
        [volume for volume in water_vols if volume != vol_water]
        + [volume for volume in soil_vols if volume != vol_subsoil]
    )

    water_vols = list(dict.fromkeys(water_vols))
    soil_vols = list(dict.fromkeys(soil_vols))
    pad_vols = list(dict.fromkeys(pad_vols))

    gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="Subsurface")
    gmsh.model.addPhysicalGroup(3, [vol_water], name="Water")
    if pad_vols:
        gmsh.model.addPhysicalGroup(3, pad_vols, name="Padding")
    gmsh.model.addPhysicalGroup(3, water_vols, name="Water_with_padding")
    gmsh.model.addPhysicalGroup(3, soil_vols, name="Subsurface_with_padding")

    occ.synchronize()

    return water_vols, soil_vols, pad_vols


def report_quality(
    gmsh,
    dim=3,
    quality_type=2,
    *,
    comm,
    parallel_print,
):
    """Compute and report Gmsh element-quality statistics.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    dim : int
        Topological dimension of the elements or callback entity.
    quality_type : int
        Gmsh element-quality metric identifier.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    None
        This function reports mesh-quality statistics and does not return a value.
    """
    gmsh.option.setNumber("Mesh.QualityType", quality_type)  # choose metric

    # Get elements
    _elem_types, elem_tags, _elem_node_tags = gmsh.model.mesh.getElements(dim)

    # Collect element tags
    all_tags = []
    for tags in elem_tags:
        all_tags.extend(tags.tolist() if hasattr(tags, "tolist") else list(tags))

    if not all_tags:
        parallel_print(f"[quality] No elements found for dim={dim}", comm=comm)
        return

    # Element quality
    q = gmsh.model.mesh.getElementQualities(all_tags)

    q = np.asarray(q, dtype=float)
    parallel_print(
        f"[quality] count={q.size}  min={q.min():.6g}  p1={np.percentile(q, 1):.6g}  "
        f"p5={np.percentile(q, 5):.6g}  median={np.median(q):.6g}  "
        f"p95={np.percentile(q, 95):.6g}  max={q.max():.6g}  mean={q.mean():.6g}",
        comm=comm,
    )
