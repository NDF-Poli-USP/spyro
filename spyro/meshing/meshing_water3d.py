from pathlib import Path

import numpy as np

from spyro.mpi.spyro_mpi import SpyroEnsemble


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
    except (TypeError, ValueError) as exc:
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

    Returns
    -------
    tuple of int
        Subsurface and water volume tags.
    """
    SpyroEnsemble.print("Generating geometrically delimited water and subsurface volumes...")

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
