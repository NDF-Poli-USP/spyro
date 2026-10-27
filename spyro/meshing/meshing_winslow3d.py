import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        """Return a decorator when Numba is unavailable."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


from .meshing_utils3d import sizing_function_xyz
from spyro.mpi.spyro_mpi import SpyroEnsemble


def run_selected_winslow(
    points,
    hexes,
    move_all,
    move_x_only,
    move_y_only,
    move_z_only,
    ef_segy,
    length_x,
    length_y,
    depth_z,
    winslow_iterations,
    winslow_omega,
    selected_winslow="numba",
):
    """Run the selected three-dimensional Winslow smoother and report its effect.

    Parameters
    ----------
    points : numpy.ndarray
        Mesh-node coordinates with shape ``(N, 3)``.
    hexes : numpy.ndarray
        Eight-node hexahedral connectivity using zero-based node indices.
    move_all : set of int
        Nodes allowed to move in x, y, and z.
    move_x_only : set of int
        Nodes whose x coordinate may move.
    move_y_only : set of int
        Nodes whose y coordinate may move.
    move_z_only : set of int
        Nodes whose z coordinate may move.
    ef_segy : callable
        Mesh-sizing function evaluated in ``(z, x, y)`` coordinates.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth.
    winslow_iterations : int
        Number of Winslow smoothing iterations.
    winslow_omega : float
        Winslow relaxation coefficient.
    selected_winslow : str
        Winslow implementation selector.

    Returns
    -------
    numpy.ndarray
        Smoothed node coordinates.
    """
    movable_nodes = (
        set(move_all) | set(move_x_only) | set(move_y_only) | set(move_z_only)
    )

    if not movable_nodes:
        raise RuntimeError(
            "Winslow smoothing has no movable nodes. Check the Water "
            "physical group and rectangular boundary classification."
        )

    movable_indices = np.asarray(sorted(movable_nodes), dtype=np.int64)

    initial_sizes = sizing_function_xyz(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        ef_segy,
        length_x,
        length_y,
        depth_z,
    )
    movable_sizes = initial_sizes[movable_indices]

    size_min = float(np.min(movable_sizes))
    size_max = float(np.max(movable_sizes))
    size_mean = float(np.mean(movable_sizes))
    size_std = float(np.std(movable_sizes))
    size_span = size_max - size_min

    SpyroEnsemble.print(
        "Winslow sizing on movable nodes | "
        f"min={size_min:.9g}, max={size_max:.9g}, "
        f"mean={size_mean:.9g}, std={size_std:.9g}"
    )
    SpyroEnsemble.print(
        "Winslow movable-node counts | "
        f"all={len(move_all)}, X={len(move_x_only)}, "
        f"Y={len(move_y_only)}, Z={len(move_z_only)}, "
        f"union={len(movable_nodes)}"
    )

    relative_span = size_span / max(abs(size_mean), 1.0)
    if relative_span <= 1.0e-10:
        SpyroEnsemble.print(
            "WARNING: the sizing field is effectively constant on the "
            "movable nodes. A uniform structured mesh will remain "
            "unchanged. Consider reducing hmin_segy."
        )

    if selected_winslow != "numba":
        raise ValueError(
            "Only selected_winslow='numba' is currently supported. "
            f"Received {selected_winslow!r}."
        )

    if NUMBA_AVAILABLE:
        SpyroEnsemble.print(
            "Using Numba accelerated 3D Winslow implementation.",
        )
    else:
        SpyroEnsemble.print(
            "Using Winslow implementation without Numba.",
        )
    smoothed = winslow_smooth_3d(
        points=points,
        hexes=hexes,
        move_all=move_all,
        move_x_only=move_x_only,
        move_y_only=move_y_only,
        move_z_only=move_z_only,
        iterations=winslow_iterations,
        omega=winslow_omega,
        ef_segy=ef_segy,
        length_x=length_x,
        length_y=length_y,
        depth_z=depth_z,
    )

    displacement = np.linalg.norm(
        np.asarray(smoothed, dtype=float) - np.asarray(points, dtype=float),
        axis=1,
    )
    movable_displacement = displacement[movable_indices]

    moved_count = int(np.count_nonzero(movable_displacement > 1.0e-10))
    max_displacement = float(np.max(movable_displacement))
    mean_displacement = float(np.mean(movable_displacement))

    SpyroEnsemble.print(
        "Winslow displacement | "
        f"moved={moved_count}/{len(movable_indices)}, "
        f"max={max_displacement:.9g}, "
        f"mean={mean_displacement:.9g}"
    )

    if moved_count == 0:
        SpyroEnsemble.print(
            "WARNING: winslow_smooth_3d completed but no movable node "
            "changed position. Inspect the sizing statistics above."
        )

    return smoothed


@njit(cache=True, inline="always")
def _cardinal_coordinate(
    coordinates: np.ndarray, center: float, neighbor: int, opposite: int
) -> float:
    """Return a cardinal-neighbor coordinate, reflecting it when missing.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Coordinate values for one spatial direction at all mesh nodes.
    center : float
        Coordinate of the current stencil-center node.
    neighbor : int
        Node index of the cardinal neighbor, or -1 when it is missing.
    opposite : int
        Node index of the opposite cardinal neighbor, or -1 when it is missing.

    Returns
    -------
    float
        The neighbor coordinate; if missing, the opposite neighbor reflected
        through the center; if both are missing, the center.
    """
    if neighbor != -1:
        return coordinates[neighbor]
    if opposite != -1:
        return 2.0 * center - coordinates[opposite]
    return center


@njit(cache=True, inline="always")
def _mixed_coordinate(
    coordinates: np.ndarray,
    center: float,
    neighbor: int,
    first_value: float,
    second_value: float,
) -> float:
    """Return a mixed (diagonal) neighbor coordinate, rebuilding it when missing.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Coordinate values for one spatial direction at all mesh nodes.
    center : float
        Coordinate of the current stencil-center node.
    neighbor : int
        Node index of the mixed neighbor, or -1 when it is missing.
    first_value : float
        Resolved coordinate of the first cardinal direction (e.g. N for NE).
    second_value : float
        Resolved coordinate of the second cardinal direction (e.g. E for NE).

    Returns
    -------
    float
        The neighbor coordinate, or `first_value + second_value - center`
        when it is missing.
    """
    if neighbor != -1:
        return coordinates[neighbor]
    return first_value + second_value - center


@njit(cache=True, inline="always")
def _neighbor_size(
    sizing_values: np.ndarray,
    is_movable: np.ndarray,
    neighbor: int,
    size_center: float,
) -> float:
    """Return a neighbor's sizing value, or the center value if not usable.

    Parameters
    ----------
    sizing_values : numpy.ndarray
        Sizing-field value associated with each mesh node.
    is_movable : numpy.ndarray
        Boolean mask identifying nodes involved in smoothing.
    neighbor : int
        Node index of the neighbor, or -1 when it is missing.
    size_center : float
        Sizing value at the current stencil-center node.

    Returns
    -------
    float
        `sizing_values[neighbor]` for an existing movable neighbor,
        otherwise `size_center`.
    """
    if neighbor != -1 and is_movable[neighbor]:
        return sizing_values[neighbor]
    return size_center


@njit(cache=True, inline="always")
def _resolve_stencil_coordinates(coordinates, center, stencil):
    """Resolve cardinal and mixed stencil coordinates for one spatial axis.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Coordinate values for one spatial direction at all mesh nodes.
    center : float
        Coordinate of the current stencil-center node.
    stencil : numpy.ndarray
        Node-index stencil ordered as E, W, N, S, T, B,
        NE, NW, SE, SW, NT, NB, ST, SB, ET, EB, WT, and WB.

    Returns
    -------
    tuple of float
        Resolved coordinates in E, W, N, S, T, B, NE, NW, SE, SW, NT,
        NB, ST, SB, ET, EB, WT, and WB order.

    Notes
    -----
    Missing cardinal neighbors are reflected through the center node from
    the available opposite neighbor. Missing mixed neighbors are reconstructed
    from their two resolved cardinal coordinates.
    """
    east, west, north, south, top, bottom = stencil[:6]
    ne, nw, se, sw, nt, nb, st, sb, et, eb, wt, wb = stencil[6:]

    east_value = _cardinal_coordinate(coordinates, center, east, west)
    west_value = _cardinal_coordinate(coordinates, center, west, east)
    north_value = _cardinal_coordinate(coordinates, center, north, south)
    south_value = _cardinal_coordinate(coordinates, center, south, north)
    top_value = _cardinal_coordinate(coordinates, center, top, bottom)
    bottom_value = _cardinal_coordinate(coordinates, center, bottom, top)

    # Explicit calls rather than a lookup table: this runs per node inside
    # the Numba hot loop, and a table would need an array allocation.
    return (
        east_value,
        west_value,
        north_value,
        south_value,
        top_value,
        bottom_value,
        _mixed_coordinate(coordinates, center, ne, north_value, east_value),
        _mixed_coordinate(coordinates, center, nw, north_value, west_value),
        _mixed_coordinate(coordinates, center, se, south_value, east_value),
        _mixed_coordinate(coordinates, center, sw, south_value, west_value),
        _mixed_coordinate(coordinates, center, nt, north_value, top_value),
        _mixed_coordinate(coordinates, center, nb, north_value, bottom_value),
        _mixed_coordinate(coordinates, center, st, south_value, top_value),
        _mixed_coordinate(coordinates, center, sb, south_value, bottom_value),
        _mixed_coordinate(coordinates, center, et, east_value, top_value),
        _mixed_coordinate(coordinates, center, eb, east_value, bottom_value),
        _mixed_coordinate(coordinates, center, wt, west_value, top_value),
        _mixed_coordinate(coordinates, center, wb, west_value, bottom_value),
    )


@njit(cache=True, parallel=True)
def _numba_winslow_3d_step(
    x_coordinates,
    y_coordinates,
    z_coordinates,
    sizing_values,
    stencils,
    is_movable,
    move_x,
    move_y,
    move_z,
    omega,
):
    """Perform one parallel Numba Winslow update over the structured mesh.

    Parameters
    ----------
    x_coordinates : numpy.ndarray
        Current x coordinates of all mesh nodes.
    y_coordinates : numpy.ndarray
        Current y coordinates of all mesh nodes.
    z_coordinates : numpy.ndarray
        Current z coordinates of all mesh nodes.
    sizing_values : numpy.ndarray
        Sizing-field value associated with each mesh node.
    stencils : numpy.ndarray
        Integer neighbor stencil with cardinal and mixed directions.
    is_movable : numpy.ndarray
        Boolean mask identifying nodes involved in smoothing.
    move_x : numpy.ndarray
        Boolean mask for nodes permitted to move in x.
    move_y : numpy.ndarray
        Boolean mask for nodes permitted to move in y.
    move_z : numpy.ndarray
        Boolean mask for nodes permitted to move in z.
    omega : float
        Relaxation coefficient applied to each Winslow update.

    Returns
    -------
    tuple of numpy.ndarray
        Updated x, y, and z coordinate arrays.
    """
    num_nodes = len(x_coordinates)

    x_new = np.empty_like(x_coordinates)
    y_new = np.empty_like(y_coordinates)
    z_new = np.empty_like(z_coordinates)

    for node in prange(num_nodes):
        x_new[node] = x_coordinates[node]
        y_new[node] = y_coordinates[node]
        z_new[node] = z_coordinates[node]

        if not (move_x[node] or move_y[node] or move_z[node]):
            continue

        x_center = x_coordinates[node]
        y_center = y_coordinates[node]
        z_center = z_coordinates[node]

        stencil = stencils[node]
        east, west, north, south, top, bottom = stencil[:6]

        (
            x_east,
            x_west,
            x_north,
            x_south,
            x_top,
            x_bottom,
            x_ne,
            x_nw,
            x_se,
            x_sw,
            x_nt,
            x_nb,
            x_st,
            x_sb,
            x_et,
            x_eb,
            x_wt,
            x_wb,
        ) = _resolve_stencil_coordinates(x_coordinates, x_center, stencil)

        (
            y_east,
            y_west,
            y_north,
            y_south,
            y_top,
            y_bottom,
            y_ne,
            y_nw,
            y_se,
            y_sw,
            y_nt,
            y_nb,
            y_st,
            y_sb,
            y_et,
            y_eb,
            y_wt,
            y_wb,
        ) = _resolve_stencil_coordinates(y_coordinates, y_center, stencil)

        (
            z_east,
            z_west,
            z_north,
            z_south,
            z_top,
            z_bottom,
            z_ne,
            z_nw,
            z_se,
            z_sw,
            z_nt,
            z_nb,
            z_st,
            z_sb,
            z_et,
            z_eb,
            z_wt,
            z_wb,
        ) = _resolve_stencil_coordinates(z_coordinates, z_center, stencil)

        x_xi = 0.5 * (x_east - x_west)
        y_xi = 0.5 * (y_east - y_west)
        z_xi = 0.5 * (z_east - z_west)

        x_eta = 0.5 * (x_north - x_south)
        y_eta = 0.5 * (y_north - y_south)
        z_eta = 0.5 * (z_north - z_south)

        x_zeta = 0.5 * (x_top - x_bottom)
        y_zeta = 0.5 * (y_top - y_bottom)
        z_zeta = 0.5 * (z_top - z_bottom)

        g11 = x_xi**2 + y_xi**2 + z_xi**2
        g22 = x_eta**2 + y_eta**2 + z_eta**2
        g33 = x_zeta**2 + y_zeta**2 + z_zeta**2
        g12 = x_xi * x_eta + y_xi * y_eta + z_xi * z_eta
        g23 = x_eta * x_zeta + y_eta * y_zeta + z_eta * z_zeta
        g13 = x_xi * x_zeta + y_xi * y_zeta + z_xi * z_zeta

        coefficient_a = g22 * g33 - g23**2
        coefficient_b = g11 * g33 - g13**2
        coefficient_c = g11 * g22 - g12**2
        coefficient_d = g13 * g23 - g12 * g33
        coefficient_e = g12 * g13 - g11 * g23
        coefficient_f = g12 * g23 - g22 * g13

        x_xi_eta = 0.25 * (x_ne - x_nw - x_se + x_sw)
        y_xi_eta = 0.25 * (y_ne - y_nw - y_se + y_sw)
        z_xi_eta = 0.25 * (z_ne - z_nw - z_se + z_sw)

        x_eta_zeta = 0.25 * (x_nt - x_nb - x_st + x_sb)
        y_eta_zeta = 0.25 * (y_nt - y_nb - y_st + y_sb)
        z_eta_zeta = 0.25 * (z_nt - z_nb - z_st + z_sb)

        x_xi_zeta = 0.25 * (x_et - x_eb - x_wt + x_wb)
        y_xi_zeta = 0.25 * (y_et - y_eb - y_wt + y_wb)
        z_xi_zeta = 0.25 * (z_et - z_eb - z_wt + z_wb)

        size_center = sizing_values[node]
        size_east = _neighbor_size(sizing_values, is_movable, east, size_center)
        size_west = _neighbor_size(sizing_values, is_movable, west, size_center)
        size_north = _neighbor_size(sizing_values, is_movable, north, size_center)
        size_south = _neighbor_size(sizing_values, is_movable, south, size_center)
        size_top = _neighbor_size(sizing_values, is_movable, top, size_center)
        size_bottom = _neighbor_size(sizing_values, is_movable, bottom, size_center)

        size_xi = 0.5 * (size_east - size_west)
        size_eta = 0.5 * (size_north - size_south)
        size_zeta = 0.5 * (size_top - size_bottom)

        jacobian = (
            x_xi * (y_eta * z_zeta - z_eta * y_zeta)
            - y_xi * (x_eta * z_zeta - z_eta * x_zeta)
            + z_xi * (x_eta * y_zeta - y_eta * x_zeta)
        )
        jacobian_over_size = jacobian / (size_center + 1.0e-12)

        adjoint_11 = y_eta * z_zeta - z_eta * y_zeta
        adjoint_21 = z_xi * y_zeta - y_xi * z_zeta
        adjoint_31 = y_xi * z_eta - z_xi * y_eta
        adjoint_12 = z_eta * x_zeta - x_eta * z_zeta
        adjoint_22 = x_xi * z_zeta - z_xi * x_zeta
        adjoint_32 = z_xi * x_eta - x_xi * z_eta
        adjoint_13 = x_eta * y_zeta - y_eta * x_zeta
        adjoint_23 = y_xi * x_zeta - x_xi * y_zeta
        adjoint_33 = x_xi * y_eta - y_xi * x_eta

        source_x = jacobian_over_size * (
            size_xi * adjoint_11 + size_eta * adjoint_21 + size_zeta * adjoint_31
        )
        source_y = jacobian_over_size * (
            size_xi * adjoint_12 + size_eta * adjoint_22 + size_zeta * adjoint_32
        )
        source_z = jacobian_over_size * (
            size_xi * adjoint_13 + size_eta * adjoint_23 + size_zeta * adjoint_33
        )

        denominator = 2.0 * (coefficient_a + coefficient_b + coefficient_c)
        if denominator < 1.0e-12:
            continue

        if move_x[node]:
            x_target = (
                coefficient_a * (x_east + x_west)
                + coefficient_b * (x_north + x_south)
                + coefficient_c * (x_top + x_bottom)
                + 2.0
                * (
                    coefficient_d * x_xi_eta
                    + coefficient_e * x_eta_zeta
                    + coefficient_f * x_xi_zeta
                )
                - source_x
            ) / denominator
            x_new[node] = (1.0 - omega) * x_center + omega * x_target

        if move_y[node]:
            y_target = (
                coefficient_a * (y_east + y_west)
                + coefficient_b * (y_north + y_south)
                + coefficient_c * (y_top + y_bottom)
                + 2.0
                * (
                    coefficient_d * y_xi_eta
                    + coefficient_e * y_eta_zeta
                    + coefficient_f * y_xi_zeta
                )
                - source_y
            ) / denominator
            y_new[node] = (1.0 - omega) * y_center + omega * y_target

        if move_z[node]:
            z_target = (
                coefficient_a * (z_east + z_west)
                + coefficient_b * (z_north + z_south)
                + coefficient_c * (z_top + z_bottom)
                + 2.0
                * (
                    coefficient_d * z_xi_eta
                    + coefficient_e * z_eta_zeta
                    + coefficient_f * z_xi_zeta
                )
                - source_z
            ) / denominator
            z_new[node] = (1.0 - omega) * z_center + omega * z_target

    return x_new, y_new, z_new


def _shared_neighbor(
    node_to_neighbors: dict[int, set[int]],
    node: int,
    first_neighbor: int | None,
    second_neighbor: int | None,
) -> int | None:
    """Return the diagonal stencil node shared by two cardinal neighbors.

    Parameters
    ----------
    node_to_neighbors : dict of int to set of int
        Edge-connected neighbors of every mesh node.
    node : int
        Index of the stencil centre, excluded from the result.
    first_neighbor : int or None
        First cardinal-neighbor node index.
    second_neighbor : int or None
        Second cardinal-neighbor node index.

    Returns
    -------
    int or None
        Smallest shared neighbor index, or ``None`` when no shared neighbor exists.
    """
    if first_neighbor is None or second_neighbor is None:
        return None

    shared = node_to_neighbors[first_neighbor].intersection(
        node_to_neighbors[second_neighbor]
    )
    shared.discard(node)

    if not shared:
        return None

    return min(shared)


def winslow_smooth_3d(
    points: np.ndarray,
    hexes: np.ndarray,
    move_all: set[int],
    move_x_only: set[int],
    move_y_only: set[int],
    move_z_only: set[int],
    ef_segy,
    length_x,
    length_y,
    depth_z,
    iterations: int = 50,
    omega: float = 0.5,
) -> np.ndarray:
    """Smooth a structured hexahedral mesh with the three-dimensional Winslow equations.

    Parameters
    ----------
    points : numpy.ndarray
        Mesh-node coordinates with shape ``(N, 3)``.
    hexes : numpy.ndarray
        Eight-node hexahedral connectivity using zero-based node indices.
    move_all : set of int
        Nodes allowed to move in x, y, and z.
    move_x_only : set of int
        Nodes whose x coordinate may move.
    move_y_only : set of int
        Nodes whose y coordinate may move.
    move_z_only : set of int
        Nodes whose z coordinate may move.
    ef_segy : callable
        Mesh-sizing function evaluated in ``(z, x, y)`` coordinates.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth.
    iterations : int
        Number of smoothing iterations.
    omega : float
        Relaxation coefficient applied to each Winslow update.

    Returns
    -------
    numpy.ndarray
        Smoothed node coordinates with shape ``(N, 3)``.

    Notes
    -----
    The movement sets may overlap. The resulting per-axis masks are formed by the union
    of ``move_all`` with each directional set.
    """
    points = np.asarray(points, dtype=float)
    hexes = np.asarray(hexes, dtype=np.int64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (number_of_nodes, 3).")
    if hexes.ndim != 2 or hexes.shape[1] != 8:
        raise ValueError("hexes must have shape (number_of_hexes, 8).")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if not 0.0 < omega <= 1.0:
        raise ValueError("omega must satisfy 0 < omega <= 1.")

    num_nodes = len(points)
    node_to_neighbors = {node: set() for node in range(num_nodes)}

    hex_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    for hex_element in hexes:
        for first_local_node, second_local_node in hex_edges:
            first_node = int(hex_element[first_local_node])
            second_node = int(hex_element[second_local_node])

            node_to_neighbors[first_node].add(second_node)
            node_to_neighbors[second_node].add(first_node)

    movable_nodes = (
        set(move_all) | set(move_x_only) | set(move_y_only) | set(move_z_only)
    )

    invalid_nodes = {node for node in movable_nodes if node < 0 or node >= num_nodes}
    if invalid_nodes:
        raise ValueError(
            "Movement sets contain node indices outside the mesh: "
            f"{sorted(invalid_nodes)}."
        )

    stencils = {}

    for node in movable_nodes:
        neighbors = list(node_to_neighbors[node])
        if not neighbors:
            continue

        cardinals = {
            "E": None,
            "W": None,
            "N": None,
            "S": None,
            "T": None,
            "B": None,
        }

        for neighbor in neighbors:
            vector = points[neighbor] - points[node]
            absolute_x, absolute_y, absolute_z = np.abs(vector)

            if absolute_x > absolute_y and absolute_x > absolute_z:
                direction = "E" if vector[0] > 0.0 else "W"
            elif absolute_y > absolute_x and absolute_y > absolute_z:
                direction = "N" if vector[1] > 0.0 else "S"
            else:
                direction = "T" if vector[2] > 0.0 else "B"

            cardinals[direction] = neighbor

        corners = {
            "NE": _shared_neighbor(
                node_to_neighbors, node, cardinals["N"], cardinals["E"]
            ),
            "NW": _shared_neighbor(
                node_to_neighbors, node, cardinals["N"], cardinals["W"]
            ),
            "SE": _shared_neighbor(
                node_to_neighbors, node, cardinals["S"], cardinals["E"]
            ),
            "SW": _shared_neighbor(
                node_to_neighbors, node, cardinals["S"], cardinals["W"]
            ),
            "NT": _shared_neighbor(
                node_to_neighbors, node, cardinals["N"], cardinals["T"]
            ),
            "NB": _shared_neighbor(
                node_to_neighbors, node, cardinals["N"], cardinals["B"]
            ),
            "ST": _shared_neighbor(
                node_to_neighbors, node, cardinals["S"], cardinals["T"]
            ),
            "SB": _shared_neighbor(
                node_to_neighbors, node, cardinals["S"], cardinals["B"]
            ),
            "ET": _shared_neighbor(
                node_to_neighbors, node, cardinals["E"], cardinals["T"]
            ),
            "EB": _shared_neighbor(
                node_to_neighbors, node, cardinals["E"], cardinals["B"]
            ),
            "WT": _shared_neighbor(
                node_to_neighbors, node, cardinals["W"], cardinals["T"]
            ),
            "WB": _shared_neighbor(
                node_to_neighbors, node, cardinals["W"], cardinals["B"]
            ),
        }
        stencils[node] = {
            **cardinals,
            **corners,
        }

    direction_columns = {
        "E": 0,
        "W": 1,
        "N": 2,
        "S": 3,
        "T": 4,
        "B": 5,
        "NE": 6,
        "NW": 7,
        "SE": 8,
        "SW": 9,
        "NT": 10,
        "NB": 11,
        "ST": 12,
        "SB": 13,
        "ET": 14,
        "EB": 15,
        "WT": 16,
        "WB": 17,
    }

    stencil_array = np.full(
        (num_nodes, 18),
        -1,
        dtype=np.int32,
    )

    for node, stencil in stencils.items():
        for direction, column in direction_columns.items():
            neighbor = stencil[direction]
            if neighbor is not None:
                stencil_array[node, column] = neighbor

    is_movable = np.zeros(num_nodes, dtype=np.bool_)
    is_movable[list(movable_nodes)] = True

    move_x_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_y_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_z_mask = np.zeros(num_nodes, dtype=np.bool_)

    move_x_mask[list(set(move_all) | set(move_x_only))] = True
    move_y_mask[list(set(move_all) | set(move_y_only))] = True
    move_z_mask[list(set(move_all) | set(move_z_only))] = True

    x_coordinates = np.ascontiguousarray(points[:, 0])
    y_coordinates = np.ascontiguousarray(points[:, 1])
    z_coordinates = np.ascontiguousarray(points[:, 2])

    for iteration in range(1, iterations + 1):
        SpyroEnsemble.print(f"\rIteration: {iteration}/{iterations}", end="")

        sizing_values = np.asarray(
            sizing_function_xyz(
                x_coordinates,
                y_coordinates,
                z_coordinates,
                ef_segy,
                length_x,
                length_y,
                depth_z,
            ),
            dtype=float,
        )

        if sizing_values.shape != (num_nodes,):
            raise ValueError("sizing_function_xyz must return one value per mesh node.")
        if not np.all(np.isfinite(sizing_values)):
            raise ValueError("sizing_function_xyz returned NaN or infinity.")
        if np.any(sizing_values <= 0.0):
            raise ValueError("sizing_function_xyz returned a non-positive value.")

        (
            x_coordinates,
            y_coordinates,
            z_coordinates,
        ) = _numba_winslow_3d_step(
            x_coordinates,
            y_coordinates,
            z_coordinates,
            sizing_values,
            stencil_array,
            is_movable,
            move_x_mask,
            move_y_mask,
            move_z_mask,
            omega,
        )

    SpyroEnsemble.print("")

    SpyroEnsemble.print("Smoothing Complete.")

    return np.column_stack(
        (
            x_coordinates,
            y_coordinates,
            z_coordinates,
        )
    )
