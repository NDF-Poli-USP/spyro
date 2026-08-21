import sys

import numpy as np
from numba import njit, prange

from .meshing_utils3D import sizing_function_xyz


def run_selected_winslow(
    points,
    hexes,
    move_all,
    move_X_only,
    move_Y_only,
    move_Z_only,
    ef_segy,
    length_x,
    length_y,
    depth_z,
    winslow_iterations,
    winslow_omega,
    comm,
    parallel_print,
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
    move_X_only : set of int
        Nodes whose x coordinate may move.
    move_Y_only : set of int
        Nodes whose y coordinate may move.
    move_Z_only : set of int
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
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.
    selected_winslow : str
        Winslow implementation selector.

    Returns
    -------
    numpy.ndarray
        Smoothed node coordinates.
    """
    movable_nodes = (
        set(move_all) | set(move_X_only) | set(move_Y_only) | set(move_Z_only)
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

    parallel_print(
        "Winslow sizing on movable nodes | "
        f"min={size_min:.9g}, max={size_max:.9g}, "
        f"mean={size_mean:.9g}, std={size_std:.9g}",
        comm=comm,)
    parallel_print(
        "Winslow movable-node counts | "
        f"all={len(move_all)}, X={len(move_X_only)}, "
        f"Y={len(move_Y_only)}, Z={len(move_Z_only)}, "
        f"union={len(movable_nodes)}",
        comm=comm,)

    relative_span = size_span / max(abs(size_mean), 1.0)
    if relative_span <= 1.0e-10:
        parallel_print(
            "WARNING: the sizing field is effectively constant on the "
            "movable nodes. A uniform structured mesh will remain "
            "unchanged. Reduce hmin_segy; the supplied reference uses "
            "a 250 m initial mesh and no 500 m sizing floor.",
            comm=comm,)

    if selected_winslow != "numba":
        raise ValueError(
            "Only selected_winslow='numba' is currently supported. "
            f"Received {selected_winslow!r}."
        )

    parallel_print("Using winslow_smooth_3d55 (Numba 3D Winslow implementation).", comm=comm)
    smoothed = winslow_smooth_3d55(
        points=points,
        hexes=hexes,
        move_all=move_all,
        move_X_only=move_X_only,
        move_Y_only=move_Y_only,
        move_Z_only=move_Z_only,
        iterations=winslow_iterations,
        omega=winslow_omega,
        ef_segy=ef_segy,
        length_x=length_x,
        length_y=length_y,
        depth_z=depth_z,
        comm=comm,
        parallel_print=parallel_print,
    )

    displacement = np.linalg.norm(
        np.asarray(smoothed, dtype=float) - np.asarray(points, dtype=float),
        axis=1,
    )
    movable_displacement = displacement[movable_indices]

    moved_count = int(np.count_nonzero(movable_displacement > 1.0e-10))
    max_displacement = float(np.max(movable_displacement))
    mean_displacement = float(np.mean(movable_displacement))

    parallel_print(
        "Winslow displacement | "
        f"moved={moved_count}/{len(movable_indices)}, "
        f"max={max_displacement:.9g}, "
        f"mean={mean_displacement:.9g}",
        comm=comm,)

    if moved_count == 0:
        parallel_print(
            "WARNING: winslow_smooth_3d55 completed but no movable node "
            "changed position. Inspect the sizing statistics above.",
            comm=comm,)

    return smoothed


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

    east_value = (
        coordinates[east] if east != -1
        else 2.0 * center - coordinates[west] if west != -1
        else center
    )
    west_value = (
        coordinates[west] if west != -1
        else 2.0 * center - coordinates[east] if east != -1
        else center
    )
    north_value = (
        coordinates[north] if north != -1
        else 2.0 * center - coordinates[south] if south != -1
        else center
    )
    south_value = (
        coordinates[south] if south != -1
        else 2.0 * center - coordinates[north] if north != -1
        else center
    )
    top_value = (
        coordinates[top] if top != -1
        else 2.0 * center - coordinates[bottom] if bottom != -1
        else center
    )
    bottom_value = (
        coordinates[bottom] if bottom != -1
        else 2.0 * center - coordinates[top] if top != -1
        else center
    )

    ne, nw, se, sw, nt, nb, st, sb, et, eb, wt, wb = stencil[6:]

    ne_value = coordinates[ne] if ne != -1 else north_value + east_value - center
    nw_value = coordinates[nw] if nw != -1 else north_value + west_value - center
    se_value = coordinates[se] if se != -1 else south_value + east_value - center
    sw_value = coordinates[sw] if sw != -1 else south_value + west_value - center
    nt_value = coordinates[nt] if nt != -1 else north_value + top_value - center
    nb_value = coordinates[nb] if nb != -1 else north_value + bottom_value - center
    st_value = coordinates[st] if st != -1 else south_value + top_value - center
    sb_value = coordinates[sb] if sb != -1 else south_value + bottom_value - center
    et_value = coordinates[et] if et != -1 else east_value + top_value - center
    eb_value = coordinates[eb] if eb != -1 else east_value + bottom_value - center
    wt_value = coordinates[wt] if wt != -1 else west_value + top_value - center
    wb_value = coordinates[wb] if wb != -1 else west_value + bottom_value - center

    return (
        east_value, west_value, north_value, south_value, top_value, bottom_value,
        ne_value, nw_value, se_value, sw_value, nt_value, nb_value,
        st_value, sb_value, et_value, eb_value, wt_value, wb_value,
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

        if not (
            move_x[node]
            or move_y[node]
            or move_z[node]
        ):
            continue

        x_center = x_coordinates[node]
        y_center = y_coordinates[node]
        z_center = z_coordinates[node]

        stencil = stencils[node]
        east, west, north, south, top, bottom = stencil[:6]

        (
            x_east, x_west, x_north, x_south, x_top, x_bottom,
            x_ne, x_nw, x_se, x_sw, x_nt, x_nb,
            x_st, x_sb, x_et, x_eb, x_wt, x_wb,
        ) = _resolve_stencil_coordinates(x_coordinates, x_center, stencil)

        (
            y_east, y_west, y_north, y_south, y_top, y_bottom,
            y_ne, y_nw, y_se, y_sw, y_nt, y_nb,
            y_st, y_sb, y_et, y_eb, y_wt, y_wb,
        ) = _resolve_stencil_coordinates(y_coordinates, y_center, stencil)

        (
            z_east, z_west, z_north, z_south, z_top, z_bottom,
            z_ne, z_nw, z_se, z_sw, z_nt, z_nb,
            z_st, z_sb, z_et, z_eb, z_wt, z_wb,
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
        size_east = sizing_values[east] if east != -1 and is_movable[east] else size_center
        size_west = sizing_values[west] if west != -1 and is_movable[west] else size_center
        size_north = sizing_values[north] if north != -1 and is_movable[north] else size_center
        size_south = sizing_values[south] if south != -1 and is_movable[south] else size_center
        size_top = sizing_values[top] if top != -1 and is_movable[top] else size_center
        size_bottom = sizing_values[bottom] if bottom != -1 and is_movable[bottom] else size_center

        size_xi = 0.5 * (size_east - size_west)
        size_eta = 0.5 * (size_north - size_south)
        size_zeta = 0.5 * (size_top - size_bottom)

        jacobian = x_xi * (y_eta * z_zeta - z_eta * y_zeta) - y_xi * (x_eta * z_zeta - z_eta * x_zeta) \
            + z_xi * (x_eta * y_zeta - y_eta * x_zeta)
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

        source_x = jacobian_over_size * (size_xi * adjoint_11 + size_eta * adjoint_21 + size_zeta * adjoint_31)
        source_y = jacobian_over_size * (size_xi * adjoint_12 + size_eta * adjoint_22 + size_zeta * adjoint_32)
        source_z = jacobian_over_size * (size_xi * adjoint_13 + size_eta * adjoint_23 + size_zeta * adjoint_33)

        denominator = 2.0 * (coefficient_a + coefficient_b + coefficient_c)
        if denominator < 1.0e-12:
            continue

        if move_x[node]:
            x_target = (coefficient_a * (x_east + x_west) + coefficient_b * (x_north + x_south) + coefficient_c * (x_top + x_bottom)
                        + 2.0 * (coefficient_d * x_xi_eta + coefficient_e * x_eta_zeta + coefficient_f * x_xi_zeta) - source_x) / denominator
            x_new[node] = (1.0 - omega) * x_center + omega * x_target

        if move_y[node]:
            y_target = (coefficient_a * (y_east + y_west) + coefficient_b * (y_north + y_south) + coefficient_c * (y_top + y_bottom)
                        + 2.0 * (coefficient_d * y_xi_eta + coefficient_e * y_eta_zeta + coefficient_f * y_xi_zeta) - source_y) / denominator
            y_new[node] = (1.0 - omega) * y_center + omega * y_target

        if move_z[node]:
            z_target = (coefficient_a * (z_east + z_west) + coefficient_b * (z_north + z_south) + coefficient_c * (z_top + z_bottom)
                        + 2.0 * (coefficient_d * z_xi_eta + coefficient_e * z_eta_zeta + coefficient_f * z_xi_zeta) - source_z) / denominator
            z_new[node] = (1.0 - omega) * z_center + omega * z_target

    return x_new, y_new, z_new


def winslow_smooth_3d55(
    points: np.ndarray,
    hexes: np.ndarray,
    move_all: set[int],
    move_X_only: set[int],
    move_Y_only: set[int],
    move_Z_only: set[int],
    ef_segy,
    length_x,
    length_y,
    depth_z,
    comm,
    parallel_print,
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
    move_X_only : set of int
        Nodes whose x coordinate may move.
    move_Y_only : set of int
        Nodes whose y coordinate may move.
    move_Z_only : set of int
        Nodes whose z coordinate may move.
    ef_segy : callable
        Mesh-sizing function evaluated in ``(z, x, y)`` coordinates.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.
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
        raise ValueError(
            "points must have shape (number_of_nodes, 3)."
        )
    if hexes.ndim != 2 or hexes.shape[1] != 8:
        raise ValueError(
            "hexes must have shape (number_of_hexes, 8)."
        )
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if not 0.0 < omega <= 1.0:
        raise ValueError("omega must satisfy 0 < omega <= 1.")

    num_nodes = len(points)
    node_to_neighbors = {
        node: set()
        for node in range(num_nodes)
    }

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
        set(move_all)
        | set(move_X_only)
        | set(move_Y_only)
        | set(move_Z_only)
    )

    invalid_nodes = {
        node
        for node in movable_nodes
        if node < 0 or node >= num_nodes
    }
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

        def get_shared_neighbor(
            first_neighbor,
            second_neighbor,
        ):
            """Return the common neighbor associated with a mixed-direction stencil entry.

            Parameters
            ----------
            first_neighbor : int or None
                First cardinal-neighbor node index.
            second_neighbor : int or None
                Second cardinal-neighbor node index.

            Returns
            -------
            int or None
                Shared neighbor index, or ``None`` when no shared neighbor exists.
            """
            if (
                first_neighbor is None
                or second_neighbor is None
            ):
                return None

            shared = node_to_neighbors[
                first_neighbor
            ].intersection(
                node_to_neighbors[second_neighbor]
            )
            shared.discard(node)  # noqa: B023

            if not shared:
                return None

            return min(shared)

        corners = {
            "NE": get_shared_neighbor(cardinals["N"], cardinals["E"]),
            "NW": get_shared_neighbor(cardinals["N"], cardinals["W"]),
            "SE": get_shared_neighbor(cardinals["S"], cardinals["E"]),
            "SW": get_shared_neighbor(cardinals["S"], cardinals["W"]),
            "NT": get_shared_neighbor(cardinals["N"], cardinals["T"]),
            "NB": get_shared_neighbor(cardinals["N"], cardinals["B"]),
            "ST": get_shared_neighbor(cardinals["S"], cardinals["T"]),
            "SB": get_shared_neighbor(cardinals["S"], cardinals["B"]),
            "ET": get_shared_neighbor(cardinals["E"], cardinals["T"]),
            "EB": get_shared_neighbor(cardinals["E"], cardinals["B"]),
            "WT": get_shared_neighbor(cardinals["W"], cardinals["T"]),
            "WB": get_shared_neighbor(cardinals["W"], cardinals["B"]),
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

    move_x_mask[list(set(move_all) | set(move_X_only))] = True
    move_y_mask[list(set(move_all) | set(move_Y_only))] = True
    move_z_mask[list(set(move_all) | set(move_Z_only))] = True

    x_coordinates = np.ascontiguousarray(points[:, 0])
    y_coordinates = np.ascontiguousarray(points[:, 1])
    z_coordinates = np.ascontiguousarray(points[:, 2])

    if comm is None:
        progress_rank = 0
    else:
        progress_comm = getattr(comm, "comm", comm)
        progress_rank = getattr(progress_comm, "rank", 0)

    for iteration in range(1, iterations + 1):
        if progress_rank == 0:
            sys.stdout.write(f"\rIteration: {iteration}/{iterations}")
            sys.stdout.flush()

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
            raise ValueError(
                "sizing_function_xyz must return one value per mesh node."
            )
        if not np.all(np.isfinite(sizing_values)):
            raise ValueError(
                "sizing_function_xyz returned NaN or infinity."
            )
        if np.any(sizing_values <= 0.0):
            raise ValueError(
                "sizing_function_xyz returned a non-positive value."
            )

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

    if progress_rank == 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    parallel_print("Smoothing Complete.", comm=comm)

    return np.column_stack(
        (
            x_coordinates,
            y_coordinates,
            z_coordinates,
        )
    )
