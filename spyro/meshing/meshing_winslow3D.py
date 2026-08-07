from collections.abc import Callable
import numpy as np
from numba import njit, prange


__all__ = ["winslow_smooth_3d55"]


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
    """Perform one parallel Winslow smoothing iteration."""
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

        east = stencils[node, 0]
        west = stencils[node, 1]
        north = stencils[node, 2]
        south = stencils[node, 3]
        top = stencils[node, 4]
        bottom = stencils[node, 5]

        north_east = stencils[node, 6]
        north_west = stencils[node, 7]
        south_east = stencils[node, 8]
        south_west = stencils[node, 9]

        north_top = stencils[node, 10]
        north_bottom = stencils[node, 11]
        south_top = stencils[node, 12]
        south_bottom = stencils[node, 13]

        east_top = stencils[node, 14]
        east_bottom = stencils[node, 15]
        west_top = stencils[node, 16]
        west_bottom = stencils[node, 17]

        x_center = x_coordinates[node]
        y_center = y_coordinates[node]
        z_center = z_coordinates[node]

        x_east = (
            x_coordinates[east]
            if east != -1
            else (
                2.0 * x_center - x_coordinates[west]
                if west != -1
                else x_center
            )
        )
        x_west = (
            x_coordinates[west]
            if west != -1
            else (
                2.0 * x_center - x_coordinates[east]
                if east != -1
                else x_center
            )
        )
        x_north = (
            x_coordinates[north]
            if north != -1
            else (
                2.0 * x_center - x_coordinates[south]
                if south != -1
                else x_center
            )
        )
        x_south = (
            x_coordinates[south]
            if south != -1
            else (
                2.0 * x_center - x_coordinates[north]
                if north != -1
                else x_center
            )
        )
        x_top = (
            x_coordinates[top]
            if top != -1
            else (
                2.0 * x_center - x_coordinates[bottom]
                if bottom != -1
                else x_center
            )
        )
        x_bottom = (
            x_coordinates[bottom]
            if bottom != -1
            else (
                2.0 * x_center - x_coordinates[top]
                if top != -1
                else x_center
            )
        )

        y_east = (
            y_coordinates[east]
            if east != -1
            else (
                2.0 * y_center - y_coordinates[west]
                if west != -1
                else y_center
            )
        )
        y_west = (
            y_coordinates[west]
            if west != -1
            else (
                2.0 * y_center - y_coordinates[east]
                if east != -1
                else y_center
            )
        )
        y_north = (
            y_coordinates[north]
            if north != -1
            else (
                2.0 * y_center - y_coordinates[south]
                if south != -1
                else y_center
            )
        )
        y_south = (
            y_coordinates[south]
            if south != -1
            else (
                2.0 * y_center - y_coordinates[north]
                if north != -1
                else y_center
            )
        )
        y_top = (
            y_coordinates[top]
            if top != -1
            else (
                2.0 * y_center - y_coordinates[bottom]
                if bottom != -1
                else y_center
            )
        )
        y_bottom = (
            y_coordinates[bottom]
            if bottom != -1
            else (
                2.0 * y_center - y_coordinates[top]
                if top != -1
                else y_center
            )
        )

        z_east = (
            z_coordinates[east]
            if east != -1
            else (
                2.0 * z_center - z_coordinates[west]
                if west != -1
                else z_center
            )
        )
        z_west = (
            z_coordinates[west]
            if west != -1
            else (
                2.0 * z_center - z_coordinates[east]
                if east != -1
                else z_center
            )
        )
        z_north = (
            z_coordinates[north]
            if north != -1
            else (
                2.0 * z_center - z_coordinates[south]
                if south != -1
                else z_center
            )
        )
        z_south = (
            z_coordinates[south]
            if south != -1
            else (
                2.0 * z_center - z_coordinates[north]
                if north != -1
                else z_center
            )
        )
        z_top = (
            z_coordinates[top]
            if top != -1
            else (
                2.0 * z_center - z_coordinates[bottom]
                if bottom != -1
                else z_center
            )
        )
        z_bottom = (
            z_coordinates[bottom]
            if bottom != -1
            else (
                2.0 * z_center - z_coordinates[top]
                if top != -1
                else z_center
            )
        )

        x_ne = (
            x_coordinates[north_east]
            if north_east != -1
            else x_north + x_east - x_center
        )
        x_nw = (
            x_coordinates[north_west]
            if north_west != -1
            else x_north + x_west - x_center
        )
        x_se = (
            x_coordinates[south_east]
            if south_east != -1
            else x_south + x_east - x_center
        )
        x_sw = (
            x_coordinates[south_west]
            if south_west != -1
            else x_south + x_west - x_center
        )
        x_nt = (
            x_coordinates[north_top]
            if north_top != -1
            else x_north + x_top - x_center
        )
        x_nb = (
            x_coordinates[north_bottom]
            if north_bottom != -1
            else x_north + x_bottom - x_center
        )
        x_st = (
            x_coordinates[south_top]
            if south_top != -1
            else x_south + x_top - x_center
        )
        x_sb = (
            x_coordinates[south_bottom]
            if south_bottom != -1
            else x_south + x_bottom - x_center
        )
        x_et = (
            x_coordinates[east_top]
            if east_top != -1
            else x_east + x_top - x_center
        )
        x_eb = (
            x_coordinates[east_bottom]
            if east_bottom != -1
            else x_east + x_bottom - x_center
        )
        x_wt = (
            x_coordinates[west_top]
            if west_top != -1
            else x_west + x_top - x_center
        )
        x_wb = (
            x_coordinates[west_bottom]
            if west_bottom != -1
            else x_west + x_bottom - x_center
        )

        y_ne = (
            y_coordinates[north_east]
            if north_east != -1
            else y_north + y_east - y_center
        )
        y_nw = (
            y_coordinates[north_west]
            if north_west != -1
            else y_north + y_west - y_center
        )
        y_se = (
            y_coordinates[south_east]
            if south_east != -1
            else y_south + y_east - y_center
        )
        y_sw = (
            y_coordinates[south_west]
            if south_west != -1
            else y_south + y_west - y_center
        )
        y_nt = (
            y_coordinates[north_top]
            if north_top != -1
            else y_north + y_top - y_center
        )
        y_nb = (
            y_coordinates[north_bottom]
            if north_bottom != -1
            else y_north + y_bottom - y_center
        )
        y_st = (
            y_coordinates[south_top]
            if south_top != -1
            else y_south + y_top - y_center
        )
        y_sb = (
            y_coordinates[south_bottom]
            if south_bottom != -1
            else y_south + y_bottom - y_center
        )
        y_et = (
            y_coordinates[east_top]
            if east_top != -1
            else y_east + y_top - y_center
        )
        y_eb = (
            y_coordinates[east_bottom]
            if east_bottom != -1
            else y_east + y_bottom - y_center
        )
        y_wt = (
            y_coordinates[west_top]
            if west_top != -1
            else y_west + y_top - y_center
        )
        y_wb = (
            y_coordinates[west_bottom]
            if west_bottom != -1
            else y_west + y_bottom - y_center
        )

        z_ne = (
            z_coordinates[north_east]
            if north_east != -1
            else z_north + z_east - z_center
        )
        z_nw = (
            z_coordinates[north_west]
            if north_west != -1
            else z_north + z_west - z_center
        )
        z_se = (
            z_coordinates[south_east]
            if south_east != -1
            else z_south + z_east - z_center
        )
        z_sw = (
            z_coordinates[south_west]
            if south_west != -1
            else z_south + z_west - z_center
        )
        z_nt = (
            z_coordinates[north_top]
            if north_top != -1
            else z_north + z_top - z_center
        )
        z_nb = (
            z_coordinates[north_bottom]
            if north_bottom != -1
            else z_north + z_bottom - z_center
        )
        z_st = (
            z_coordinates[south_top]
            if south_top != -1
            else z_south + z_top - z_center
        )
        z_sb = (
            z_coordinates[south_bottom]
            if south_bottom != -1
            else z_south + z_bottom - z_center
        )
        z_et = (
            z_coordinates[east_top]
            if east_top != -1
            else z_east + z_top - z_center
        )
        z_eb = (
            z_coordinates[east_bottom]
            if east_bottom != -1
            else z_east + z_bottom - z_center
        )
        z_wt = (
            z_coordinates[west_top]
            if west_top != -1
            else z_west + z_top - z_center
        )
        z_wb = (
            z_coordinates[west_bottom]
            if west_bottom != -1
            else z_west + z_bottom - z_center
        )

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
        size_east = (
            sizing_values[east]
            if east != -1 and is_movable[east]
            else size_center
        )
        size_west = (
            sizing_values[west]
            if west != -1 and is_movable[west]
            else size_center
        )
        size_north = (
            sizing_values[north]
            if north != -1 and is_movable[north]
            else size_center
        )
        size_south = (
            sizing_values[south]
            if south != -1 and is_movable[south]
            else size_center
        )
        size_top = (
            sizing_values[top]
            if top != -1 and is_movable[top]
            else size_center
        )
        size_bottom = (
            sizing_values[bottom]
            if bottom != -1 and is_movable[bottom]
            else size_center
        )

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
            size_xi * adjoint_11
            + size_eta * adjoint_21
            + size_zeta * adjoint_31
        )
        source_y = jacobian_over_size * (
            size_xi * adjoint_12
            + size_eta * adjoint_22
            + size_zeta * adjoint_32
        )
        source_z = jacobian_over_size * (
            size_xi * adjoint_13
            + size_eta * adjoint_23
            + size_zeta * adjoint_33
        )

        denominator = 2.0 * (
            coefficient_a
            + coefficient_b
            + coefficient_c
        )
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
            x_new[node] = (
                (1.0 - omega) * x_center
                + omega * x_target
            )

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
            y_new[node] = (
                (1.0 - omega) * y_center
                + omega * y_target
            )

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
            z_new[node] = (
                (1.0 - omega) * z_center
                + omega * z_target
            )

    return x_new, y_new, z_new


def winslow_smooth_3d55(
    points: np.ndarray,
    hexes: np.ndarray,
    sizing_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ],
    move_all: set[int],
    move_X_only: set[int],
    move_Y_only: set[int],
    move_Z_only: set[int],
    iterations: int = 50,
    omega: float = 0.5,
) -> np.ndarray:
    """Smooth a structured hexahedral mesh using the 3-D Winslow equations.

    Parameters
    ----------
    points
        Node coordinates with shape ``(number_of_nodes, 3)``.
    hexes
        Hexahedral connectivity with eight node indices per element.
    sizing_fn
        Callable evaluated as ``sizing_fn(x, y, z)``.
    move_all
        Nodes allowed to move in all coordinate directions.
    move_X_only
        Nodes allowed to move only in the x direction.
    move_Y_only
        Nodes allowed to move only in the y direction.
    move_Z_only
        Nodes allowed to move only in the z direction.
    iterations
        Number of Winslow iterations.
    omega
        Relaxation coefficient.

    Returns
    -------
    numpy.ndarray
        Smoothed node coordinates.
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
            "NE": get_shared_neighbor(
                cardinals["N"],
                cardinals["E"],
            ),
            "NW": get_shared_neighbor(
                cardinals["N"],
                cardinals["W"],
            ),
            "SE": get_shared_neighbor(
                cardinals["S"],
                cardinals["E"],
            ),
            "SW": get_shared_neighbor(
                cardinals["S"],
                cardinals["W"],
            ),
            "NT": get_shared_neighbor(
                cardinals["N"],
                cardinals["T"],
            ),
            "NB": get_shared_neighbor(
                cardinals["N"],
                cardinals["B"],
            ),
            "ST": get_shared_neighbor(
                cardinals["S"],
                cardinals["T"],
            ),
            "SB": get_shared_neighbor(
                cardinals["S"],
                cardinals["B"],
            ),
            "ET": get_shared_neighbor(
                cardinals["E"],
                cardinals["T"],
            ),
            "EB": get_shared_neighbor(
                cardinals["E"],
                cardinals["B"],
            ),
            "WT": get_shared_neighbor(
                cardinals["W"],
                cardinals["T"],
            ),
            "WB": get_shared_neighbor(
                cardinals["W"],
                cardinals["B"],
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

    move_x_mask[list(set(move_all) | set(move_X_only))] = True
    move_y_mask[list(set(move_all) | set(move_Y_only))] = True
    move_z_mask[list(set(move_all) | set(move_Z_only))] = True

    x_coordinates = np.ascontiguousarray(points[:, 0])
    y_coordinates = np.ascontiguousarray(points[:, 1])
    z_coordinates = np.ascontiguousarray(points[:, 2])

    for iteration in range(1, iterations + 1):
        print(
            f"Iteration: {iteration}/{iterations}",
            end="\r",
        )

        sizing_values = np.asarray(
            sizing_fn(
                x_coordinates,
                y_coordinates,
                z_coordinates,
            ),
            dtype=float,
        )

        if sizing_values.shape != (num_nodes,):
            raise ValueError(
                "sizing_fn must return one value per mesh node."
            )
        if not np.all(np.isfinite(sizing_values)):
            raise ValueError(
                "sizing_fn returned NaN or infinity."
            )
        if np.any(sizing_values <= 0.0):
            raise ValueError(
                "sizing_fn returned a non-positive value."
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

    print("\nSmoothing Complete.")

    return np.column_stack(
        (
            x_coordinates,
            y_coordinates,
            z_coordinates,
        )
    )
