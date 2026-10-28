import numpy as np
from typing import Callable, Tuple

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True  # pragma: no cover
except ImportError:
    NUMBA_AVAILABLE = False


def winslow_smooth_default(
    points: np.ndarray,
    quads: np.ndarray,
    sizing_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    move_all: set,
    move_x_only: set,
    move_z_only: set,
    move_hyperellipse: set = None,
    hyperellipse_params: tuple = None,
    iterations: int = 1500,
    omega: float = 0.05,
    plot_sources: bool = True,
) -> np.ndarray:
    """Apply Winslow mesh smoothing using a default Python implementation.

    Uses finite difference approximations to solve elliptic generation equations
    and iteratively move internal nodes, adapting element sizing and shapes
    to the provided sizing function.

    Parameters
    ----------
    points : ndarray
        2D array of shape (N, 2) containing node coordinates [x, z].
    quads : ndarray
        2D array of shape (M, 4) containing node indices for each quadrilateral.
    sizing_fn : callable
        Function f(X, Z) returning the target element size at coordinates X, Z.
    move_all : set
        Set of node indices permitted to move in both X and Z directions.
    move_x_only : set
        Set of node indices constrained to move only along the X-axis.
    move_z_only : set
        Set of node indices constrained to move only along the Z-axis.
    move_hyperellipse : set, optional
        Set of node indices constrained to move along an elliptical boundary.
    hyperellipse_params : tuple, optional
        Tuple containing (a, b, xc, zc, n) defining an superhyperellipse boundary.
    iterations : int, optional
        Total number of smoothing iterations. Default is 1500.
    omega : float, optional
        Relaxation factor for node movement (0 < omega <= 1). Default is 0.05.
    plot_sources : bool, optional
        Placeholder flag for enabling source tracking or plotting. Default is True.

    Returns
    -------
    ndarray
        Updated node coordinates array of shape (N, 2).
    """
    if move_hyperellipse is None:
        move_hyperellipse = set()

    num_nodes = len(points)

    node_to_neighbors = {i: set() for i in range(num_nodes)}
    for quad in quads:
        for i in range(4):
            start_node = quad[i]
            end_node = quad[(i + 1) % 4]
            node_to_neighbors[start_node].add(end_node)
            node_to_neighbors[end_node].add(start_node)

    stencils = {}

    for i in range(num_nodes):
        neighbors = list(node_to_neighbors[i])
        if not neighbors:
            continue

        vecs = points[neighbors] - points[i]
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        sort_idx = np.argsort(angles)
        sorted_neighbors = [neighbors[idx] for idx in sort_idx]
        sorted_angles = angles[sort_idx]
        cardinals = {"E": None, "N": None, "W": None, "S": None}

        if len(sorted_neighbors) == 4:
            cardinals["E"] = sorted_neighbors[0]
            cardinals["N"] = sorted_neighbors[1]
            cardinals["W"] = sorted_neighbors[2]
            cardinals["S"] = sorted_neighbors[3]
        elif len(sorted_neighbors) > 0:
            diffs = np.diff(sorted_angles)
            diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2 * np.pi)
            gap_idx = np.argmax(diffs)
            rolled_neighbors = np.roll(sorted_neighbors, -(gap_idx + 1))

            if len(sorted_neighbors) == 3:
                cardinals["E"] = rolled_neighbors[0]
                cardinals["N"] = rolled_neighbors[1]
                cardinals["W"] = rolled_neighbors[2]
            elif len(sorted_neighbors) == 2:
                cardinals["E"] = rolled_neighbors[0]
                cardinals["N"] = rolled_neighbors[1]

        def get_shared(first_neighbor, second_neighbor):
            if first_neighbor is None or second_neighbor is None:
                return None
            shared = node_to_neighbors[first_neighbor].intersection(node_to_neighbors[second_neighbor])
            shared.discard(i)
            return list(shared)[0] if shared else None

        corners = {
            "NE": get_shared(cardinals["N"], cardinals["E"]),
            "NW": get_shared(cardinals["N"], cardinals["W"]),
            "SE": get_shared(cardinals["S"], cardinals["E"]),
            "SW": get_shared(cardinals["S"], cardinals["W"]),
        }
        stencils[i] = {**cardinals, **corners}

    x = points[:, 0].copy()
    z = points[:, 1].copy()
    x_new = x.copy()
    z_new = z.copy()
    movable_nodes = move_all | move_x_only | move_z_only | move_hyperellipse
    jacobian_over_size_values = np.zeros(num_nodes)
    iteration = 0
    for _ in range(iterations):
        iteration += 1
        print(f"Iteration: {iteration}", end="\r")
        sizing_values = sizing_fn(x, z)

        for i, stencil in stencils.items():
            is_all = i in move_all
            is_x_only = i in move_x_only
            is_z_only = i in move_z_only
            is_hyperellipse = i in move_hyperellipse

            if not (is_all or is_x_only or is_z_only or is_hyperellipse):
                continue

            stencil_coords = {"X": {}, "Z": {}}

            for direction, opposite in [("E", "W"), ("W", "E"), ("N", "S"), ("S", "N")]:
                idx = stencil[direction]
                if idx is not None:
                    stencil_coords["X"][direction] = x[idx]
                    stencil_coords["Z"][direction] = z[idx]
                else:
                    opp_idx = stencil[opposite]
                    stencil_coords["X"][direction] = 2 * x[i] - x[opp_idx] if opp_idx is not None else x[i]
                    stencil_coords["Z"][direction] = 2 * z[i] - z[opp_idx] if opp_idx is not None else z[i]

            for direction, first_cardinal, second_cardinal in [
                ("NE", "N", "E"),
                ("NW", "N", "W"),
                ("SE", "S", "E"),
                ("SW", "S", "W"),
            ]:
                idx = stencil[direction]
                if idx is not None:
                    stencil_coords["X"][direction] = x[idx]
                    stencil_coords["Z"][direction] = z[idx]
                else:
                    stencil_coords["X"][direction] = stencil_coords["X"][first_cardinal] + stencil_coords["X"][second_cardinal] - x[i]
                    stencil_coords["Z"][direction] = stencil_coords["Z"][first_cardinal] + stencil_coords["Z"][second_cardinal] - z[i]

            x_e, x_w, x_n, x_s = stencil_coords["X"]["E"], stencil_coords["X"]["W"], stencil_coords["X"]["N"], stencil_coords["X"]["S"]
            z_e, z_w, z_n, z_s = stencil_coords["Z"]["E"], stencil_coords["Z"]["W"], stencil_coords["Z"]["N"], stencil_coords["Z"]["S"]

            x_xi = 0.5 * (x_e - x_w)
            x_eta = 0.5 * (x_n - x_s)
            z_xi = 0.5 * (z_e - z_w)
            z_eta = 0.5 * (z_n - z_s)
            alpha = x_eta**2 + z_eta**2
            beta = x_xi * x_eta + z_xi * z_eta
            gamma = x_xi**2 + z_xi**2
            x_xita = 0.25 * (stencil_coords["X"]["NE"] - stencil_coords["X"]["NW"] - stencil_coords["X"]["SE"] + stencil_coords["X"]["SW"])
            z_xita = 0.25 * (stencil_coords["Z"]["NE"] - stencil_coords["Z"]["NW"] - stencil_coords["Z"]["SE"] + stencil_coords["Z"]["SW"])

            size_center = sizing_values[i]
            size_east = (
                sizing_values[stencil["E"]]
                if (stencil["E"] is not None and stencil["E"] in movable_nodes)
                else size_center
            )
            size_west = (
                sizing_values[stencil["W"]]
                if (stencil["W"] is not None and stencil["W"] in movable_nodes)
                else size_center
            )
            size_north = (
                sizing_values[stencil["N"]]
                if (stencil["N"] is not None and stencil["N"] in movable_nodes)
                else size_center
            )
            size_south = (
                sizing_values[stencil["S"]]
                if (stencil["S"] is not None and stencil["S"] in movable_nodes)
                else size_center
            )

            size_xi = 0.5 * (size_east - size_west)
            size_eta = 0.5 * (size_north - size_south)
            jacobian = x_xi * z_eta - x_eta * z_xi
            jacobian_over_size_values[i] = jacobian / (size_center + 1e-12)
            source_x = (jacobian / (size_center + 1e-12)) * (size_xi * z_eta - size_eta * z_xi)
            source_z = (jacobian / (size_center + 1e-12)) * (size_eta * x_xi - size_xi * x_eta)
            denominator = 2.0 * (alpha + gamma)

            if denominator < 1e-12:
                continue

            x_target = x[i]
            z_target = z[i]

            if is_all or is_x_only or is_hyperellipse:
                x_target = (
                    alpha * (x_e + x_w)
                    + gamma * (x_n + x_s)
                    - 2.0 * beta * x_xita
                    - source_x
                ) / denominator

            if is_all or is_z_only or is_hyperellipse:
                z_target = (
                    alpha * (z_e + z_w)
                    + gamma * (z_n + z_s)
                    - 2.0 * beta * z_xita
                    - source_z
                ) / denominator

            if is_hyperellipse and hyperellipse_params is not None:
                a_e, b_e, xc_e, zc_e, n_e = hyperellipse_params
                dx = x_target - xc_e
                dz = z_target - zc_e

                if abs(dx) < 1e-12 and abs(dz) < 1e-12:
                    x_new[i] = x[i]
                    z_new[i] = z[i]
                else:
                    scale = (abs(dx / a_e) ** n_e + abs(dz / b_e) ** n_e) ** (
                        -1.0 / n_e
                    )
                    x_proj = xc_e + scale * dx
                    z_proj = zc_e + scale * dz

                    x_new[i] = (1.0 - omega) * x[i] + omega * x_proj
                    z_new[i] = (1.0 - omega) * z[i] + omega * z_proj

            else:
                if is_all or is_x_only:
                    x_new[i] = (1.0 - omega) * x[i] + omega * x_target

                if is_all or is_z_only:
                    z_new[i] = (1.0 - omega) * z[i] + omega * z_target

        x[:] = x_new[:]
        z[:] = z_new[:]

    return np.column_stack((x, z))


if NUMBA_AVAILABLE:  # pragma: no cover

    @njit(cache=True)
    def bilinear_interp_2d(
        x: float,
        z: float,
        grid_x: np.ndarray,
        grid_z: np.ndarray,
        grid_vals: np.ndarray,
    ) -> float:
        """Perform 2D bilinear interpolation for a single point.

        Parameters
        ----------
        x : float
            Target x-coordinate for interpolation.
        z : float
            Target z-coordinate for interpolation.
        grid_x : ndarray
            Monotonically increasing or decreasing x-coordinates of the grid.
        grid_z : ndarray
            Monotonically increasing or decreasing z-coordinates of the grid.
        grid_vals : ndarray
            Values indexed as ``grid_vals[x_index, z_index]``, with shape
            ``(len(grid_x), len(grid_z))``. The transposed layout is also
            accepted for non-square grids. Square grids always use x/z order.

        Returns
        -------
        float
            The bilinearly interpolated value at (x, z).

        Raises
        ------
        ValueError
            If grid dimensions are too small, shape mismatches occur, or
            degenerate coordinate domains are passed.
        """
        nx = grid_x.shape[0]
        nz = grid_z.shape[0]

        if nx < 2 or nz < 2:
            raise ValueError("grid_x and grid_z must each have at least 2 points.")

        gv0, gv1 = grid_vals.shape
        vals_are_nx_nz = gv0 == nx and gv1 == nz
        vals_are_nz_nx = gv0 == nz and gv1 == nx
        if not vals_are_nx_nz and not vals_are_nz_nx:
            raise ValueError(
                "grid_vals shape must be (len(grid_x), len(grid_z)) or (len(grid_z), len(grid_x))."
            )

        x_asc = grid_x[0] < grid_x[nx - 1]
        z_asc = grid_z[0] < grid_z[nz - 1]

        if grid_x[0] == grid_x[nx - 1]:
            raise ValueError("grid_x domain is degenerate.")
        if grid_z[0] == grid_z[nz - 1]:
            raise ValueError("grid_z domain is degenerate.")

        x_min = grid_x[0] if x_asc else grid_x[nx - 1]
        x_max = grid_x[nx - 1] if x_asc else grid_x[0]
        z_min = grid_z[0] if z_asc else grid_z[nz - 1]
        z_max = grid_z[nz - 1] if z_asc else grid_z[0]

        if x < x_min:
            xq = x_min
        elif x > x_max:
            xq = x_max
        else:
            xq = x

        if z < z_min:
            zq = z_min
        elif z > z_max:
            zq = z_max
        else:
            zq = z

        if x_asc:
            i = np.searchsorted(grid_x, xq, side="right") - 1
        else:
            lo = 0
            hi = nx
            while lo < hi:
                mid = (lo + hi) // 2
                val = grid_x[nx - 1 - mid]
                if val < xq or val == xq:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo - 1

        if z_asc:
            j = np.searchsorted(grid_z, zq, side="right") - 1
        else:
            lo = 0
            hi = nz
            while lo < hi:
                mid = (lo + hi) // 2
                val = grid_z[nz - 1 - mid]
                if val < zq or val == zq:
                    lo = mid + 1
                else:
                    hi = mid
            j = lo - 1

        if i < 0:
            i = 0
        elif i > nx - 2:
            i = nx - 2

        if j < 0:
            j = 0
        elif j > nz - 2:
            j = nz - 2

        if x_asc:
            x0 = grid_x[i]
            x1 = grid_x[i + 1]
            ip0 = i
            ip1 = i + 1
        else:
            ip0 = nx - 1 - (i + 1)
            ip1 = nx - 1 - i
            x0 = grid_x[ip0]
            x1 = grid_x[ip1]

        if z_asc:
            z0 = grid_z[j]
            z1 = grid_z[j + 1]
            jp0 = j
            jp1 = j + 1
        else:
            jp0 = nz - 1 - (j + 1)
            jp1 = nz - 1 - j
            z0 = grid_z[jp0]
            z1 = grid_z[jp1]

        dx = x1 - x0
        dz = z1 - z0

        xd = (xq - x0) / dx if dx != 0.0 else 0.0
        zd = (zq - z0) / dz if dz != 0.0 else 0.0

        if xd < 0.0:
            xd = 0.0
        elif xd > 1.0:
            xd = 1.0

        if zd < 0.0:
            zd = 0.0
        elif zd > 1.0:
            zd = 1.0

        if vals_are_nx_nz:
            c00 = grid_vals[ip0, jp0]
            c10 = grid_vals[ip1, jp0]
            c01 = grid_vals[ip0, jp1]
            c11 = grid_vals[ip1, jp1]
        else:
            c00 = grid_vals[jp0, ip0]
            c10 = grid_vals[jp0, ip1]
            c01 = grid_vals[jp1, ip0]
            c11 = grid_vals[jp1, ip1]

        c0 = c00 * (1.0 - xd) + c10 * xd
        c1 = c01 * (1.0 - xd) + c11 * xd

        return c0 * (1.0 - zd) + c1 * zd

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

    @njit(cache=True, parallel=True)
    def _numba_winslow_loop2(
        x,
        z,
        grid_x,
        grid_z,
        grid_vals,
        stencils,
        is_movable,
        move_x,
        move_z,
        move_hyperellipse,
        hyperellipse_params,
        omega,
        iterations,
    ):
        """Internal Numba-accelerated inner loop for Winslow smoothing."""
        num_nodes = len(x)
        x_new = np.empty_like(x)
        z_new = np.empty_like(z)
        sizing_values = np.empty_like(x)
        a_e, b_e, xc_e, zc_e, n_e = (
            hyperellipse_params[0],
            hyperellipse_params[1],
            hyperellipse_params[2],
            hyperellipse_params[3],
            hyperellipse_params[4],
        )

        for iteration in range(iterations):
            for i in prange(num_nodes):
                sizing_values[i] = bilinear_interp_2d(x[i], z[i], grid_x, grid_z, grid_vals)

            for i in prange(num_nodes):
                x_new[i] = x[i]
                z_new[i] = z[i]

                if not (move_x[i] or move_z[i] or move_hyperellipse[i]):
                    continue

                i_e, i_w, i_n, i_s = (
                    stencils[i, 0],
                    stencils[i, 1],
                    stencils[i, 2],
                    stencils[i, 3],
                )
                i_ne, i_nw, i_se, i_sw = (
                    stencils[i, 4],
                    stencils[i, 5],
                    stencils[i, 6],
                    stencils[i, 7],
                )

                x_c, z_c = x[i], z[i]
                x_e = _cardinal_coordinate(x, x_c, i_e, i_w)
                x_w = _cardinal_coordinate(x, x_c, i_w, i_e)
                x_n = _cardinal_coordinate(x, x_c, i_n, i_s)
                x_s = _cardinal_coordinate(x, x_c, i_s, i_n)

                z_e = _cardinal_coordinate(z, z_c, i_e, i_w)
                z_w = _cardinal_coordinate(z, z_c, i_w, i_e)
                z_n = _cardinal_coordinate(z, z_c, i_n, i_s)
                z_s = _cardinal_coordinate(z, z_c, i_s, i_n)

                x_ne = _mixed_coordinate(x, x_c, i_ne, x_n, x_e)
                x_nw = _mixed_coordinate(x, x_c, i_nw, x_n, x_w)
                x_se = _mixed_coordinate(x, x_c, i_se, x_s, x_e)
                x_sw = _mixed_coordinate(x, x_c, i_sw, x_s, x_w)

                z_ne = _mixed_coordinate(z, z_c, i_ne, z_n, z_e)
                z_nw = _mixed_coordinate(z, z_c, i_nw, z_n, z_w)
                z_se = _mixed_coordinate(z, z_c, i_se, z_s, z_e)
                z_sw = _mixed_coordinate(z, z_c, i_sw, z_s, z_w)

                x_xi = 0.5 * (x_e - x_w)
                x_eta = 0.5 * (x_n - x_s)
                z_xi = 0.5 * (z_e - z_w)
                z_eta = 0.5 * (z_n - z_s)

                x_xita = 0.25 * (x_ne - x_nw - x_se + x_sw)
                z_xita = 0.25 * (z_ne - z_nw - z_se + z_sw)

                alpha = x_eta**2 + z_eta**2
                beta = x_xi * x_eta + z_xi * z_eta
                gamma = x_xi**2 + z_xi**2

                size_center = sizing_values[i]
                size_east = _neighbor_size(sizing_values, is_movable, i_e, size_center)
                size_west = _neighbor_size(sizing_values, is_movable, i_w, size_center)
                size_north = _neighbor_size(sizing_values, is_movable, i_n, size_center)
                size_south = _neighbor_size(sizing_values, is_movable, i_s, size_center)

                size_xi = 0.5 * (size_east - size_west)
                size_eta = 0.5 * (size_north - size_south)

                jacobian = x_xi * z_eta - x_eta * z_xi
                jacobian_over_size = jacobian / (size_center + 1e-12)

                source_x = jacobian_over_size * (size_xi * z_eta - size_eta * z_xi)
                source_z = jacobian_over_size * (size_eta * x_xi - size_xi * x_eta)

                denominator = 2.0 * (alpha + gamma)
                if denominator < 1e-12:
                    continue

                x_target = x[i]
                z_target = z[i]

                if move_x[i] or move_hyperellipse[i]:
                    x_target = (
                        alpha * (x_e + x_w)
                        + gamma * (x_n + x_s)
                        - 2.0 * beta * x_xita
                        - source_x
                    ) / denominator

                if move_z[i] or move_hyperellipse[i]:
                    z_target = (
                        alpha * (z_e + z_w)
                        + gamma * (z_n + z_s)
                        - 2.0 * beta * z_xita
                        - source_z
                    ) / denominator

                if move_hyperellipse[i]:
                    dx = x_target - xc_e
                    dz = z_target - zc_e

                    if abs(dx) < 1e-12 and abs(dz) < 1e-12:
                        x_new[i] = x[i]
                        z_new[i] = z[i]
                    else:
                        scale = (abs(dx / a_e) ** n_e + abs(dz / b_e) ** n_e) ** (
                            -1.0 / n_e
                        )
                        x_proj = xc_e + scale * dx
                        z_proj = zc_e + scale * dz
                        x_new[i] = (1.0 - omega) * x[i] + omega * x_proj
                        z_new[i] = (1.0 - omega) * z[i] + omega * z_proj
                else:
                    if move_x[i]:
                        x_new[i] = (1.0 - omega) * x[i] + omega * x_target
                    if move_z[i]:
                        z_new[i] = (1.0 - omega) * z[i] + omega * z_target

            for i in prange(num_nodes):
                x[i] = x_new[i]
                z[i] = z_new[i]

        return x, z

    def winslow_smooth_numba(
        points: np.ndarray,
        quads: np.ndarray,
        segy_grid_x: np.ndarray,
        segy_grid_z: np.ndarray,
        segy_grid_vals: np.ndarray,
        move_all: set,
        move_x_only: set,
        move_z_only: set,
        move_hyperellipse: set = None,
        hyperellipse_params: tuple = None,
        iterations: int = 1500,
        omega: float = 0.05,
    ) -> np.ndarray:
        """Apply Winslow mesh smoothing using Numba JIT compilation.

        Parameters
        ----------
        points : ndarray
            2D array of shape (N, 2) containing node coordinates [x, z].
        quads : ndarray
            2D array of shape (M, 4) containing node indices for each quadrilateral.
        segy_grid_x : ndarray
            1D array of x-coordinates for the velocity/sizing field grid.
        segy_grid_z : ndarray
            1D array of z-coordinates for the velocity/sizing field grid.
        segy_grid_vals : ndarray
            Element sizes in x/z order, with shape
            ``(len(segy_grid_x), len(segy_grid_z))``. Transposed non-square
            arrays are also accepted; square arrays always use x/z order.
        move_all : set
            Set of node indices permitted to move in both X and Z directions.
        move_x_only : set
            Set of node indices constrained to move only along the X-axis.
        move_z_only : set
            Set of node indices constrained to move only along the Z-axis.
        move_hyperellipse : set, optional
            Set of node indices constrained to move along an elliptical boundary.
        hyperellipse_params : tuple, optional
            Tuple containing parameters for an elliptical boundary.
        iterations : int, optional
            Total number of smoothing iterations. Default is 1500.
        omega : float, optional
            Relaxation factor for node movement (0 < omega <= 1). Default is 0.05.

        Returns
        -------
        ndarray
            Updated node coordinates array of shape (N, 2).
        """
        if move_hyperellipse is None:
            move_hyperellipse = set()

        if hyperellipse_params is None:
            e_params = np.array([1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.float64)
        else:
            e_params = np.array(hyperellipse_params, dtype=np.float64)

        num_nodes = len(points)

        node_to_neighbors = {i: set() for i in range(num_nodes)}
        for quad in quads:
            for i in range(4):
                start_node = quad[i]
                end_node = quad[(i + 1) % 4]
                node_to_neighbors[start_node].add(end_node)
                node_to_neighbors[end_node].add(start_node)

        stencils = {}
        for i in range(num_nodes):
            neighbors = list(node_to_neighbors[i])
            if not neighbors:
                stencils[i] = {
                    direction: None for direction in ["E", "N", "W", "S", "NE", "NW", "SE", "SW"]
                }
                continue

            vecs = points[neighbors] - points[i]
            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            sort_idx = np.argsort(angles)
            sorted_neighbors = [neighbors[idx] for idx in sort_idx]
            sorted_angles = angles[sort_idx]
            cardinals = {"E": None, "N": None, "W": None, "S": None}

            if len(sorted_neighbors) == 4:
                cardinals["E"] = sorted_neighbors[0]
                cardinals["N"] = sorted_neighbors[1]
                cardinals["W"] = sorted_neighbors[2]
                cardinals["S"] = sorted_neighbors[3]
            elif len(sorted_neighbors) > 0:
                diffs = np.diff(sorted_angles)
                diffs = np.append(
                    diffs, sorted_angles[0] - sorted_angles[-1] + 2 * np.pi
                )
                gap_idx = np.argmax(diffs)
                rolled_neighbors = np.roll(sorted_neighbors, -(gap_idx + 1))

                if len(sorted_neighbors) == 3:
                    cardinals["E"] = rolled_neighbors[0]
                    cardinals["N"] = rolled_neighbors[1]
                    cardinals["W"] = rolled_neighbors[2]
                elif len(sorted_neighbors) == 2:
                    cardinals["E"] = rolled_neighbors[0]
                    cardinals["N"] = rolled_neighbors[1]

            def get_shared(first_neighbor, second_neighbor):
                if first_neighbor is None or second_neighbor is None:
                    return None
                shared = node_to_neighbors[first_neighbor].intersection(node_to_neighbors[second_neighbor])
                shared.discard(i)
                return list(shared)[0] if shared else None

            corners = {
                "NE": get_shared(cardinals["N"], cardinals["E"]),
                "NW": get_shared(cardinals["N"], cardinals["W"]),
                "SE": get_shared(cardinals["S"], cardinals["E"]),
                "SW": get_shared(cardinals["S"], cardinals["W"]),
            }
            stencils[i] = {**cardinals, **corners}

        dir_map = {"E": 0, "W": 1, "N": 2, "S": 3, "NE": 4, "NW": 5, "SE": 6, "SW": 7}
        stencil_arr = np.full((num_nodes, 8), -1, dtype=np.int32)

        for i, stencil in stencils.items():
            for direction, col_idx in dir_map.items():
                if stencil[direction] is not None:
                    stencil_arr[i, col_idx] = stencil[direction]

        movable_nodes = move_all | move_x_only | move_z_only | move_hyperellipse
        is_movable = np.zeros(num_nodes, dtype=np.bool_)
        is_movable[list(movable_nodes)] = True

        move_x_mask = np.zeros(num_nodes, dtype=np.bool_)
        move_z_mask = np.zeros(num_nodes, dtype=np.bool_)
        move_hyperellipse_mask = np.zeros(num_nodes, dtype=np.bool_)

        move_x_mask[list(move_all | move_x_only)] = True
        move_z_mask[list(move_all | move_z_only)] = True
        move_hyperellipse_mask[list(move_hyperellipse)] = True

        x = np.ascontiguousarray(points[:, 0], dtype=np.float64)
        z = np.ascontiguousarray(points[:, 1], dtype=np.float64)

        print(f"Starting {iterations} Winslow iterations...")

        print_interval = 50  # Update the console every 50 iterations
        current_iter = 0

        while current_iter < iterations:
            step_iters = min(print_interval, iterations - current_iter)

            x, z = _numba_winslow_loop2(
                x,
                z,
                segy_grid_x,
                segy_grid_z,
                segy_grid_vals,
                stencil_arr,
                is_movable,
                move_x_mask,
                move_z_mask,
                move_hyperellipse_mask,
                e_params,
                omega,
                step_iters,
            )

            current_iter += step_iters

            percent = (current_iter / iterations) * 100
            print(
                f"Iteration: {current_iter} / {iterations} [{percent:.1f}%]   ",
                end="\r",
            )

        print("\nSmoothing Complete.")
        return np.column_stack((x, z))


# Define a fallback if Numba is not available
else:
    NUMBA_ERROR = (
        "Numba is not installed in this environment. "
        "Please run 'pip install numba' to use the 'numba' implementation, "
        "or switch configuration to winslow_implementation='fast' or 'default'."
    )

    def winslow_smooth_numba(*args, **kwargs):
        raise ImportError(NUMBA_ERROR)

    def _numba_winslow_loop2(*args, **kwargs):
        raise ImportError(NUMBA_ERROR)

    def bilinear_interp_2d(*args, **kwargs):
        raise ImportError(NUMBA_ERROR)


def vectorized_bilinear_interp(
    x: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    grid_vals: np.ndarray,
) -> np.ndarray:
    """Perform vectorized 2D bilinear interpolation over multiple points.

    Parameters
    ----------
    x : ndarray
        1D array of target x-coordinates.
    z : ndarray
        1D array of target z-coordinates.
    grid_x : ndarray
        Monotonically increasing or decreasing x-coordinates of the grid.
    grid_z : ndarray
        Monotonically increasing or decreasing z-coordinates of the grid.
    grid_vals : ndarray
        Values indexed as ``grid_vals[x_index, z_index]``, with shape
        ``(len(grid_x), len(grid_z))``. The transposed layout is also
        accepted for non-square grids. Square grids always use x/z order.

    Returns
    -------
    ndarray
        Interpolated values for all (x, z) coordinate pairs.

    Raises
    ------
    ValueError
        If grid inputs are degenerate, repeated, or misaligned.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    grid_x = np.asarray(grid_x, dtype=np.float64).ravel()
    grid_z = np.asarray(grid_z, dtype=np.float64).ravel()
    grid_vals = np.asarray(grid_vals, dtype=np.float64)

    nx = grid_x.size
    nz = grid_z.size

    if nx < 2 or nz < 2:
        raise ValueError("grid_x and grid_z must each contain at least 2 points.")

    if grid_vals.shape not in ((nx, nz), (nz, nx)):
        raise ValueError(
            f"grid_vals has shape {grid_vals.shape}, but expected "
            f"({nx}, {nz}) or ({nz}, {nx})."
        )
    if grid_vals.shape != (nx, nz):
        grid_vals = grid_vals.T

    if np.any(np.diff(grid_x) == 0.0):
        raise ValueError("grid_x contains repeated coordinates.")
    if np.any(np.diff(grid_z) == 0.0):
        raise ValueError("grid_z contains repeated coordinates.")

    if grid_x[0] > grid_x[-1]:
        grid_x = grid_x[::-1]
        grid_vals = grid_vals[::-1, :]

    if grid_z[0] > grid_z[-1]:
        grid_z = grid_z[::-1]
        grid_vals = grid_vals[:, ::-1]

    xq = np.clip(x, grid_x[0], grid_x[-1])
    zq = np.clip(z, grid_z[0], grid_z[-1])
    i = np.searchsorted(grid_x, xq, side="right") - 1
    j = np.searchsorted(grid_z, zq, side="right") - 1
    i = np.clip(i, 0, nx - 2)
    j = np.clip(j, 0, nz - 2)
    x0 = grid_x[i]
    x1 = grid_x[i + 1]
    z0 = grid_z[j]
    z1 = grid_z[j + 1]

    dx = x1 - x0
    dz = z1 - z0
    xd = np.empty_like(xq)
    zd = np.empty_like(zq)
    mask_dx = dx > 0.0
    mask_dz = dz > 0.0

    xd[mask_dx] = (xq[mask_dx] - x0[mask_dx]) / dx[mask_dx]
    xd[~mask_dx] = 0.0

    zd[mask_dz] = (zq[mask_dz] - z0[mask_dz]) / dz[mask_dz]
    zd[~mask_dz] = 0.0

    xd = np.clip(xd, 0.0, 1.0)
    zd = np.clip(zd, 0.0, 1.0)
    c00 = grid_vals[i, j]
    c10 = grid_vals[i + 1, j]
    c01 = grid_vals[i, j + 1]
    c11 = grid_vals[i + 1, j + 1]

    c0 = (1.0 - xd) * c00 + xd * c10
    c1 = (1.0 - xd) * c01 + xd * c11
    out = (1.0 - zd) * c0 + zd * c1

    return out


def _vectorized_winslow_loop(
    x: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    grid_vals: np.ndarray,
    stencil_arr: np.ndarray,
    is_movable: np.ndarray,
    move_x_mask: np.ndarray,
    move_z_mask: np.ndarray,
    move_hyperellipse_mask: np.ndarray,
    hyperellipse_params: np.ndarray,
    omega: float,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal NumPy-vectorized inner loop for fast Winslow smoothing."""
    a_e, b_e, xc_e, zc_e, n_e = hyperellipse_params

    i_e, i_w, i_n, i_s = (
        stencil_arr[:, 0],
        stencil_arr[:, 1],
        stencil_arr[:, 2],
        stencil_arr[:, 3],
    )
    i_ne, i_nw, i_se, i_sw = (
        stencil_arr[:, 4],
        stencil_arr[:, 5],
        stencil_arr[:, 6],
        stencil_arr[:, 7],
    )

    has_e, has_w = i_e != -1, i_w != -1
    has_n, has_s = i_n != -1, i_s != -1
    has_ne, has_nw = i_ne != -1, i_nw != -1
    has_se, has_sw = i_se != -1, i_sw != -1

    safe_i_e, safe_i_w = np.maximum(i_e, 0), np.maximum(i_w, 0)
    safe_i_n, safe_i_s = np.maximum(i_n, 0), np.maximum(i_s, 0)
    safe_i_ne, safe_i_nw = np.maximum(i_ne, 0), np.maximum(i_nw, 0)
    safe_i_se, safe_i_sw = np.maximum(i_se, 0), np.maximum(i_sw, 0)

    movable_e = has_e & is_movable[safe_i_e]
    movable_w = has_w & is_movable[safe_i_w]
    movable_n = has_n & is_movable[safe_i_n]
    movable_s = has_s & is_movable[safe_i_s]

    for _ in range(iterations):
        sizing_values = vectorized_bilinear_interp(x, z, grid_x, grid_z, grid_vals)

        x_e = np.where(has_e, x[safe_i_e], np.where(has_w, 2.0 * x - x[safe_i_w], x))
        x_w = np.where(has_w, x[safe_i_w], np.where(has_e, 2.0 * x - x[safe_i_e], x))
        x_n = np.where(has_n, x[safe_i_n], np.where(has_s, 2.0 * x - x[safe_i_s], x))
        x_s = np.where(has_s, x[safe_i_s], np.where(has_n, 2.0 * x - x[safe_i_n], x))

        z_e = np.where(has_e, z[safe_i_e], np.where(has_w, 2.0 * z - z[safe_i_w], z))
        z_w = np.where(has_w, z[safe_i_w], np.where(has_e, 2.0 * z - z[safe_i_e], z))
        z_n = np.where(has_n, z[safe_i_n], np.where(has_s, 2.0 * z - z[safe_i_s], z))
        z_s = np.where(has_s, z[safe_i_s], np.where(has_n, 2.0 * z - z[safe_i_n], z))

        x_ne = np.where(has_ne, x[safe_i_ne], x_n + x_e - x)
        x_nw = np.where(has_nw, x[safe_i_nw], x_n + x_w - x)
        x_se = np.where(has_se, x[safe_i_se], x_s + x_e - x)
        x_sw = np.where(has_sw, x[safe_i_sw], x_s + x_w - x)

        z_ne = np.where(has_ne, z[safe_i_ne], z_n + z_e - z)
        z_nw = np.where(has_nw, z[safe_i_nw], z_n + z_w - z)
        z_se = np.where(has_se, z[safe_i_se], z_s + z_e - z)
        z_sw = np.where(has_sw, z[safe_i_sw], z_s + z_w - z)

        x_xi = 0.5 * (x_e - x_w)
        x_eta = 0.5 * (x_n - x_s)
        z_xi = 0.5 * (z_e - z_w)
        z_eta = 0.5 * (z_n - z_s)

        x_xita = 0.25 * (x_ne - x_nw - x_se + x_sw)
        z_xita = 0.25 * (z_ne - z_nw - z_se + z_sw)

        alpha = x_eta**2 + z_eta**2
        beta = x_xi * x_eta + z_xi * z_eta
        gamma = x_xi**2 + z_xi**2

        size_center = sizing_values
        size_east = np.where(movable_e, sizing_values[safe_i_e], size_center)
        size_west = np.where(movable_w, sizing_values[safe_i_w], size_center)
        size_north = np.where(movable_n, sizing_values[safe_i_n], size_center)
        size_south = np.where(movable_s, sizing_values[safe_i_s], size_center)

        size_xi = 0.5 * (size_east - size_west)
        size_eta = 0.5 * (size_north - size_south)
        jacobian = x_xi * z_eta - x_eta * z_xi
        jacobian_over_size = jacobian / (size_center + 1e-12)
        source_x = jacobian_over_size * (size_xi * z_eta - size_eta * z_xi)
        source_z = jacobian_over_size * (size_eta * x_xi - size_xi * x_eta)
        denominator = 2.0 * (alpha + gamma)
        valid_denom_mask = denominator >= 1e-12
        denom_safe = np.where(valid_denom_mask, denominator, 1.0)  # Prevent DivZero

        x_target = np.where(
            valid_denom_mask & (move_x_mask | move_hyperellipse_mask),
            (alpha * (x_e + x_w) + gamma * (x_n + x_s) - 2.0 * beta * x_xita - source_x)
            / denom_safe,
            x,
        )
        z_target = np.where(
            valid_denom_mask & (move_z_mask | move_hyperellipse_mask),
            (alpha * (z_e + z_w) + gamma * (z_n + z_s) - 2.0 * beta * z_xita - source_z)
            / denom_safe,
            z,
        )

        dx = x_target - xc_e
        dz = z_target - zc_e
        zero_dist_mask = (np.abs(dx) < 1e-12) & (np.abs(dz) < 1e-12)
        dist_term = np.abs(dx / a_e) ** n_e + np.abs(dz / b_e) ** n_e
        dist_safe = np.where(dist_term < 1e-12, 1.0, dist_term)  # Prevent DivZero
        scale = dist_safe ** (-1.0 / n_e)
        x_proj = xc_e + scale * dx
        z_proj = zc_e + scale * dz
        x_hyperellipse_new = np.where(
            zero_dist_mask, x, (1.0 - omega) * x + omega * x_proj
        )
        z_hyperellipse_new = np.where(
            zero_dist_mask, z, (1.0 - omega) * z + omega * z_proj
        )
        x_reg_new = np.where(move_x_mask, (1.0 - omega) * x + omega * x_target, x)
        z_reg_new = np.where(move_z_mask, (1.0 - omega) * z + omega * z_target, z)
        x = np.where(move_hyperellipse_mask, x_hyperellipse_new, x_reg_new)
        z = np.where(move_hyperellipse_mask, z_hyperellipse_new, z_reg_new)

    return x, z


def winslow_smooth_vectorized(
    points: np.ndarray,
    quads: np.ndarray,
    segy_grid_x: np.ndarray,
    segy_grid_z: np.ndarray,
    segy_grid_vals: np.ndarray,
    move_all: set,
    move_x_only: set,
    move_z_only: set,
    move_hyperellipse: set = None,
    hyperellipse_params: tuple = None,
    iterations: int = 1500,
    omega: float = 0.05,
) -> np.ndarray:
    """Apply Winslow mesh smoothing using vectorized NumPy operations.

    Parameters
    ----------
    points : ndarray
        2D array of shape (N, 2) containing node coordinates [x, z].
    quads : ndarray
        2D array of shape (M, 4) containing node indices for each quadrilateral.
    segy_grid_x : ndarray
        1D array of x-coordinates for the velocity/sizing field grid.
    segy_grid_z : ndarray
        1D array of z-coordinates for the velocity/sizing field grid.
    segy_grid_vals : ndarray
        Element sizes in x/z order, with shape
        ``(len(segy_grid_x), len(segy_grid_z))``. Transposed non-square
        arrays are also accepted; square arrays always use x/z order.
    move_all : set
        Set of node indices permitted to move in both X and Z directions.
    move_x_only : set
        Set of node indices constrained to move only along the X-axis.
    move_z_only : set
        Set of node indices constrained to move only along the Z-axis.
    move_hyperellipse : set, optional
        Set of node indices constrained to move along an elliptical boundary.
    hyperellipse_params : tuple, optional
        Tuple containing parameters for an elliptical boundary.
    iterations : int, optional
        Total number of smoothing iterations. Default is 1500.
    omega : float, optional
        Relaxation factor for node movement (0 < omega <= 1). Default is 0.05.

    Returns
    -------
    ndarray
        Updated node coordinates array of shape (N, 2).
    """
    if move_hyperellipse is None:
        move_hyperellipse = set()

    if hyperellipse_params is None:
        e_params = np.array([1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.float64)
    else:
        e_params = np.array(hyperellipse_params, dtype=np.float64)

    num_nodes = len(points)

    node_to_neighbors = {i: set() for i in range(num_nodes)}
    for quad in quads:
        for i in range(4):
            start_node = quad[i]
            end_node = quad[(i + 1) % 4]
            node_to_neighbors[start_node].add(end_node)
            node_to_neighbors[end_node].add(start_node)

    stencils = {}
    for i in range(num_nodes):
        neighbors = list(node_to_neighbors[i])
        if not neighbors:
            stencils[i] = {
                direction: None for direction in ["E", "N", "W", "S", "NE", "NW", "SE", "SW"]
            }
            continue

        vecs = points[neighbors] - points[i]
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        sort_idx = np.argsort(angles)
        sorted_neighbors = [neighbors[idx] for idx in sort_idx]
        sorted_angles = angles[sort_idx]
        cardinals = {"E": None, "N": None, "W": None, "S": None}

        if len(sorted_neighbors) == 4:
            cardinals["E"] = sorted_neighbors[0]
            cardinals["N"] = sorted_neighbors[1]
            cardinals["W"] = sorted_neighbors[2]
            cardinals["S"] = sorted_neighbors[3]
        elif len(sorted_neighbors) > 0:
            diffs = np.diff(sorted_angles)
            diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2 * np.pi)
            gap_idx = np.argmax(diffs)
            rolled_neighbors = np.roll(sorted_neighbors, -(gap_idx + 1))

            if len(sorted_neighbors) == 3:
                cardinals["E"] = rolled_neighbors[0]
                cardinals["N"] = rolled_neighbors[1]
                cardinals["W"] = rolled_neighbors[2]
            elif len(sorted_neighbors) == 2:
                cardinals["E"] = rolled_neighbors[0]
                cardinals["N"] = rolled_neighbors[1]

        def get_shared(first_neighbor, second_neighbor):
            if first_neighbor is None or second_neighbor is None:
                return None
            shared = node_to_neighbors[first_neighbor].intersection(node_to_neighbors[second_neighbor])
            shared.discard(i)
            return list(shared)[0] if shared else None

        corners = {
            "NE": get_shared(cardinals["N"], cardinals["E"]),
            "NW": get_shared(cardinals["N"], cardinals["W"]),
            "SE": get_shared(cardinals["S"], cardinals["E"]),
            "SW": get_shared(cardinals["S"], cardinals["W"]),
        }
        stencils[i] = {**cardinals, **corners}

    dir_map = {"E": 0, "W": 1, "N": 2, "S": 3, "NE": 4, "NW": 5, "SE": 6, "SW": 7}
    stencil_arr = np.full((num_nodes, 8), -1, dtype=np.int32)

    for i, stencil in stencils.items():
        for direction, col_idx in dir_map.items():
            if stencil[direction] is not None:
                stencil_arr[i, col_idx] = stencil[direction]

    movable_nodes = move_all | move_x_only | move_z_only | move_hyperellipse
    is_movable = np.zeros(num_nodes, dtype=np.bool_)
    is_movable[list(movable_nodes)] = True

    move_x_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_z_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_hyperellipse_mask = np.zeros(num_nodes, dtype=np.bool_)

    move_x_mask[list(move_all | move_x_only)] = True
    move_z_mask[list(move_all | move_z_only)] = True
    move_hyperellipse_mask[list(move_hyperellipse)] = True

    x = np.ascontiguousarray(points[:, 0], dtype=np.float64)
    z = np.ascontiguousarray(points[:, 1], dtype=np.float64)

    print(f"Starting {iterations} Fast Winslow iterations...")

    print_interval = 50
    current_iter = 0

    while current_iter < iterations:
        step_iters = min(print_interval, iterations - current_iter)

        x, z = _vectorized_winslow_loop(
            x,
            z,
            segy_grid_x,
            segy_grid_z,
            segy_grid_vals,
            stencil_arr,
            is_movable,
            move_x_mask,
            move_z_mask,
            move_hyperellipse_mask,
            e_params,
            omega,
            step_iters,
        )

        current_iter += step_iters
        percent = (current_iter / iterations) * 100
        print(f"Iteration: {current_iter} / {iterations} [{percent:.1f}%]   ", end="\r")

    print("\nSmoothing Complete.")
    return np.column_stack((x, z))
