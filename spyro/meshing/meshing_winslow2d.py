import numpy as np
from typing import Callable, Tuple

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def winslow_smooth_default(
    points: np.ndarray,
    quads: np.ndarray,
    sizing_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    move_all: set,
    move_X_only: set,
    move_Z_only: set,
    move_hyperellipse: set = None,
    hyperellipse_params: tuple = None,
    iterations: int = 1500,
    omega: float = 0.05,
    plot_sources: bool = True
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
    move_X_only : set
        Set of node indices constrained to move only along the X-axis.
    move_Z_only : set
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
            n1 = quad[i]
            n2 = quad[(i + 1) % 4]
            node_to_neighbors[n1].add(n2)
            node_to_neighbors[n2].add(n1)

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
        cardinals = {'E': None, 'N': None, 'W': None, 'S': None}

        if len(sorted_neighbors) == 4:
            cardinals['E'] = sorted_neighbors[0]
            cardinals['N'] = sorted_neighbors[1]
            cardinals['W'] = sorted_neighbors[2]
            cardinals['S'] = sorted_neighbors[3]
        elif len(sorted_neighbors) > 0:
            diffs = np.diff(sorted_angles)
            diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2 * np.pi)
            gap_idx = np.argmax(diffs)
            rolled_neighbors = np.roll(sorted_neighbors, -(gap_idx + 1))

            if len(sorted_neighbors) == 3:
                cardinals['E'] = rolled_neighbors[0]
                cardinals['N'] = rolled_neighbors[1]
                cardinals['W'] = rolled_neighbors[2]
            elif len(sorted_neighbors) == 2:
                cardinals['E'] = rolled_neighbors[0]
                cardinals['N'] = rolled_neighbors[1]

        def get_shared(nA, nB):
            if nA is None or nB is None:
                return None
            shared = node_to_neighbors[nA].intersection(node_to_neighbors[nB])
            shared.discard(i)
            return list(shared)[0] if shared else None

        corners = {
            'NE': get_shared(cardinals['N'], cardinals['E']),
            'NW': get_shared(cardinals['N'], cardinals['W']),
            'SE': get_shared(cardinals['S'], cardinals['E']),
            'SW': get_shared(cardinals['S'], cardinals['W'])
        }
        stencils[i] = {**cardinals, **corners}

    X = points[:, 0].copy()
    Z = points[:, 1].copy()
    X_new = X.copy()
    Z_new = Z.copy()
    movable_nodes = move_all | move_X_only | move_Z_only | move_hyperellipse
    j_dc_arr = np.zeros(num_nodes)
    jj = 0
    for _ in range(iterations):
        jj += 1
        print(f"Iteration: {jj}", end='\r')
        h = sizing_fn(X, Z)

        for i, st in stencils.items():
            is_all = i in move_all
            is_X = i in move_X_only
            is_Z = i in move_Z_only
            is_hyperellipse = i in move_hyperellipse

            if not (is_all or is_X or is_Z or is_hyperellipse):
                continue

            C = {'X': {}, 'Z': {}}

            for d, opp in [('E', 'W'), ('W', 'E'), ('N', 'S'), ('S', 'N')]:
                idx = st[d]
                if idx is not None:
                    C['X'][d] = X[idx]
                    C['Z'][d] = Z[idx]
                else:
                    opp_idx = st[opp]
                    C['X'][d] = 2 * X[i] - X[opp_idx] if opp_idx is not None else X[i]
                    C['Z'][d] = 2 * Z[i] - Z[opp_idx] if opp_idx is not None else Z[i]

            for d, d1, d2 in [('NE', 'N', 'E'), ('NW', 'N', 'W'), ('SE', 'S', 'E'), ('SW', 'S', 'W')]:
                idx = st[d]
                if idx is not None:
                    C['X'][d] = X[idx]
                    C['Z'][d] = Z[idx]
                else:
                    C['X'][d] = C['X'][d1] + C['X'][d2] - X[i]
                    C['Z'][d] = C['Z'][d1] + C['Z'][d2] - Z[i]

            X_E, X_W, X_N, X_S = C['X']['E'], C['X']['W'], C['X']['N'], C['X']['S']
            Z_E, Z_W, Z_N, Z_S = C['Z']['E'], C['Z']['W'], C['Z']['N'], C['Z']['S']

            X_xi = 0.5 * (X_E - X_W)
            X_eta = 0.5 * (X_N - X_S)
            Z_xi = 0.5 * (Z_E - Z_W)
            Z_eta = 0.5 * (Z_N - Z_S)
            alpha = X_eta**2 + Z_eta**2
            beta = X_xi * X_eta + Z_xi * Z_eta
            gamma = X_xi**2 + Z_xi**2
            X_xita = 0.25 * (C['X']['NE'] - C['X']['NW'] - C['X']['SE'] + C['X']['SW'])
            Z_xita = 0.25 * (C['Z']['NE'] - C['Z']['NW'] - C['Z']['SE'] + C['Z']['SW'])

            D_C = h[i]
            D_E = h[st['E']] if (st['E'] is not None and st['E'] in movable_nodes) else D_C
            D_W = h[st['W']] if (st['W'] is not None and st['W'] in movable_nodes) else D_C
            D_N = h[st['N']] if (st['N'] is not None and st['N'] in movable_nodes) else D_C
            D_S = h[st['S']] if (st['S'] is not None and st['S'] in movable_nodes) else D_C

            D_phi = 0.5 * (D_E - D_W)
            D_psi = 0.5 * (D_N - D_S)
            J = X_xi * Z_eta - X_eta * Z_xi
            j_dc_arr[i] = J / (D_C + 1e-12)
            Source_X = (J / (D_C + 1e-12)) * (D_phi * Z_eta - D_psi * Z_xi)
            Source_Z = (J / (D_C + 1e-12)) * (D_psi * X_xi - D_phi * X_eta)
            Denominator = 2.0 * (alpha + gamma)

            if Denominator < 1e-12:
                continue

            X_target = X[i]
            Z_target = Z[i]

            if is_all or is_X or is_hyperellipse:
                X_target = (alpha * (X_E + X_W) + gamma * (X_N + X_S) - 2.0 * beta * X_xita - Source_X) / Denominator

            if is_all or is_Z or is_hyperellipse:
                Z_target = (alpha * (Z_E + Z_W) + gamma * (Z_N + Z_S) - 2.0 * beta * Z_xita - Source_Z) / Denominator

            if is_hyperellipse and hyperellipse_params is not None:
                a_e, b_e, xc_e, zc_e, n_e = hyperellipse_params
                dx = X_target - xc_e
                dz = Z_target - zc_e

                if abs(dx) < 1e-12 and abs(dz) < 1e-12:
                    X_new[i] = X[i]
                    Z_new[i] = Z[i]
                else:
                    scale = (abs(dx / a_e)**n_e + abs(dz / b_e)**n_e)**(-1.0 / n_e)
                    X_proj = xc_e + scale * dx
                    Z_proj = zc_e + scale * dz

                    X_new[i] = (1.0 - omega) * X[i] + omega * X_proj
                    Z_new[i] = (1.0 - omega) * Z[i] + omega * Z_proj

            else:
                if is_all or is_X:
                    X_new[i] = (1.0 - omega) * X[i] + omega * X_target

                if is_all or is_Z:
                    Z_new[i] = (1.0 - omega) * Z[i] + omega * Z_target

        X[:] = X_new[:]
        Z[:] = Z_new[:]

    return np.column_stack((X, Z))


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def bilinear_interp_2d(x, z, grid_x, grid_z, grid_vals):
        """Perform 2D bilinear interpolation for a single point.

        Parameters
        ----------
        x : float
            Target x-coordinate for interpolation.
        z : float
            Target z-coordinate for interpolation.
        grid_x : ndarray
            1D array of x-coordinates defining the grid.
        grid_z : ndarray
            1D array of z-coordinates defining the grid.
        grid_vals : ndarray
            2D array containing values at the grid vertices.

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
        vals_are_nx_nz = (gv0 == nx and gv1 == nz)
        vals_are_nz_nx = (gv0 == nz and gv1 == nx)
        if not vals_are_nx_nz and not vals_are_nz_nx:
            raise ValueError("grid_vals shape must be (len(grid_x), len(grid_z)) or (len(grid_z), len(grid_x)).")

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
            i = np.searchsorted(grid_x, xq, side='right') - 1
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
            j = np.searchsorted(grid_z, zq, side='right') - 1
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

    @njit(cache=True, parallel=True)
    def _numba_winslow_loop2(X, Z, grid_x, grid_z, grid_vals, stencils, is_movable,
                             move_X, move_Z, move_hyperellipse, hyperellipse_params, omega, iterations):
        """Internal Numba-accelerated inner loop for Winslow smoothing."""
        num_nodes = len(X)
        X_new = np.empty_like(X)
        Z_new = np.empty_like(Z)
        h = np.empty_like(X)
        a_e, b_e, xc_e, zc_e, n_e = hyperellipse_params[0], hyperellipse_params[1], hyperellipse_params[2], hyperellipse_params[3], hyperellipse_params[4]

        for jj in range(iterations):
            for i in prange(num_nodes):
                h[i] = bilinear_interp_2d(X[i], Z[i], grid_x, grid_z, grid_vals)

            for i in prange(num_nodes):
                X_new[i] = X[i]
                Z_new[i] = Z[i]

                if not (move_X[i] or move_Z[i] or move_hyperellipse[i]):
                    continue

                iE, iW, iN, iS = stencils[i, 0], stencils[i, 1], stencils[i, 2], stencils[i, 3]
                iNE, iNW, iSE, iSW = stencils[i, 4], stencils[i, 5], stencils[i, 6], stencils[i, 7]

                xE = X[iE] if iE != -1 else (2.0 * X[i] - X[iW] if iW != -1 else X[i])
                xW = X[iW] if iW != -1 else (2.0 * X[i] - X[iE] if iE != -1 else X[i])
                xN = X[iN] if iN != -1 else (2.0 * X[i] - X[iS] if iS != -1 else X[i])
                xS = X[iS] if iS != -1 else (2.0 * X[i] - X[iN] if iN != -1 else X[i])

                zE = Z[iE] if iE != -1 else (2.0 * Z[i] - Z[iW] if iW != -1 else Z[i])
                zW = Z[iW] if iW != -1 else (2.0 * Z[i] - Z[iE] if iE != -1 else Z[i])
                zN = Z[iN] if iN != -1 else (2.0 * Z[i] - Z[iS] if iS != -1 else Z[i])
                zS = Z[iS] if iS != -1 else (2.0 * Z[i] - Z[iN] if iN != -1 else Z[i])

                xNE = X[iNE] if iNE != -1 else (xN + xE - X[i])
                xNW = X[iNW] if iNW != -1 else (xN + xW - X[i])
                xSE = X[iSE] if iSE != -1 else (xS + xE - X[i])
                xSW = X[iSW] if iSW != -1 else (xS + xW - X[i])

                zNE = Z[iNE] if iNE != -1 else (zN + zE - Z[i])
                zNW = Z[iNW] if iNW != -1 else (zN + zW - Z[i])
                zSE = Z[iSE] if iSE != -1 else (zS + zE - Z[i])
                zSW = Z[iSW] if iSW != -1 else (zS + zW - Z[i])

                X_xi = 0.5 * (xE - xW)
                X_eta = 0.5 * (xN - xS)
                Z_xi = 0.5 * (zE - zW)
                Z_eta = 0.5 * (zN - zS)

                X_xita = 0.25 * (xNE - xNW - xSE + xSW)
                Z_xita = 0.25 * (zNE - zNW - zSE + zSW)

                alpha = X_eta**2 + Z_eta**2
                beta = X_xi * X_eta + Z_xi * Z_eta
                gamma = X_xi**2 + Z_xi**2

                DC = h[i]
                DE = h[iE] if (iE != -1 and is_movable[iE]) else DC
                DW = h[iW] if (iW != -1 and is_movable[iW]) else DC
                DN = h[iN] if (iN != -1 and is_movable[iN]) else DC
                DS = h[iS] if (iS != -1 and is_movable[iS]) else DC

                D_phi = 0.5 * (DE - DW)
                D_psi = 0.5 * (DN - DS)

                J = X_xi * Z_eta - X_eta * Z_xi
                j_dc = J / (DC + 1e-12)

                Source_X = j_dc * (D_phi * Z_eta - D_psi * Z_xi)
                Source_Z = j_dc * (D_psi * X_xi - D_phi * X_eta)

                Denominator = 2.0 * (alpha + gamma)
                if Denominator < 1e-12:
                    continue

                X_target = X[i]
                Z_target = Z[i]

                if move_X[i] or move_hyperellipse[i]:
                    X_target = (alpha * (xE + xW) + gamma * (xN + xS) - 2.0 * beta * X_xita - Source_X) / Denominator

                if move_Z[i] or move_hyperellipse[i]:
                    Z_target = (alpha * (zE + zW) + gamma * (zN + zS) - 2.0 * beta * Z_xita - Source_Z) / Denominator

                if move_hyperellipse[i]:
                    dx = X_target - xc_e
                    dz = Z_target - zc_e

                    if abs(dx) < 1e-12 and abs(dz) < 1e-12:
                        X_new[i] = X[i]
                        Z_new[i] = Z[i]
                    else:
                        scale = (abs(dx / a_e)**n_e + abs(dz / b_e)**n_e)**(-1.0 / n_e)
                        X_proj = xc_e + scale * dx
                        Z_proj = zc_e + scale * dz
                        X_new[i] = (1.0 - omega) * X[i] + omega * X_proj
                        Z_new[i] = (1.0 - omega) * Z[i] + omega * Z_proj
                else:
                    if move_X[i]:
                        X_new[i] = (1.0 - omega) * X[i] + omega * X_target
                    if move_Z[i]:
                        Z_new[i] = (1.0 - omega) * Z[i] + omega * Z_target

            for i in prange(num_nodes):
                X[i] = X_new[i]
                Z[i] = Z_new[i]

        return X, Z

    def winslow_smooth_numba(
        points: np.ndarray,
        quads: np.ndarray,
        segy_grid_x: np.ndarray,
        segy_grid_z: np.ndarray,
        segy_grid_vals: np.ndarray,
        move_all: set,
        move_X_only: set,
        move_Z_only: set,
        move_hyperellipse: set = None,
        hyperellipse_params: tuple = None,
        iterations: int = 1500,
        omega: float = 0.05
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
            2D array of element sizing values.
        move_all : set
            Set of node indices permitted to move in both X and Z directions.
        move_X_only : set
            Set of node indices constrained to move only along the X-axis.
        move_Z_only : set
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
                n1 = quad[i]
                n2 = quad[(i + 1) % 4]
                node_to_neighbors[n1].add(n2)
                node_to_neighbors[n2].add(n1)

        stencils = {}
        for i in range(num_nodes):
            neighbors = list(node_to_neighbors[i])
            if not neighbors:
                stencils[i] = {d: None for d in ['E', 'N', 'W', 'S', 'NE', 'NW', 'SE', 'SW']}
                continue

            vecs = points[neighbors] - points[i]
            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            sort_idx = np.argsort(angles)
            sorted_neighbors = [neighbors[idx] for idx in sort_idx]
            sorted_angles = angles[sort_idx]
            cardinals = {'E': None, 'N': None, 'W': None, 'S': None}

            if len(sorted_neighbors) == 4:
                cardinals['E'] = sorted_neighbors[0]
                cardinals['N'] = sorted_neighbors[1]
                cardinals['W'] = sorted_neighbors[2]
                cardinals['S'] = sorted_neighbors[3]
            elif len(sorted_neighbors) > 0:
                diffs = np.diff(sorted_angles)
                diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2 * np.pi)
                gap_idx = np.argmax(diffs)
                rolled_neighbors = np.roll(sorted_neighbors, -(gap_idx + 1))

                if len(sorted_neighbors) == 3:
                    cardinals['E'] = rolled_neighbors[0]
                    cardinals['N'] = rolled_neighbors[1]
                    cardinals['W'] = rolled_neighbors[2]
                elif len(sorted_neighbors) == 2:
                    cardinals['E'] = rolled_neighbors[0]
                    cardinals['N'] = rolled_neighbors[1]

            def get_shared(nA, nB):
                if nA is None or nB is None:
                    return None
                shared = node_to_neighbors[nA].intersection(node_to_neighbors[nB])
                shared.discard(i)
                return list(shared)[0] if shared else None

            corners = {
                'NE': get_shared(cardinals['N'], cardinals['E']),
                'NW': get_shared(cardinals['N'], cardinals['W']),
                'SE': get_shared(cardinals['S'], cardinals['E']),
                'SW': get_shared(cardinals['S'], cardinals['W'])
            }
            stencils[i] = {**cardinals, **corners}

        dir_map = {'E': 0, 'W': 1, 'N': 2, 'S': 3, 'NE': 4, 'NW': 5, 'SE': 6, 'SW': 7}
        stencil_arr = np.full((num_nodes, 8), -1, dtype=np.int32)

        for i, st in stencils.items():
            for d, col_idx in dir_map.items():
                if st[d] is not None:
                    stencil_arr[i, col_idx] = st[d]

        movable_nodes = move_all | move_X_only | move_Z_only | move_hyperellipse
        is_movable = np.zeros(num_nodes, dtype=np.bool_)
        is_movable[list(movable_nodes)] = True

        move_X_mask = np.zeros(num_nodes, dtype=np.bool_)
        move_Z_mask = np.zeros(num_nodes, dtype=np.bool_)
        move_hyperellipse_mask = np.zeros(num_nodes, dtype=np.bool_)

        move_X_mask[list(move_all | move_X_only)] = True
        move_Z_mask[list(move_all | move_Z_only)] = True
        move_hyperellipse_mask[list(move_hyperellipse)] = True

        X = np.ascontiguousarray(points[:, 0], dtype=np.float64)
        Z = np.ascontiguousarray(points[:, 1], dtype=np.float64)

        print(f"Starting {iterations} Winslow iterations...")

        print_interval = 50  # Update the console every 50 iterations
        current_iter = 0

        while current_iter < iterations:
            step_iters = min(print_interval, iterations - current_iter)

            X, Z = _numba_winslow_loop2(
                X, Z,
                segy_grid_x, segy_grid_z, segy_grid_vals,
                stencil_arr, is_movable, move_X_mask, move_Z_mask, move_hyperellipse_mask, e_params, omega, step_iters
            )

            current_iter += step_iters

            percent = (current_iter / iterations) * 100
            print(f"Iteration: {current_iter} / {iterations} [{percent:.1f}%]   ", end='\r')

        print("\nSmoothing Complete.")
        return np.column_stack((X, Z))

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
    grid_vals: np.ndarray
) -> np.ndarray:
    """Perform vectorized 2D bilinear interpolation over multiple points.

    Parameters
    ----------
    x : ndarray
        1D array of target x-coordinates.
    z : ndarray
        1D array of target z-coordinates.
    grid_x : ndarray
        1D array of grid x-coordinates.
    grid_z : ndarray
        1D array of grid z-coordinates.
    grid_vals : ndarray
        2D array containing values at the grid vertices.

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

    if grid_vals.shape == (nz, nx):
        grid_vals = grid_vals.T
    elif grid_vals.shape != (nx, nz):
        raise ValueError(
            f"grid_vals has shape {grid_vals.shape}, but expected "
            f"({nx}, {nz}) or ({nz}, {nx})."
        )

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
    X: np.ndarray, Z: np.ndarray,
    grid_x: np.ndarray, grid_z: np.ndarray, grid_vals: np.ndarray,
    stencil_arr: np.ndarray, is_movable: np.ndarray,
    move_X_mask: np.ndarray, move_Z_mask: np.ndarray, move_hyperellipse_mask: np.ndarray,
    hyperellipse_params: np.ndarray, omega: float, iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal NumPy-vectorized inner loop for fast Winslow smoothing."""

    a_e, b_e, xc_e, zc_e, n_e = hyperellipse_params

    iE, iW, iN, iS = stencil_arr[:, 0], stencil_arr[:, 1], stencil_arr[:, 2], stencil_arr[:, 3]
    iNE, iNW, iSE, iSW = stencil_arr[:, 4], stencil_arr[:, 5], stencil_arr[:, 6], stencil_arr[:, 7]

    mE, mW = iE != -1, iW != -1
    mN, mS = iN != -1, iS != -1
    mNE, mNW = iNE != -1, iNW != -1
    mSE, mSW = iSE != -1, iSW != -1

    safe_iE, safe_iW = np.maximum(iE, 0), np.maximum(iW, 0)
    safe_iN, safe_iS = np.maximum(iN, 0), np.maximum(iS, 0)
    safe_iNE, safe_iNW = np.maximum(iNE, 0), np.maximum(iNW, 0)
    safe_iSE, safe_iSW = np.maximum(iSE, 0), np.maximum(iSW, 0)

    movable_E = mE & is_movable[safe_iE]
    movable_W = mW & is_movable[safe_iW]
    movable_N = mN & is_movable[safe_iN]
    movable_S = mS & is_movable[safe_iS]

    for _ in range(iterations):
        h = vectorized_bilinear_interp(X, Z, grid_x, grid_z, grid_vals)

        xE = np.where(mE, X[safe_iE], np.where(mW, 2.0 * X - X[safe_iW], X))
        xW = np.where(mW, X[safe_iW], np.where(mE, 2.0 * X - X[safe_iE], X))
        xN = np.where(mN, X[safe_iN], np.where(mS, 2.0 * X - X[safe_iS], X))
        xS = np.where(mS, X[safe_iS], np.where(mN, 2.0 * X - X[safe_iN], X))

        zE = np.where(mE, Z[safe_iE], np.where(mW, 2.0 * Z - Z[safe_iW], Z))
        zW = np.where(mW, Z[safe_iW], np.where(mE, 2.0 * Z - Z[safe_iE], Z))
        zN = np.where(mN, Z[safe_iN], np.where(mS, 2.0 * Z - Z[safe_iS], Z))
        zS = np.where(mS, Z[safe_iS], np.where(mN, 2.0 * Z - Z[safe_iN], Z))

        xNE = np.where(mNE, X[safe_iNE], xN + xE - X)
        xNW = np.where(mNW, X[safe_iNW], xN + xW - X)
        xSE = np.where(mSE, X[safe_iSE], xS + xE - X)
        xSW = np.where(mSW, X[safe_iSW], xS + xW - X)

        zNE = np.where(mNE, Z[safe_iNE], zN + zE - Z)
        zNW = np.where(mNW, Z[safe_iNW], zN + zW - Z)
        zSE = np.where(mSE, Z[safe_iSE], zS + zE - Z)
        zSW = np.where(mSW, Z[safe_iSW], zS + zW - Z)

        X_xi = 0.5 * (xE - xW)
        X_eta = 0.5 * (xN - xS)
        Z_xi = 0.5 * (zE - zW)
        Z_eta = 0.5 * (zN - zS)

        X_xita = 0.25 * (xNE - xNW - xSE + xSW)
        Z_xita = 0.25 * (zNE - zNW - zSE + zSW)

        alpha = X_eta**2 + Z_eta**2
        beta = X_xi * X_eta + Z_xi * Z_eta
        gamma = X_xi**2 + Z_xi**2

        DC = h
        DE = np.where(movable_E, h[safe_iE], DC)
        DW = np.where(movable_W, h[safe_iW], DC)
        DN = np.where(movable_N, h[safe_iN], DC)
        DS = np.where(movable_S, h[safe_iS], DC)

        D_phi = 0.5 * (DE - DW)
        D_psi = 0.5 * (DN - DS)
        J = X_xi * Z_eta - X_eta * Z_xi
        j_dc = J / (DC + 1e-12)
        Source_X = j_dc * (D_phi * Z_eta - D_psi * Z_xi)
        Source_Z = j_dc * (D_psi * X_xi - D_phi * X_eta)
        Denominator = 2.0 * (alpha + gamma)
        valid_denom_mask = Denominator >= 1e-12
        Denom_safe = np.where(valid_denom_mask, Denominator, 1.0)  # Prevent DivZero

        X_target = np.where(
            valid_denom_mask & (move_X_mask | move_hyperellipse_mask),
            (alpha * (xE + xW) + gamma * (xN + xS) - 2.0 * beta * X_xita - Source_X) / Denom_safe,
            X
        )
        Z_target = np.where(
            valid_denom_mask & (move_Z_mask | move_hyperellipse_mask),
            (alpha * (zE + zW) + gamma * (zN + zS) - 2.0 * beta * Z_xita - Source_Z) / Denom_safe,
            Z
        )

        dx = X_target - xc_e
        dz = Z_target - zc_e
        zero_dist_mask = (np.abs(dx) < 1e-12) & (np.abs(dz) < 1e-12)
        dist_term = np.abs(dx / a_e)**n_e + np.abs(dz / b_e)**n_e
        dist_safe = np.where(dist_term < 1e-12, 1.0, dist_term)  # Prevent DivZero
        scale = dist_safe ** (-1.0 / n_e)
        X_proj = xc_e + scale * dx
        Z_proj = zc_e + scale * dz
        X_hyperellipse_new = np.where(zero_dist_mask, X, (1.0 - omega) * X + omega * X_proj)
        Z_hyperellipse_new = np.where(zero_dist_mask, Z, (1.0 - omega) * Z + omega * Z_proj)
        X_reg_new = np.where(move_X_mask, (1.0 - omega) * X + omega * X_target, X)
        Z_reg_new = np.where(move_Z_mask, (1.0 - omega) * Z + omega * Z_target, Z)
        X = np.where(move_hyperellipse_mask, X_hyperellipse_new, X_reg_new)
        Z = np.where(move_hyperellipse_mask, Z_hyperellipse_new, Z_reg_new)

    return X, Z


def winslow_smooth_vectorized(
    points: np.ndarray,
    quads: np.ndarray,
    segy_grid_x: np.ndarray,
    segy_grid_z: np.ndarray,
    segy_grid_vals: np.ndarray,
    move_all: set,
    move_X_only: set,
    move_Z_only: set,
    move_hyperellipse: set = None,
    hyperellipse_params: tuple = None,
    iterations: int = 1500,
    omega: float = 0.05
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
        2D array of element sizing values.
    move_all : set
        Set of node indices permitted to move in both X and Z directions.
    move_X_only : set
        Set of node indices constrained to move only along the X-axis.
    move_Z_only : set
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
            n1 = quad[i]
            n2 = quad[(i + 1) % 4]
            node_to_neighbors[n1].add(n2)
            node_to_neighbors[n2].add(n1)

    stencils = {}
    for i in range(num_nodes):
        neighbors = list(node_to_neighbors[i])
        if not neighbors:
            stencils[i] = {d: None for d in ['E', 'N', 'W', 'S', 'NE', 'NW', 'SE', 'SW']}
            continue

        vecs = points[neighbors] - points[i]
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        sort_idx = np.argsort(angles)
        sorted_neighbors = [neighbors[idx] for idx in sort_idx]
        sorted_angles = angles[sort_idx]
        cardinals = {'E': None, 'N': None, 'W': None, 'S': None}

        if len(sorted_neighbors) == 4:
            cardinals['E'] = sorted_neighbors[0]
            cardinals['N'] = sorted_neighbors[1]
            cardinals['W'] = sorted_neighbors[2]
            cardinals['S'] = sorted_neighbors[3]
        elif len(sorted_neighbors) > 0:
            diffs = np.diff(sorted_angles)
            diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2 * np.pi)
            gap_idx = np.argmax(diffs)
            rolled_neighbors = np.roll(sorted_neighbors, -(gap_idx + 1))

            if len(sorted_neighbors) == 3:
                cardinals['E'] = rolled_neighbors[0]
                cardinals['N'] = rolled_neighbors[1]
                cardinals['W'] = rolled_neighbors[2]
            elif len(sorted_neighbors) == 2:
                cardinals['E'] = rolled_neighbors[0]
                cardinals['N'] = rolled_neighbors[1]

        def get_shared(nA, nB):
            if nA is None or nB is None:
                return None
            shared = node_to_neighbors[nA].intersection(node_to_neighbors[nB])
            shared.discard(i)
            return list(shared)[0] if shared else None

        corners = {
            'NE': get_shared(cardinals['N'], cardinals['E']),
            'NW': get_shared(cardinals['N'], cardinals['W']),
            'SE': get_shared(cardinals['S'], cardinals['E']),
            'SW': get_shared(cardinals['S'], cardinals['W'])
        }
        stencils[i] = {**cardinals, **corners}

    dir_map = {'E': 0, 'W': 1, 'N': 2, 'S': 3, 'NE': 4, 'NW': 5, 'SE': 6, 'SW': 7}
    stencil_arr = np.full((num_nodes, 8), -1, dtype=np.int32)

    for i, st in stencils.items():
        for d, col_idx in dir_map.items():
            if st[d] is not None:
                stencil_arr[i, col_idx] = st[d]

    movable_nodes = move_all | move_X_only | move_Z_only | move_hyperellipse
    is_movable = np.zeros(num_nodes, dtype=np.bool_)
    is_movable[list(movable_nodes)] = True

    move_X_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_Z_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_hyperellipse_mask = np.zeros(num_nodes, dtype=np.bool_)

    move_X_mask[list(move_all | move_X_only)] = True
    move_Z_mask[list(move_all | move_Z_only)] = True
    move_hyperellipse_mask[list(move_hyperellipse)] = True

    X = np.ascontiguousarray(points[:, 0], dtype=np.float64)
    Z = np.ascontiguousarray(points[:, 1], dtype=np.float64)

    print(f"Starting {iterations} Fast Winslow iterations...")

    print_interval = 50
    current_iter = 0

    while current_iter < iterations:
        step_iters = min(print_interval, iterations - current_iter)

        X, Z = _vectorized_winslow_loop(
            X, Z,
            segy_grid_x, segy_grid_z, segy_grid_vals,
            stencil_arr, is_movable, move_X_mask, move_Z_mask, move_hyperellipse_mask, e_params, omega, step_iters
        )

        current_iter += step_iters
        percent = (current_iter / iterations) * 100
        print(f"Iteration: {current_iter} / {iterations} [{percent:.1f}%]   ", end='\r')

    print("\nSmoothing Complete.")
    return np.column_stack((X, Z))
