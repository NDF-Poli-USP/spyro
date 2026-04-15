import numpy as np
import gmsh
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator

def create_sizing_function(fname, hmin=None, bbox=None, wl=10, freq=2, pad_type=None, pad_size_x=-1.0, pad_size_z=-1.0, grade=None, vp_water=None):
    """Create a mesh sizing function from a SEGY velocity model.

    This function reads a SEGY file, extracts the velocity model, applies
    optional water velocity substitution, and calculates an appropriate
    element size field based on wavelength constraints and padding.

    Parameters
    ----------
    fname : str
        Filename of the SEGY velocity model.
    hmin : float, optional
        Minimum element size allowed. Element sizes below this will be clipped.
    bbox : tuple
        Bounding box tuple in the format (zmin, zmax, xmin, xmax).
    wl : int, optional
        Number of elements per wavelength. Default is 10.
    freq : float, optional
        Maximum source frequency in Hz. Default is 2.
    pad_type : str, optional
        Type of padding to apply ('rectangular', 'elliptical', or None).
    pad_size_x : float, optional
        Size of padding in the x-direction. Default is -1.0.
    pad_size_z : float, optional
        Size of padding in the z-direction. Default is -1.0.
    grade : float, optional
        Grading parameter for applying a 2D Savitzky-Golay filter to smooth
        transitions. Default is None.
    vp_water : float, optional
        Velocity value to substitute where the SEGY model has zeros
        (representing water). If None, defaults to 1500.0.

    Returns
    -------
    tuple
        A tuple containing:
        - sizing_function (callable): A function f(x, y) returning element size.
        - min_val (float): Minimum element size in the domain.
        - max_val (float): Maximum element size in the domain.
        - n_samples (int): Number of depth samples.
        - n_traces (int): Number of lateral traces.
    """
    
    # Read velocity model with provided bbox
    vp, n_samples, n_traces = read_segy_velocity_model(fname)
    # Set water velocity if value = 0
    if vp_water is not None:
        vp = np.where(vp == 0, vp_water, vp)
    else:
        vp = np.where(vp == 0, 1500.0, vp)
    # Calculate wavelength-based sizing
    cell_size = calculate_wavelength_sizing(vp, wl, freq)
    # Enforce minimum element size
    if hmin is not None:
        cell_size = np.maximum(cell_size, hmin) #if hmin is less than function miminum, it will not set

    #Applying padding
    if (pad_type == "rectangular" or pad_type == "elliptical" ):
        dz=(bbox[1]-bbox[0])/n_traces
        dx=(bbox[3]-bbox[2])/n_samples
        nnz = int(pad_size_z / dz)
        nnx = int(pad_size_x / dx)
        print(nnx,nnz,n_samples,n_traces)
        print(pad_size_z,bbox[0],bbox[2])
        padding = ((0, nnz), (nnx, nnx))
        cell_size = np.pad(cell_size, padding, "edge")
        bbox = (
                bbox[0] - pad_size_z,
                bbox[1],
                bbox[2] - pad_size_x,
                bbox[3] + pad_size_x,
            )

    print(cell_size.shape[0])
    print(cell_size.shape[1])
    if grade is not None:
        window_length_z = int((1.0 - grade) *0.1* cell_size.shape[0])
        window_length_x = int((1.0 - grade) *0.1* cell_size.shape[1])
        cell_size = apply_savitzky_golay_filter_2d(cell_size,window_length_x,window_length_z )
        
    print("Function Minimum and Maximum values:")
    print(cell_size.min(),cell_size.max())
    
    # Create interpolation function
    def sizing_function(x, y):
        """Return element size at position (x, y)
        
        Parameters:
        -----------
        x, y : float or array-like
            Coordinates where to evaluate element size
            
        Returns:
        --------
        float or array
            Element size at the given position(s)
        """
        return interpolate_size(x, y, cell_size, bbox)
    
    return sizing_function,cell_size.min(),cell_size.max(),n_samples,n_traces

def read_segy_velocity_model(fname):
    """Read a velocity model array from a SEGY file.

    Parameters
    ----------
    fname : str
        Path to the SEGY filename.

    Returns
    -------
    tuple
        A tuple containing:
        - vp (ndarray): 2D Velocity model array of shape (n_samples, n_traces).
        - n_traces (int): Number of traces (columns).
        - n_samples (int): Number of samples per trace (rows).
    """
    import segyio
    import numpy as np
    
    print(f"Reading SEGY file: {fname}")
    
    # Open SEGY file
    with segyio.open(fname, 'r', ignore_geometry=True) as segy:
         
        n_traces = len(segy.trace)
        n_samples = len(segy.samples)
        
        # Read traces directly into array
        vp = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            vp[:, i] = segy.trace[i]
    print(f"Final velocity range: {vp.min():.1f} - {vp.max():.1f}")
    return vp,n_traces,n_samples

def calculate_wavelength_sizing(vp, wl, freq):
    """Calculate the target element size based on a wavelength criterion.

    Parameters
    ----------
    vp : ndarray
        2D array representing the velocity model.
    wl : int
        Desired number of elements per wavelength.
    freq : float
        Maximum frequency in Hz for the simulation.

    Returns
    -------
    ndarray
        2D array of the calculated element sizes based on the velocity field.
    """
    # Wavelength = velocity / frequency
    wavelength = vp / freq
    
    # Element size = wavelength / number of elements per wavelength
    cell_size = wavelength / wl
    
    return cell_size

def interpolate_size(x, y, cell_size, bbox):
    """Interpolate element size at specific spatial coordinates.

    Uses RegularGridInterpolator to return the interpolated
    element size at a specific value.

    Parameters
    ----------
    x : float or array-like
        x-coordinates (lateral) where to evaluate the element size.
    y : float or array-like
        y-coordinates (depth/z-axis) where to evaluate the element size.
    cell_size : ndarray
        2D array of pre-calculated element sizes.
    bbox : tuple
        Bounding box tuple defining the domain boundaries (zmin, zmax, xmin, xmax).

    Returns
    -------
    float or ndarray
        Interpolated element size(s) at the given spatial positions.
    """
    # Create coordinate arrays
    z_coords = np.linspace(bbox[0], bbox[1], cell_size.shape[0])
    x_coords = np.linspace(bbox[2], bbox[3], cell_size.shape[1])
    cell_size_flipped = np.flipud(cell_size) 
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (z_coords, x_coords), 
        cell_size_flipped, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    
    # Handle scalar or array inputs
    if np.isscalar(x) and np.isscalar(y):
        points = np.array([[y, x]])  # Note: y corresponds to z (depth), x to x
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        points = np.column_stack([y.ravel(), x.ravel()])
    
    result = interpolator(points)
    
    # Return scalar if input was scalar
    if np.isscalar(x) and np.isscalar(y):
        return float(result[0])
    else:
        return result.reshape(x.shape)


def apply_savitzky_golay_filter_2d(grid_values, window_length_x=501, window_length_z=501, polyorder=3):
    """Apply a 2D Savitzky-Golay filter to a grid of values.

    Smooths a 2D array by applying 1D Savitzky-Golay filters sequentially
    along the columns and rows to grade element sizes and prevent abrupt
    transitions.

    Parameters
    ----------
    grid_values : array-like
        Input 2D array of grid values to be filtered.
    window_length_x : int, optional
        The length of the filter window along the x-axis. Must be odd and positive.
        Default is 501.
    window_length_z : int, optional
        The length of the filter window along the z-axis. Must be odd and positive.
        Default is 501.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Must be less
        than the window lengths. Default is 3.

    Returns
    -------
    ndarray
        Filtered 2D array with the same shape as the input.

    Raises
    ------
    ValueError
        If window lengths are not odd, not positive, or if polyorder is too large.
    """
    
    # Convert input to numpy array
    grid_values = np.asarray(grid_values)
    
    rows, cols = grid_values.shape
    print(window_length_z,window_length_x)
    # First filter along axis 0 (columns), then along axis 1 (rows)
    filtered_values = savgol_filter(grid_values, window_length_x, polyorder, axis=0)
    filtered_values = savgol_filter(filtered_values, window_length_z, polyorder, axis=1)
    
    return filtered_values

from pathlib import Path
import segyio

def generate_water_profile_from_segy(segy_path, z_min, z_max, x_min, x_max, value=0.0, tolerance=1e-3, x_chunk=1024):
    """Generate a water interface profile from a SEGY file.

    Reads a 2D SEGY file, computes grid spacing from physical domain bounds,
    and searches down the z-axis to find the water interface.

    Parameters
    ----------
    segy_path : str or pathlib.Path
        Path to the SEGY velocity model.
    z_min : float
        Minimum z-coordinate of the physical domain.
    z_max : float
        Maximum z-coordinate of the physical domain.
    x_min : float
        Minimum x-coordinate of the physical domain.
    x_max : float
        Maximum x-coordinate of the physical domain.
    value : float, optional
        Velocity value identifying the water layer. Default is 0.0.
    tolerance : float, optional
        Numerical tolerance for matching the water velocity value. Default is 1e-3.
    x_chunk : int, optional
        Number of traces to process in memory at once. Default is 1024.

    Returns
    -------
    tuple
        A tuple containing:
        - Xs (ndarray): 1D array of physical X coordinates.
        - Z_bottom (ndarray): 1D array of the corresponding Z coordinates
          representing the water interface depth.

    Raises
    ------
    FileNotFoundError
        If the specified SEGY file does not exist.
    """
    segy_path = Path(segy_path)
    if not segy_path.exists():
        raise FileNotFoundError(f"SEGY file not found: {segy_path}")

    print(f"Reading SEGY file: {segy_path}")

    # Extract SEGY Data and Compute Spacing 
    with segyio.open(segy_path, 'r', ignore_geometry=True) as segy:
        nx_tot = segy.tracecount
        nz_tot = len(segy.samples)

        total_length_z = abs(float(z_max) - float(z_min))
        _dz = total_length_z / (nz_tot - 1) if nz_tot > 1 else total_length_z

        total_length_x = abs(float(x_max) - float(x_min))
        _dx = total_length_x / (nx_tot - 1) if nx_tot > 1 else total_length_x

        print(f"Detected parameters: nx={nx_tot}, nz={nz_tot}")
        print(f"Calculated spacing: dx={_dx:.2f} m, dz={_dz:.2f} m")

        plane_zx = segy.trace.raw[:].T

    # Domain Ranges
    z_top = float(max(z_min, z_max))
    z_bot = float(min(z_min, z_max))
    x_low, x_high = sorted([float(x_min), float(x_max)])

    Lx = (nx_tot - 1) * _dx
    Lz = (nz_tot - 1) * _dz

    x_low  = np.clip(x_low,  0.0, Lx)
    x_high = np.clip(x_high, 0.0, Lx)
    z_top  = np.clip(z_top,  -Lz, 0.0)
    z_bot  = np.clip(z_bot,  -Lz, 0.0)

    # Map meters → nearest indices
    ix_min = int(np.rint(x_low  / _dx))
    ix_max = int(np.rint(x_high / _dx))
    ix_min, ix_max = max(0, min(ix_min, nx_tot - 1)), max(0, min(ix_max, nx_tot - 1))
    if ix_max < ix_min: 
        ix_min, ix_max = ix_max, ix_min

    iz_top = int(np.rint(-z_top / _dz))
    iz_bot = int(np.rint(-z_bot / _dz))
    iz_top = max(0, min(iz_top, nz_tot - 1))
    iz_bot = max(0, min(iz_bot, nz_tot - 1))
    if iz_bot < iz_top:
        iz_top, iz_bot = iz_bot, iz_top

    # Physical X coordinates for the points
    Xs = _dx * np.arange(ix_min, ix_max + 1, dtype=np.float64)
    Nx = (ix_max - ix_min)
    x_chunk = int(max(1, x_chunk))

    Z_bottom = np.empty(Nx + 1, dtype=np.float64)
    target = float(value)
    tol = float(tolerance)

    # Water Interface Search 
    for xs in range(0, Nx + 1, x_chunk):
        xe = min(xs + x_chunk, Nx + 1)
        ix_tile = ix_min + np.arange(xs, xe, dtype=int)
        
        block = plane_zx[iz_top:iz_bot + 1, ix_tile]
        block = np.asarray(block, dtype=np.float32)

        in_water  = np.abs(block - target) <= tol 
        non_water = ~in_water

        any_non_water = np.any(non_water, axis=0) 
        first_non_idx = np.argmax(non_water, axis=0).astype(np.int32) 

        first_non_idx = np.where(any_non_water, first_non_idx, (iz_bot - iz_top)).astype(np.int32)

        k_global = iz_top + first_non_idx
        z_phys = -k_global.astype(np.float64) * _dz
        Z_bottom[xs:xe] = z_phys

    # Calculate Final Shifted Coordinates 
    x_shift = float(min(x_min, x_max))
    Xs = Xs + x_shift  

    return Xs, Z_bottom


from typing import Callable
from scipy.interpolate import griddata

def winslow_smooth_default(
    points: np.ndarray,
    quads: np.ndarray,
    sizing_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    move_all: set,
    move_X_only: set,
    move_Z_only: set,
    move_ellipse: set = None,        
    ellipse_params: tuple = None,   
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
    move_ellipse : set, optional
        Set of node indices constrained to move along an elliptical boundary.
    ellipse_params : tuple, optional
        Tuple containing (a, b, xc, zc, n) defining an superellipse boundary.
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
    
    if move_ellipse is None:
        move_ellipse = set()

    num_nodes = len(points)
    
    node_to_neighbors = {i: set() for i in range(num_nodes)}
    for quad in quads:
        for i in range(4):
            n1 = quad[i]
            n2 = quad[(i+1)%4]
            node_to_neighbors[n1].add(n2)
            node_to_neighbors[n2].add(n1)

    stencils = {}
    
    for i in range(num_nodes):
        neighbors = list(node_to_neighbors[i])
        if not neighbors: continue
            
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
            diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2*np.pi)
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
            if nA is None or nB is None: return None
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

    movable_nodes = move_all | move_X_only | move_Z_only | move_ellipse

    # Array to store the J / D_C values
    j_dc_arr = np.zeros(num_nodes)

    jj = 0
    for _ in range(iterations):
        jj += 1
        print(f"Iteration: {jj}", end='\r')
        
        h = sizing_fn(X, Z) 
        
        for i, st in stencils.items():
            is_all     = i in move_all
            is_X       = i in move_X_only
            is_Z       = i in move_Z_only
            is_ellipse = i in move_ellipse
            
            # Skip if completely locked
            if not (is_all or is_X or is_Z or is_ellipse): continue
            
            C = {'X': {}, 'Z': {}}
            
            for d, opp in [('E', 'W'), ('W', 'E'), ('N', 'S'), ('S', 'N')]:
                idx = st[d]
                if idx is not None:
                    C['X'][d] = X[idx]; C['Z'][d] = Z[idx]
                else:
                    opp_idx = st[opp]
                    C['X'][d] = 2*X[i] - X[opp_idx] if opp_idx is not None else X[i]
                    C['Z'][d] = 2*Z[i] - Z[opp_idx] if opp_idx is not None else Z[i]
                    
            for d, d1, d2 in [('NE','N','E'), ('NW','N','W'), ('SE','S','E'), ('SW','S','W')]:
                idx = st[d]
                if idx is not None:
                    C['X'][d] = X[idx]; C['Z'][d] = Z[idx]
                else:
                    C['X'][d] = C['X'][d1] + C['X'][d2] - X[i]
                    C['Z'][d] = C['Z'][d1] + C['Z'][d2] - Z[i]

            X_E, X_W, X_N, X_S = C['X']['E'], C['X']['W'], C['X']['N'], C['X']['S']
            Z_E, Z_W, Z_N, Z_S = C['Z']['E'], C['Z']['W'], C['Z']['N'], C['Z']['S']
            
            X_xi  = 0.5 * (X_E - X_W)
            X_eta = 0.5 * (X_N - X_S)
            Z_xi  = 0.5 * (Z_E - Z_W)
            Z_eta = 0.5 * (Z_N - Z_S)
            
            alpha = X_eta**2 + Z_eta**2
            beta  = X_xi * X_eta + Z_xi * Z_eta
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
            
            if Denominator < 1e-12: continue

            # --- Target Calculation ---
            X_target = X[i]
            Z_target = Z[i]
            
            if is_all or is_X or is_ellipse:
                X_target = (alpha * (X_E + X_W) + gamma * (X_N + X_S) - 2.0 * beta * X_xita - Source_X) / Denominator
                
            if is_all or is_Z or is_ellipse:
                Z_target = (alpha * (Z_E + Z_W) + gamma * (Z_N + Z_S) - 2.0 * beta * Z_xita - Source_Z) / Denominator

            if is_ellipse and ellipse_params is not None:
                a_e, b_e, xc_e, zc_e, n_e = ellipse_params
                dx = X_target - xc_e
                dz = Z_target - zc_e
                
                # Prevent division by zero if node lands on center
                if abs(dx) < 1e-12 and abs(dz) < 1e-12:
                    X_new[i] = X[i]
                    Z_new[i] = Z[i]
                else:
                    # Find scale to snap back to the boundary using L-n norm
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

from typing import Callable

# Attempt to import Numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Only define the JIT-compiled functions if Numba was found
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def bilinear_interp_2d(x, z, grid_x, grid_z, grid_vals):
        """Perform 2D bilinear interpolation for a single point.

        Designed to be compiled via Numba for fast execution within iterative
        mesh smoothing loops.

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

        # Detect storage layout
        vals_are_nx_nz = (gv0 == nx and gv1 == nz)
        vals_are_nz_nx = (gv0 == nz and gv1 == nx)

        if not vals_are_nx_nz and not vals_are_nz_nx:
            raise ValueError("grid_vals shape must be (len(grid_x), len(grid_z)) or (len(grid_z), len(grid_x)).")

        x_asc = grid_x[0] < grid_x[nx - 1]
        z_asc = grid_z[0] < grid_z[nz - 1]

        # Reject repeated endpoints / zero-length domains
        if grid_x[0] == grid_x[nx - 1]:
            raise ValueError("grid_x domain is degenerate.")
        if grid_z[0] == grid_z[nz - 1]:
            raise ValueError("grid_z domain is degenerate.")

        # Clamp query point to domain
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

        # Find cell index in x
        if x_asc:
            i = np.searchsorted(grid_x, xq, side='right') - 1
        else:
            # logical ascending index on descending array
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

        # Find cell index in z
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

        # Logical cell bounds in ascending order
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

        # Safety against roundoff excursions
        if xd < 0.0:
            xd = 0.0
        elif xd > 1.0:
            xd = 1.0

        if zd < 0.0:
            zd = 0.0
        elif zd > 1.0:
            zd = 1.0

        # Fetch corner values
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
                             move_X, move_Z, move_ellipse, ellipse_params, omega, iterations):
        """Internal Numba-accelerated inner loop for Winslow smoothing."""
        num_nodes = len(X)
        X_new = np.empty_like(X)
        Z_new = np.empty_like(Z)
        h = np.empty_like(X)
        
        # ellipse parameters
        a_e, b_e, xc_e, zc_e, n_e = ellipse_params[0], ellipse_params[1], ellipse_params[2], ellipse_params[3], ellipse_params[4]

        for jj in range(iterations):
            for i in prange(num_nodes):
                h[i] = bilinear_interp_2d(X[i], Z[i], grid_x, grid_z, grid_vals)

            for i in prange(num_nodes):
                X_new[i] = X[i]
                Z_new[i] = Z[i]

                if not (move_X[i] or move_Z[i] or move_ellipse[i]): 
                    continue

                iE, iW, iN, iS = stencils[i, 0], stencils[i, 1], stencils[i, 2], stencils[i, 3]
                iNE, iNW, iSE, iSW = stencils[i, 4], stencils[i, 5], stencils[i, 6], stencils[i, 7]

                # Cardinals
                xE = X[iE] if iE != -1 else (2.0*X[i] - X[iW] if iW != -1 else X[i])
                xW = X[iW] if iW != -1 else (2.0*X[i] - X[iE] if iE != -1 else X[i])
                xN = X[iN] if iN != -1 else (2.0*X[i] - X[iS] if iS != -1 else X[i])
                xS = X[iS] if iS != -1 else (2.0*X[i] - X[iN] if iN != -1 else X[i])

                zE = Z[iE] if iE != -1 else (2.0*Z[i] - Z[iW] if iW != -1 else Z[i])
                zW = Z[iW] if iW != -1 else (2.0*Z[i] - Z[iE] if iE != -1 else Z[i])
                zN = Z[iN] if iN != -1 else (2.0*Z[i] - Z[iS] if iS != -1 else Z[i])
                zS = Z[iS] if iS != -1 else (2.0*Z[i] - Z[iN] if iN != -1 else Z[i])

                # Corners
                xNE = X[iNE] if iNE != -1 else (xN + xE - X[i])
                xNW = X[iNW] if iNW != -1 else (xN + xW - X[i])
                xSE = X[iSE] if iSE != -1 else (xS + xE - X[i])
                xSW = X[iSW] if iSW != -1 else (xS + xW - X[i])

                zNE = Z[iNE] if iNE != -1 else (zN + zE - Z[i])
                zNW = Z[iNW] if iNW != -1 else (zN + zW - Z[i])
                zSE = Z[iSE] if iSE != -1 else (zS + zE - Z[i])
                zSW = Z[iSW] if iSW != -1 else (zS + zW - Z[i])

                # Derivatives
                X_xi  = 0.5 * (xE - xW)
                X_eta = 0.5 * (xN - xS)
                Z_xi  = 0.5 * (zE - zW)
                Z_eta = 0.5 * (zN - zS)
                
                X_xita = 0.25 * (xNE - xNW - xSE + xSW)
                Z_xita = 0.25 * (zNE - zNW - zSE + zSW)
                
                alpha = X_eta**2 + Z_eta**2
                beta  = X_xi * X_eta + Z_xi * Z_eta
                gamma = X_xi**2 + Z_xi**2

                # Sizing Function Values
                DC = h[i]
                DE = h[iE] if (iE != -1 and is_movable[iE]) else DC
                DW = h[iW] if (iW != -1 and is_movable[iW]) else DC
                DN = h[iN] if (iN != -1 and is_movable[iN]) else DC
                DS = h[iS] if (iS != -1 and is_movable[iS]) else DC

                D_phi = 0.5 * (DE - DW)
                D_psi = 0.5 * (DN - DS)

                # Sources and targets
                J = X_xi * Z_eta - X_eta * Z_xi
                j_dc = J / (DC + 1e-12)

                Source_X = j_dc * (D_phi * Z_eta - D_psi * Z_xi)
                Source_Z = j_dc * (D_psi * X_xi - D_phi * X_eta) 

                Denominator = 2.0 * (alpha + gamma)
                if Denominator < 1e-12: 
                    continue

                # Compute Raw Targets
                X_target = X[i]
                Z_target = Z[i]
                
                if move_X[i] or move_ellipse[i]:
                    X_target = (alpha * (xE + xW) + gamma * (xN + xS) - 2.0 * beta * X_xita - Source_X) / Denominator
                
                if move_Z[i] or move_ellipse[i]:
                    Z_target = (alpha * (zE + zW) + gamma * (zN + zS) - 2.0 * beta * Z_xita - Source_Z) / Denominator

                # Apply relaxation with Elliptical constraints
                if move_ellipse[i]:
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
        move_ellipse: set = None,        
        ellipse_params: tuple = None,    
        iterations: int = 1500,
        omega: float = 0.05
    ) -> np.ndarray:
        """Apply Winslow mesh smoothing using Numba JIT compilation.

        Accelerates the Winslow iteration loop by compiling the
        stencil operations, derivative calculations, and bilinear interpolation.

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
        move_ellipse : set, optional
            Set of node indices constrained to move along an elliptical boundary.
        ellipse_params : tuple, optional
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
        
        if move_ellipse is None:
            move_ellipse = set()
            
        if ellipse_params is None:
            # Dummy values 
            e_params = np.array([1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.float64)
        else:
            e_params = np.array(ellipse_params, dtype=np.float64)

        num_nodes = len(points)
        
        node_to_neighbors = {i: set() for i in range(num_nodes)}
        for quad in quads:
            for i in range(4):
                n1 = quad[i]
                n2 = quad[(i+1)%4]
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
                diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2*np.pi)
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
                if nA is None or nB is None: return None
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

        # Movement constraint masks
        movable_nodes = move_all | move_X_only | move_Z_only | move_ellipse
        is_movable = np.zeros(num_nodes, dtype=np.bool_)
        is_movable[list(movable_nodes)] = True
        
        move_X_mask = np.zeros(num_nodes, dtype=np.bool_)
        move_Z_mask = np.zeros(num_nodes, dtype=np.bool_)
        move_ellipse_mask = np.zeros(num_nodes, dtype=np.bool_)
        
        move_X_mask[list(move_all | move_X_only)] = True
        move_Z_mask[list(move_all | move_Z_only)] = True
        move_ellipse_mask[list(move_ellipse)] = True

        X = np.ascontiguousarray(points[:, 0], dtype=np.float64)
        Z = np.ascontiguousarray(points[:, 1], dtype=np.float64)

        print(f"Starting {iterations} Winslow iterations...")
        
        print_interval = 50 # Update the console every 50 iterations
        current_iter = 0
        
        while current_iter < iterations:
            step_iters = min(print_interval, iterations - current_iter)
            
            X, Z = _numba_winslow_loop2(
                X, Z, 
                segy_grid_x, segy_grid_z, segy_grid_vals, 
                stencil_arr, is_movable, move_X_mask, move_Z_mask, move_ellipse_mask, e_params, omega, step_iters
            )
            
            current_iter += step_iters
            
            percent = (current_iter / iterations) * 100
            print(f"Iteration: {current_iter} / {iterations} [{percent:.1f}%]   ", end='\r')

        print("\nSmoothing Complete.")
        return np.column_stack((X, Z))

# Define a fallback if Numba is not available 
else:
    def winslow_smooth_numba(*args, **kwargs):
        raise ImportError(
            "Numba is not installed in this environment. "
            "Please run 'pip install numba' to use the 'numba' implementation, "
            "or switch configuration to winslow_implementation='fast' or 'default'."
        )
    def _numba_winslow_loop2(*args, **kwargs):
        raise ImportError(
            "Numba is not installed in this environment. "
            "Please run 'pip install numba' to use the 'numba' implementation, "
            "or switch configuration to winslow_implementation='fast' or 'default'."
        )   
    def bilinear_interp_2d(*args, **kwargs):
        raise ImportError(
            "Numba is not installed in this environment. "
            "Please run 'pip install numba' to use the 'numba' implementation, "
            "or switch configuration to winslow_implementation='fast' or 'default'."
        )

import numpy as np
from typing import Callable, Tuple

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

    # Normalize grid_vals shape to (nx, nz)
    if grid_vals.shape == (nz, nx):
        grid_vals = grid_vals.T
    elif grid_vals.shape != (nx, nz):
        raise ValueError(
            f"grid_vals has shape {grid_vals.shape}, but expected "
            f"({nx}, {nz}) or ({nz}, {nx})."
        )

    # Check for repeated coordinates
    if np.any(np.diff(grid_x) == 0.0):
        raise ValueError("grid_x contains repeated coordinates.")
    if np.any(np.diff(grid_z) == 0.0):
        raise ValueError("grid_z contains repeated coordinates.")

    # Make axes ascending for searchsorted
    if grid_x[0] > grid_x[-1]:
        grid_x = grid_x[::-1]
        grid_vals = grid_vals[::-1, :]

    if grid_z[0] > grid_z[-1]:
        grid_z = grid_z[::-1]
        grid_vals = grid_vals[:, ::-1]

    # Clip query points to domain to avoid extrapolation
    xq = np.clip(x, grid_x[0], grid_x[-1])
    zq = np.clip(z, grid_z[0], grid_z[-1])

    # Find containing cell
    i = np.searchsorted(grid_x, xq, side="right") - 1
    j = np.searchsorted(grid_z, zq, side="right") - 1

    i = np.clip(i, 0, nx - 2)
    j = np.clip(j, 0, nz - 2)

    # Cell corners
    x0 = grid_x[i]
    x1 = grid_x[i + 1]
    z0 = grid_z[j]
    z1 = grid_z[j + 1]

    dx = x1 - x0
    dz = z1 - z0

    # Local coordinates in [0, 1]
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

    # Values at corners
    c00 = grid_vals[i,     j    ]
    c10 = grid_vals[i + 1, j    ]
    c01 = grid_vals[i,     j + 1]
    c11 = grid_vals[i + 1, j + 1]

    # Bilinear interpolation
    c0 = (1.0 - xd) * c00 + xd * c10
    c1 = (1.0 - xd) * c01 + xd * c11
    out = (1.0 - zd) * c0 + zd * c1

    return out

def _vectorized_winslow_loop(
    X: np.ndarray, Z: np.ndarray, 
    grid_x: np.ndarray, grid_z: np.ndarray, grid_vals: np.ndarray, 
    stencil_arr: np.ndarray, is_movable: np.ndarray, 
    move_X_mask: np.ndarray, move_Z_mask: np.ndarray, move_ellipse_mask: np.ndarray, 
    ellipse_params: np.ndarray, omega: float, iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal NumPy-vectorized inner loop for fast Winslow smoothing."""
    
    # ellipse parameters
    a_e, b_e, xc_e, zc_e, n_e = ellipse_params

    # Unpack stencil columns
    iE, iW, iN, iS = stencil_arr[:, 0], stencil_arr[:, 1], stencil_arr[:, 2], stencil_arr[:, 3]
    iNE, iNW, iSE, iSW = stencil_arr[:, 4], stencil_arr[:, 5], stencil_arr[:, 6], stencil_arr[:, 7]

    # Valid neighbor masks
    mE, mW = iE != -1, iW != -1
    mN, mS = iN != -1, iS != -1
    mNE, mNW = iNE != -1, iNW != -1
    mSE, mSW = iSE != -1, iSW != -1

    # Safe indices 
    safe_iE, safe_iW = np.maximum(iE, 0), np.maximum(iW, 0)
    safe_iN, safe_iS = np.maximum(iN, 0), np.maximum(iS, 0)
    safe_iNE, safe_iNW = np.maximum(iNE, 0), np.maximum(iNW, 0)
    safe_iSE, safe_iSW = np.maximum(iSE, 0), np.maximum(iSW, 0)

    # Sizing masks
    movable_E = mE & is_movable[safe_iE]
    movable_W = mW & is_movable[safe_iW]
    movable_N = mN & is_movable[safe_iN]
    movable_S = mS & is_movable[safe_iS]

    for _ in range(iterations):
        # Update Sizing Function
        h = vectorized_bilinear_interp(X, Z, grid_x, grid_z, grid_vals)

        # Extract Cardinal Coordinate Arrays
        xE = np.where(mE, X[safe_iE], np.where(mW, 2.0*X - X[safe_iW], X))
        xW = np.where(mW, X[safe_iW], np.where(mE, 2.0*X - X[safe_iE], X))
        xN = np.where(mN, X[safe_iN], np.where(mS, 2.0*X - X[safe_iS], X))
        xS = np.where(mS, X[safe_iS], np.where(mN, 2.0*X - X[safe_iN], X))

        zE = np.where(mE, Z[safe_iE], np.where(mW, 2.0*Z - Z[safe_iW], Z))
        zW = np.where(mW, Z[safe_iW], np.where(mE, 2.0*Z - Z[safe_iE], Z))
        zN = np.where(mN, Z[safe_iN], np.where(mS, 2.0*Z - Z[safe_iS], Z))
        zS = np.where(mS, Z[safe_iS], np.where(mN, 2.0*Z - Z[safe_iN], Z))

        # Extract Corner Coordinate Arrays
        xNE = np.where(mNE, X[safe_iNE], xN + xE - X)
        xNW = np.where(mNW, X[safe_iNW], xN + xW - X)
        xSE = np.where(mSE, X[safe_iSE], xS + xE - X)
        xSW = np.where(mSW, X[safe_iSW], xS + xW - X)

        zNE = np.where(mNE, Z[safe_iNE], zN + zE - Z)
        zNW = np.where(mNW, Z[safe_iNW], zN + zW - Z)
        zSE = np.where(mSE, Z[safe_iSE], zS + zE - Z)
        zSW = np.where(mSW, Z[safe_iSW], zS + zW - Z)

        # Compute Derivatives & Metrics
        X_xi  = 0.5 * (xE - xW)
        X_eta = 0.5 * (xN - xS)
        Z_xi  = 0.5 * (zE - zW)
        Z_eta = 0.5 * (zN - zS)
        
        X_xita = 0.25 * (xNE - xNW - xSE + xSW)
        Z_xita = 0.25 * (zNE - zNW - zSE + zSW)
        
        alpha = X_eta**2 + Z_eta**2
        beta  = X_xi * X_eta + Z_xi * Z_eta
        gamma = X_xi**2 + Z_xi**2

        # Compute Sizing Differentials
        DC = h
        DE = np.where(movable_E, h[safe_iE], DC)
        DW = np.where(movable_W, h[safe_iW], DC)
        DN = np.where(movable_N, h[safe_iN], DC)
        DS = np.where(movable_S, h[safe_iS], DC)

        D_phi = 0.5 * (DE - DW)
        D_psi = 0.5 * (DN - DS)

        # Source Terms
        J = X_xi * Z_eta - X_eta * Z_xi
        j_dc = J / (DC + 1e-12)

        Source_X = j_dc * (D_phi * Z_eta - D_psi * Z_xi)
        Source_Z = j_dc * (D_psi * X_xi - D_phi * X_eta) 

        # Calculate Targets
        Denominator = 2.0 * (alpha + gamma)
        valid_denom_mask = Denominator >= 1e-12
        Denom_safe = np.where(valid_denom_mask, Denominator, 1.0) # Prevent DivZero

        # Raw Targets
        X_target = np.where(
            valid_denom_mask & (move_X_mask | move_ellipse_mask),
            (alpha * (xE + xW) + gamma * (xN + xS) - 2.0 * beta * X_xita - Source_X) / Denom_safe,
            X
        )
        Z_target = np.where(
            valid_denom_mask & (move_Z_mask | move_ellipse_mask),
            (alpha * (zE + zW) + gamma * (zN + zS) - 2.0 * beta * Z_xita - Source_Z) / Denom_safe,
            Z
        )

        # Apply Elliptical Constraints
        dx = X_target - xc_e
        dz = Z_target - zc_e
        
        zero_dist_mask = (np.abs(dx) < 1e-12) & (np.abs(dz) < 1e-12)
        dist_term = np.abs(dx / a_e)**n_e + np.abs(dz / b_e)**n_e
        dist_safe = np.where(dist_term < 1e-12, 1.0, dist_term) # Prevent DivZero
        
        scale = dist_safe ** (-1.0 / n_e)
        X_proj = xc_e + scale * dx
        Z_proj = zc_e + scale * dz

        X_ellipse_new = np.where(zero_dist_mask, X, (1.0 - omega) * X + omega * X_proj)
        Z_ellipse_new = np.where(zero_dist_mask, Z, (1.0 - omega) * Z + omega * Z_proj)

        # Apply Relaxation
        X_reg_new = np.where(move_X_mask, (1.0 - omega) * X + omega * X_target, X)
        Z_reg_new = np.where(move_Z_mask, (1.0 - omega) * Z + omega * Z_target, Z)

        # Merge and Update Coordinates
        X = np.where(move_ellipse_mask, X_ellipse_new, X_reg_new)
        Z = np.where(move_ellipse_mask, Z_ellipse_new, Z_reg_new)

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
    move_ellipse: set = None,        
    ellipse_params: tuple = None,    
    iterations: int = 1500,
    omega: float = 0.05
) -> np.ndarray:
    """Apply Winslow mesh smoothing using vectorized NumPy operations.

    An alternative to the Numba implementation that utilizes pure NumPy
    vectorization for the iterative stencil update steps.

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
    move_ellipse : set, optional
        Set of node indices constrained to move along an elliptical boundary.
    ellipse_params : tuple, optional
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
    
    if move_ellipse is None:
        move_ellipse = set()
        
    if ellipse_params is None:
        e_params = np.array([1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.float64)
    else:
        e_params = np.array(ellipse_params, dtype=np.float64)

    num_nodes = len(points)
    
    node_to_neighbors = {i: set() for i in range(num_nodes)}
    for quad in quads:
        for i in range(4):
            n1 = quad[i]
            n2 = quad[(i+1)%4]
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
            diffs = np.append(diffs, sorted_angles[0] - sorted_angles[-1] + 2*np.pi)
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
            if nA is None or nB is None: return None
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

    # Boolean Masks 
    movable_nodes = move_all | move_X_only | move_Z_only | move_ellipse
    is_movable = np.zeros(num_nodes, dtype=np.bool_)
    is_movable[list(movable_nodes)] = True
    
    move_X_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_Z_mask = np.zeros(num_nodes, dtype=np.bool_)
    move_ellipse_mask = np.zeros(num_nodes, dtype=np.bool_)
    
    move_X_mask[list(move_all | move_X_only)] = True
    move_Z_mask[list(move_all | move_Z_only)] = True
    move_ellipse_mask[list(move_ellipse)] = True

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
            stencil_arr, is_movable, move_X_mask, move_Z_mask, move_ellipse_mask, e_params, omega, step_iters
        )
        
        current_iter += step_iters
        percent = (current_iter / iterations) * 100
        print(f"Iteration: {current_iter} / {iterations} [{percent:.1f}%]   ", end='\r')

    print("\nSmoothing Complete.")
    return np.column_stack((X, Z))


# Functions for water block alignment
def get_surface_entities_by_physical_name(name):
    """Retrieve Gmsh surface entity tags belonging to a specified physical group.

    Parameters
    ----------
    name : str
        The physical name of the target group in the Gmsh model.

    Returns
    -------
    list
        A list of integer tags representing the geometric surface entities
        associated with the physical group name.
    """
    entities = []
    for dim, pg_tag in gmsh.model.getPhysicalGroups(dim=2):
        if gmsh.model.getPhysicalName(dim, pg_tag) == name:
            entities.extend(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
    return entities

def get_nodes_on_surface_entities(tag_to_index, surface_entities):
    """Find node indices belonging to specific geometric surfaces.

    Parameters
    ----------
    tag_to_index : dict
        Mapping from Gmsh node tags to local internal node indices.
    surface_entities : list
        List of Gmsh geometric surface entity tags.

    Returns
    -------
    set
        A set of local node indices located on the provided surfaces.
    """
    nodes = set()
    for surf in surface_entities:
        n_tags, _, _ = gmsh.model.mesh.getNodes(2, surf, includeBoundary=True)
        for t in n_tags:
            if t in tag_to_index:
                nodes.add(tag_to_index[t])
    return nodes

def get_water_interface_node_indices(tag_to_index, water_surface_entities, length_x, tol=1e-8):
    """Identify node indices lying on the water interface.

    Scans the boundaries of water region surface entities and extracts nodes
    that do not sit on the top, left, or right edges of the domain, thus
    isolating the bottom interface.

    Parameters
    ----------
    tag_to_index : dict
        Mapping from Gmsh node tags to local internal node indices.
    water_surface_entities : list
        List of surface tags representing the water layer.
    length_x : float
        Total length of the domain in the x-direction.
    tol : float, optional
        Geometric tolerance for boundary checks. Default is 1e-8.

    Returns
    -------
    set
        A set of local node indices defining the water interface.
    """
    interface_nodes = set()
    for surf in water_surface_entities:
        boundary = gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False)
        for dim, curve_tag in boundary:
            if dim != 1: continue
            n_tags, coords, _ = gmsh.model.mesh.getNodes(1, curve_tag, includeBoundary=True)
            if len(n_tags) == 0: continue
            xy = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]
            xs, zs = xy[:, 0], xy[:, 1]
            is_top = np.all(np.abs(zs - 0.0) < tol)
            is_left = np.all(np.abs(xs - 0.0) < tol)
            is_right = np.all(np.abs(xs - length_x) < tol)
            if not (is_top or is_left or is_right):
                for t in n_tags:
                    if t in tag_to_index:
                        interface_nodes.add(tag_to_index[t])
    return interface_nodes

def align_water_columns_to_interface_x(points_2d, water_surface_nodes, interface_nodes, quads):
    """Align water nodes vertically with the interface nodes.

    Adjusts the x-coordinates of nodes within the water column so that they
    align vertically with the varying water-bottom interface nodes,
    improving mesh orthogonality in the water layer.

    Parameters
    ----------
    points_2d : ndarray
        2D array of shape (N, 2) containing node coordinates [x, z].
    water_surface_nodes : set
        Set of node indices located inside the water layer.
    interface_nodes : set
        Set of node indices defining the water bottom interface.
    quads : ndarray
        2D array containing quadrilateral element connectivity.

    Returns
    -------
    tuple
        A tuple containing:
        - ndarray: Updated node coordinates.
        - int: Number of nodes successfully snapped/aligned.
        - int: Number of vertical columns processed.
    """
    if not interface_nodes: return points_2d, 0, 0
    edge_to_quads = {}
    for q in quads:
        if not all(n in water_surface_nodes for n in q): continue
        pair_A = [frozenset((q[0], q[1])), frozenset((q[2], q[3]))]
        pair_B = [frozenset((q[1], q[2])), frozenset((q[3], q[0]))]
        q_tuple = tuple(q)
        for edge in pair_A + pair_B:
            if edge not in edge_to_quads: edge_to_quads[edge] = []
            edge_to_quads[edge].append(q_tuple)

    horizontal_queue = []
    for e in edge_to_quads.keys():
        u, v = list(e)
        if abs(points_2d[u, 1] - 0.0) < 1e-3 and abs(points_2d[v, 1] - 0.0) < 1e-3:
            horizontal_queue.append(e)

    horizontal_edges = set(horizontal_queue)
    vertical_edges = set()
    visited_quads = set()

    while horizontal_queue:
        curr_e = horizontal_queue.pop(0)
        if curr_e not in edge_to_quads: continue
        for q in edge_to_quads[curr_e]:
            if q in visited_quads: continue
            visited_quads.add(q)
            pair_A = [frozenset((q[0], q[1])), frozenset((q[2], q[3]))]
            pair_B = [frozenset((q[1], q[2])), frozenset((q[3], q[0]))]
            if curr_e in pair_A:
                opp = pair_A[1] if curr_e == pair_A[0] else pair_A[0]
                vert1, vert2 = pair_B[0], pair_B[1]
            elif curr_e in pair_B:
                opp = pair_B[1] if curr_e == pair_B[0] else pair_B[0]
                vert1, vert2 = pair_A[0], pair_A[1]
            else: continue
            
            if opp not in horizontal_edges:
                horizontal_edges.add(opp)
                horizontal_queue.append(opp)
            vertical_edges.add(vert1)
            vertical_edges.add(vert2)

    vertical_adj = {i: set() for i in water_surface_nodes}
    for e in vertical_edges:
        u, v = list(e)
        if u in vertical_adj and v in vertical_adj:
            vertical_adj[u].add(v)
            vertical_adj[v].add(u)

    visited = set()
    columns = []
    for i in water_surface_nodes:
        if i not in visited:
            col = []
            stack = [i]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    col.append(curr)
                    stack.extend(list(vertical_adj[curr]))
            if len(col) > 1: columns.append(col)

    snapped = 0
    n_cols = 0
    for col in columns:
        spline_x = None
        for idx in col:
            if idx in interface_nodes:
                spline_x = points_2d[idx, 0]
                break
        if spline_x is not None:
            n_cols += 1
            for idx in col:
                if points_2d[idx, 0] != spline_x:
                    points_2d[idx, 0] = spline_x
                    snapped += 1
    return points_2d, snapped, n_cols