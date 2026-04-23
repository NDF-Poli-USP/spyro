import numpy as np
import segyio
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

try:
    import gmsh
except ImportError:
    gmsh = None


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
        cell_size = np.maximum(cell_size, hmin)

    # Applying padding
    if pad_type == "rectangular" or pad_type == "hyperelliptical":
        dz = (bbox[1] - bbox[0]) / n_traces
        dx = (bbox[3] - bbox[2]) / n_samples
        nnz = int(pad_size_z / dz)
        nnx = int(pad_size_x / dx)
        print(nnx, nnz, n_samples, n_traces)
        print(pad_size_z, bbox[0], bbox[2])
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
        window_length_z = int((1.0 - grade) * 0.1 * cell_size.shape[0])
        window_length_x = int((1.0 - grade) * 0.1 * cell_size.shape[1])
        cell_size = apply_savitzky_golay_filter_2d(cell_size, window_length_x, window_length_z)

    print("Function Minimum and Maximum values:")
    print(cell_size.min(), cell_size.max())

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

    return sizing_function, cell_size.min(), cell_size.max(), n_samples, n_traces


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
    return vp, n_traces, n_samples


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
    print(window_length_z, window_length_x)
    # First filter along axis 0 (columns), then along axis 1 (rows)
    filtered_values = savgol_filter(grid_values, window_length_x, polyorder, axis=0)
    filtered_values = savgol_filter(filtered_values, window_length_z, polyorder, axis=1)

    return filtered_values


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

    x_low = np.clip(x_low, 0.0, Lx)
    x_high = np.clip(x_high, 0.0, Lx)
    z_top = np.clip(z_top, -Lz, 0.0)
    z_bot = np.clip(z_bot, -Lz, 0.0)

    # Map meters → nearest indices
    ix_min = int(np.rint(x_low / _dx))
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

        in_water = np.abs(block - target) <= tol
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


# Functions for water block alignment
@check_gmsh
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


@check_gmsh
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


@check_gmsh
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
            if dim != 1:
                continue
            n_tags, coords, _ = gmsh.model.mesh.getNodes(1, curve_tag, includeBoundary=True)
            if len(n_tags) == 0:
                continue
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
    if not interface_nodes:
        return points_2d, 0, 0
    edge_to_quads = {}
    for q in quads:
        if not all(n in water_surface_nodes for n in q):
            continue
        pair_A = [frozenset((q[0], q[1])), frozenset((q[2], q[3]))]
        pair_B = [frozenset((q[1], q[2])), frozenset((q[3], q[0]))]
        q_tuple = tuple(q)
        for edge in pair_A + pair_B:
            if edge not in edge_to_quads:
                edge_to_quads[edge] = []
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
        if curr_e not in edge_to_quads:
            continue
        for q in edge_to_quads[curr_e]:
            if q in visited_quads:
                continue
            visited_quads.add(q)
            pair_A = [frozenset((q[0], q[1])), frozenset((q[2], q[3]))]
            pair_B = [frozenset((q[1], q[2])), frozenset((q[3], q[0]))]
            if curr_e in pair_A:
                opp = pair_A[1] if curr_e == pair_A[0] else pair_A[0]
                vert1, vert2 = pair_B[0], pair_B[1]
            elif curr_e in pair_B:
                opp = pair_B[1] if curr_e == pair_B[0] else pair_B[0]
                vert1, vert2 = pair_A[0], pair_A[1]
            else:
                continue

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
            if len(col) > 1:
                columns.append(col)

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


def check_gmsh(func):
    """Decorator for gmsh check.

    If gmsh isn't available returns ImportError
    """
    def wrapper(*args, **kwargs):
        if gmsh is None:
            raise ImportError("Please install gmsh to use this function.")
        else:
            return func(*args, **kwargs)
    
    return wrapper
