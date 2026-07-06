import numpy as np
import segyio
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

try:
    import gmsh
except ImportError:
    gmsh = None


try:
    import SeismicMesh
except ImportError:
    SeismicMesh = None


def check_seismicmesh(func):
    """Decorator for SeismicMesh check.

    If SeismicMesh isn't available raises ImportError
    """
    def wrapper(*args, **kwargs):
        if SeismicMesh is None:
            raise ImportError(
                "SeismicMesh is not available. Please "
                + "install it to use this function."
            )
        else:
            return func(*args, **kwargs)

    return wrapper


def check_gmsh(func):
    """Decorator for gmsh check.

    If gmsh isn't available raises ImportError
    """
    def wrapper(*args, **kwargs):
        if gmsh is None:
            raise ImportError("Please install gmsh to use this function.")
        else:
            return func(*args, **kwargs)

    return wrapper


def calculate_edge_length(cpw, minimum_velocity, frequency):
    """
    Calculate the edge length for mesh generation.

    Parameters
    ----------
    cpw : float
        Cells per wavelength.
    minimum_velocity : float
        Minimum velocity in the domain.
    frequency : float
        Source frequency.

    Returns
    -------
    edge_length : float
        Calculated edge length for mesh elements.
    """
    if cpw == 0.0 or cpw is None:
        raise ValueError("cpw value of {cpw} invalid for edge length calculation.")
    v_min = minimum_velocity

    lbda_min = v_min / frequency

    edge_length = lbda_min / cpw
    return edge_length


def vp_to_sizing(vp, cpw, frequency):
    """
    Convert velocity field to mesh sizing function.

    Parameters
    ----------
    vp : numpy.ndarray
        P-wave velocity field.
    cpw : float
        Cells per wavelength(must be positive).
    frequency : float
        Source frequency in Hz(must be positive).

    Returns
    -------
    sizing : numpy.ndarray
        Mesh element sizes corresponding to the velocity field.

    Raises
    ------
    ValueError
        If cpw or frequency is not positive.

    Notes
    -----
    The mesh size is calculated as: `size = vp / (frequency * cpw)`
    This ensures that the mesh has the specified number of cells per
    wavelength throughout the domain.
    """
    if cpw < 0.0 or cpw == 0.0:
        raise ValueError(f"Cells-per-wavelength value of {cpw} not supported.")
    if frequency < 0.0 or frequency == 0.0:
        raise ValueError(f"Frequency must be positive and non zero, not {frequency}")

    return vp / (frequency * cpw)


def create_sizing_function(
    fname,
    hmin=None,
    bbox=None,
    wl=3,
    freq=5,
    pad_type=None,
    pad_size_x=-1.0,
    pad_size_z=-1.0,
    grade=None,
    vp_water=None,
):
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
        - n_samples (int): Number of depth samples per trace (rows).
        - n_traces (int): Number of lateral traces (columns).
    """

    # Read velocity model with provided bbox
    vp, n_samples, n_traces = read_segy_velocity_model(fname)
    # Set water velocity if value = 0
    if vp_water is not None:
        vp = np.where(vp == 0, vp_water, vp)
    else:
        # If no water velocity provided and model contains zeros, raise error
        if np.any(vp == 0):
            raise ValueError(
                "Velocity model contains zero values (water). Provide 'vp_water' "
                "to substitute water velocities or preprocess the SEGY file."
            )
    # Calculate wavelength-based sizing
    cell_size = calculate_wavelength_sizing(vp, wl, freq)
    # Enforce minimum element size
    if hmin is not None:
        cell_size = np.maximum(cell_size, hmin)

    # Applying padding
    if pad_type == "rectangular" or pad_type == "hyperelliptical":
        if n_samples < 2 or n_traces < 2:
            raise ValueError(
                f"SEGY model must have at least 2 samples and 2 traces; "
                f"got n_samples={n_samples}, n_traces={n_traces}."
            )

        dz = (bbox[1] - bbox[0]) / (n_samples - 1)
        dx = (bbox[3] - bbox[2]) / (n_traces - 1)
        nnz = int(pad_size_z / dz)
        nnx = int(pad_size_x / dx)

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
    if grade is not None and grade > 0.0:
        polyorder = 3

        def calculate_window(dim_size, grade_factor, polyorder=3):
            """
            Return a valid odd Savitzky-Golay window length.

            grade_factor:
                0.1 -> small/local smoothing
                0.9 -> high smoothing

            Safety rule:
                The window may not exceed 90% of the corresponding axis size.
            """
            if not 0.0 < grade_factor <= 1.0:
                raise ValueError(
                    f"grade must be in the interval (0, 1], got {grade_factor}."
                )

            # Savitzky-Golay requires window_length > polyorder.
            min_valid = polyorder + 2 if polyorder % 2 != 0 else polyorder + 1

            if dim_size < min_valid:
                return None

            requested_window = int(round(grade_factor * dim_size))

            # Maximum window size
            max_window = int(np.floor(0.90 * dim_size))

            # Savitzky-Golay requires an odd window.
            if max_window % 2 == 0:
                max_window -= 1

            # For very small dimensions, do not violate the 90% safety rule.
            if max_window < min_valid:
                return None

            # Keep the window inside the valid range.
            window = max(min_valid, min(requested_window, max_window))

            # Force odd.
            if window % 2 == 0:
                window -= 1

            return window

        window_length_z = calculate_window(cell_size.shape[0], grade)
        window_length_x = calculate_window(cell_size.shape[1], grade)

        # Apply the filter
        cell_size = apply_savitzky_golay_filter_2d(
            cell_size,
            window_length_x=window_length_x,
            window_length_z=window_length_z,
            polyorder=polyorder
        )

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
    return vp, n_samples, n_traces


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


def apply_savitzky_golay_filter_2d(
    grid_values,
    window_length_x=501,
    window_length_z=501,
    polyorder=3,
):
    """Apply a two-dimensional Savitzky-Golay grading filter.

    Parameters
    ----------
    grid_values : array-like
        Two-dimensional array containing mesh element sizes.
    window_length_x : int, optional
        Odd Savitzky-Golay window length along the lateral x direction
    window_length_z : int, optional
        Odd Savitzky-Golay window length along the depth z direction
    polyorder : int, optional
        Polynomial order used by the Savitzky-Golay filter. Default is 3.

    Returns
    -------
    numpy.ndarray
        Filtered sizing field. If any negative value is created by the
        filter, all values below the original minimum positive element size
        are replaced by that minimum size.

    Raises
    ------
    ValueError
        If the input is invalid, contains NaN/Inf, has no positive values,
        or uses incompatible Savitzky-Golay window parameters.
    """
    grid_values = np.asarray(grid_values, dtype=np.float64)

    if grid_values.ndim != 2:
        raise ValueError(
            f"grid_values must be a 2D array, received shape "
            f"{grid_values.shape}."
        )

    if not np.all(np.isfinite(grid_values)):
        raise ValueError(
            "Sizing field contains NaN or Inf values before filtering."
        )

    if window_length_z is None or window_length_x is None:
        return grid_values.copy()

    if window_length_z <= 0 or window_length_x <= 0:
        raise ValueError(
            "Savitzky-Golay window lengths must be positive. "
            f"Got z={window_length_z}, x={window_length_x}."
        )

    if window_length_z % 2 == 0 or window_length_x % 2 == 0:
        raise ValueError(
            "Savitzky-Golay window lengths must be odd. "
            f"Got z={window_length_z}, x={window_length_x}."
        )

    if window_length_z > grid_values.shape[0]:
        raise ValueError(
            f"window_length_z={window_length_z} exceeds z-axis size "
            f"{grid_values.shape[0]}."
        )

    if window_length_x > grid_values.shape[1]:
        raise ValueError(
            f"window_length_x={window_length_x} exceeds x-axis size "
            f"{grid_values.shape[1]}."
        )

    if window_length_z <= polyorder or window_length_x <= polyorder:
        raise ValueError(
            "Savitzky-Golay window lengths must be greater than polyorder. "
            f"Got z={window_length_z}, x={window_length_x}, "
            f"polyorder={polyorder}."
        )

    positive_values = grid_values[grid_values > 0.0]

    if positive_values.size == 0:
        raise ValueError(
            "Sizing field contains no positive values, so the minimum "
            "element size cannot be determined."
        )

    min_element_size = float(np.min(positive_values))

    filtered_values = savgol_filter(
        grid_values,
        window_length_z,
        polyorder,
        axis=0,
        mode="nearest",
    )

    filtered_values = savgol_filter(
        filtered_values,
        window_length_x,
        polyorder,
        axis=1,
        mode="nearest",
    )

    if not np.all(np.isfinite(filtered_values)):
        raise ValueError(
            "Sizing field contains NaN or Inf values after filtering."
        )

    negative_mask = filtered_values < 0.0
    n_negative = int(np.count_nonzero(negative_mask))

    if n_negative > 0:
        below_min_mask = filtered_values < min_element_size
        n_replaced = int(np.count_nonzero(below_min_mask))

        print(
            f"Warning: Savitzky-Golay grading produced {n_negative} negative "
            f"sizing values. Replacing all {n_replaced} values below "
            f"min_element_size={min_element_size:.6f} with the minimum "
            "element size."
        )

        filtered_values[below_min_mask] = min_element_size

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


def intersect(x0, z0, dx, dz, xc, zc, a_val, b_val, hyper_n):
    def f(s):
        x, z = x0 + s * dx, z0 + s * dz
        return (abs(x - xc) / a_val)**hyper_n + (abs(z - zc) / b_val)**hyper_n - 1.0
    s_low, s_high = 0.0, 1.0
    while f(s_high) < 0:
        s_high *= 2.0
    for _ in range(100):
        s_mid = (s_low + s_high) / 2.0
        if f(s_mid) > 0:
            s_high = s_mid
        else:
            s_low = s_mid
    return x0 + s_mid * dx, z0 + s_mid * dz


def get_theta(x, z, xc, zc, a_val, b_val, hyper_n):
    vx, vz = (x - xc) / a_val, (z - zc) / b_val
    vx = vx if abs(vx) > 1e-12 else 0.0
    vz = vz if abs(vz) > 1e-12 else 0.0
    return np.arctan2(np.sign(vz) * np.abs(vz)**(hyper_n / 2.0), np.sign(vx) * np.abs(vx)**(hyper_n / 2.0))


def make_arc(p1_tag, p2_tag, x1, z1, x2, z2, xc, zc, a_val, b_val, hyper_n, num_pts=25):
    t1, t2 = get_theta(x1, z1, xc, zc, a_val, b_val, hyper_n), get_theta(x2, z2, xc, zc, a_val, b_val, hyper_n)
    if t2 - t1 > np.pi:
        t1 += 2 * np.pi
    elif t1 - t2 > np.pi:
        t2 += 2 * np.pi
    pts = [p1_tag]
    for t in np.linspace(t1, t2, num_pts)[1:-1]:
        cos_t, sin_t = np.cos(t), np.sin(t)
        x = xc + a_val * np.sign(cos_t) * np.abs(cos_t)**(2.0 / hyper_n)
        z = zc + b_val * np.sign(sin_t) * np.abs(sin_t)**(2.0 / hyper_n)
        pts.append(gmsh.model.occ.addPoint(x, z, 0.0))
    pts.append(p2_tag)
    return gmsh.model.occ.addSpline(pts)
