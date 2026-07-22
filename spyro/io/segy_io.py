import numpy as np
import segyio
from io import BytesIO
import matplotlib.pyplot as plt


def segy_to_png(segy_filename, output_file='debug.png', cmap='seismic', vmin=None, vmax=None, dpi=150, flip=True, show=False):
    """Read a SEGY file and return a PNG image (bytes) or save to disk.

    Parameters
    ----------
    segy_filename : str
        Path to the input SEGY file.
    output_file : str or None
        If provided, the PNG will be written to this path and the path
        returned. If None, the PNG bytes are returned.
    cmap : str
        Matplotlib colormap to use.
    vmin, vmax : float or None
        Color scale limits passed to imshow.
    dpi : int
        DPI for saving the PNG.
    flip : bool
        If True, flip the data vertically for common display orientation.
    show : bool
        If True shows the image

    Returns
    -------
    bytes or str
        PNG bytes if `output_file` is None, otherwise the output path.
    """
    # read segy file
    with segyio.open(segy_filename, "r", ignore_geometry=True) as seg:
        n_traces = seg.tracecount
        n_samples = len(seg.samples)
        data = np.zeros((n_samples, n_traces), dtype=float)
        for i in range(n_traces):
            data[:, i] = seg.trace[i]

    if flip:
        data = np.flipud(data)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_axis_off()

    if show:
        plt.show()

    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return output_file
    else:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


def create_segy_from_grid(velocity, filename, rotate=False):
    """
    Create a SEG-Y file from a velocity grid.

    Parameters
    ----------
    velocity : numpy.ndarray
        A 2D array representing the velocity grid. The shape of the array
        determines the number of samples and traces.
    filename : str
        The name of the output SEG-Y file.
    rotate : bool, optional
        If True, the velocity grid is rotated before being written to the file.
        Defaults to False.

    Raises
    ------
    AssertionError
        If the velocity grid contains NaN values.

    Notes
    -----
    - The `spec.format` parameter is set to 1, which corresponds to IBM float format.
    - The `spec.samples`, `spec.ilines`, and `spec.xlines` are configured based
      on the shape of the velocity grid.
    """
    spec = segyio.spec()

    if rotate is False:
        velocity = np.flipud(velocity.T)

        spec.sorting = 2  # not sure what this means
        spec.format = 1  # not sure what this means
        spec.samples = range(velocity.shape[0])
        spec.ilines = range(velocity.shape[1])
        spec.xlines = range(velocity.shape[0])

        assert np.sum(np.isnan(velocity[:])) == 0

        with segyio.create(filename, spec) as f:
            for tr, il in enumerate(spec.ilines):
                f.trace[tr] = velocity[:, tr]
    else:
        velocity = np.flipud(velocity)

        spec.sorting = 2  # not sure what this means
        spec.format = 1  # not sure what this means
        # velocity is (n_samples, n_traces) here, so samples are rows
        # and traces correspond to columns
        spec.samples = range(velocity.shape[0])
        spec.ilines = range(velocity.shape[1])
        spec.xlines = range(velocity.shape[1])

        assert np.sum(np.isnan(velocity[:])) == 0

        with segyio.create(filename, spec) as f:
            for tr, il in enumerate(spec.ilines):
                f.trace[tr] = velocity[:, tr]


def create_segy(function, V, grid_spacing, filename):
    """Write the velocity data into a segy file named filename

    Parameters
    ----------
    function : firedrake.Function
        Function to interpolate
    V : firedrake.FunctionSpace
        Function space of function
    grid_spacing : float
        Spacing of grid points
    filename: str
        Name of the segy file to save

    Returns
    -------
    None
    """
    from ..io import write_function_to_grid  # Here to avoid circular import
    velocity_grid_data = write_function_to_grid(function, V, grid_spacing, buffer=True)

    return create_segy_from_grid(velocity_grid_data, filename)


def read_segy_velocity_model(fname):
    """Read a velocity model from a SEG-Y file.

    Parameters
    ----------
    fname : str
        Filename of the SEG-Y velocity model.

    Returns
    -------
    vp : numpy.ndarray
        Velocity model array in ``(z, x)`` order.
    nz : int
        Number of samples per trace, corresponding to the z direction.
    nx : int
        Number of traces in the SEG-Y file, corresponding to the x direction.

    Raises
    ------
    ImportError
        If ``segyio`` is not installed.
    """
    with segyio.open(fname, "r", ignore_geometry=True) as segy:
        nx = len(segy.trace)
        nz = len(segy.samples)
        vp = np.zeros((nz, nx), dtype=np.float32)

        for i in range(nx):
            vp[:, i] = segy.trace[i]

    vp = np.flipud(vp)

    return vp, nz, nx


def create_grid_dictionary_from_segy(filename, length_z=None, length_x=None):
    """Read a SEG-Y file and return a grid velocity dictionary.

    Parameters
    ----------
    filename : str
        Path to the SEG-Y file.
    length_z : float
        Physical model length in the z direction.
    length_x : float
        Physical model length in the x direction.

    Returns
    -------
    dict
        Grid velocity data with ``vp_values`` in ``(z, x)`` order.

    Raises
    ------
    ValueError
        If either ``length_z`` or ``length_x`` is not provided.

    Notes
    -----
    The returned dictionary follows the structured-grid convention used by
    the rest of the I/O layer, including directional spacing metadata and a
    default ``abc_pad_length`` of ``0.0``.
    """
    if length_z is None or length_x is None:
        raise ValueError(
            "length_z and length_x are required to build a grid dictionary from SEG-Y."
        )

    vp_values, nz, nx = read_segy_velocity_model(filename)

    spacing_z = float(length_z) / float(nz - 1)
    spacing_x = float(length_x) / float(nx - 1)
    grid_spacing = spacing_z if np.isclose(spacing_z, spacing_x) else None

    grid_velocity_data = {
        "vp_values": vp_values,
        "grid_spacing": grid_spacing,
        "grid_spacing_z": spacing_z,
        "grid_spacing_x": spacing_x,
        "length_z": float(length_z),
        "length_x": float(length_x),
        "abc_pad_length": 0.0,
    }
    return grid_velocity_data
