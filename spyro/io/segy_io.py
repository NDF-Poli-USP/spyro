import numpy as np
import segyio
from io import BytesIO
from ..io import write_function_to_grid
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
    velocity_grid_data = write_function_to_grid(function, V, grid_spacing, buffer=True)

    return create_segy_from_grid(velocity_grid_data, filename)
