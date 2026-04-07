import os
from scipy.ndimage import gaussian_filter
import segyio
import numpy as np
import matplotlib.pyplot as plt

try:
    from SeismicMesh import write_velocity_model

    HAS_SEISMICMESH = True
except ImportError:
    HAS_SEISMICMESH = False


def smooth_velocity_field_file(
    input_filename,
    output_filename,
    sigma,
    save_fig=False,
    show=False,
    write_hdf5=True,
    i_limit=None,
    vp_limit=None,
    tol=1e-5,
):
    """Smooth a velocity field from a SEG-Y file using a Gaussian filter.

    Reads a velocity model from a SEG-Y file, applies a Gaussian smoothing
    filter, optionally preserves shallow-layer values and water-column
    velocities, writes the result back as a SEG-Y file, and optionally exports
    an HDF5 file via SeismicMesh.

    Parameters
    ----------
    input_filename : str
        Path to the input SEG-Y velocity file.
    output_filename : str
        Path for the output (smoothed) SEG-Y velocity file.
    sigma : float
        Standard deviation (in grid cells) for the Gaussian filter.  Larger
        values produce stronger smoothing.
    save_fig : bool, optional
        If ``True``, save a PNG image of the smoothed model next to
        *output_filename*. Default is ``False``.
    show : bool, optional
        If ``True``, display the smoothed model interactively with
        ``matplotlib``. Default is ``False``.
    write_hdf5 : bool, optional
        If ``True`` and SeismicMesh is available, write an HDF5 version of
        the smoothed model using
        :func:`SeismicMesh.write_velocity_model`. Default is ``True``.
    i_limit : int or None, optional
        Row index below which smoothing is **not** applied; rows
        ``0 … i_limit-1`` retain their original values.  ``None`` disables
        this protection (equivalent to ``i_limit=0``). Default is ``None``.
    vp_limit : float or None, optional
        Velocity threshold used to identify water/air cells.  Any cell whose
        original velocity is at or below ``vp_limit + tol`` is reset to the
        global minimum velocity after smoothing.  ``None`` uses the global
        minimum (i.e., no cells are reset). Default is ``None``.
    tol : float, optional
        Tolerance added to *vp_limit* when testing whether a cell belongs to
        the water/air layer. Default is ``1e-5``.

    Returns
    -------
    None
        The function writes its results to *output_filename* (and optionally
        to an HDF5 file and a PNG image) as side effects.

    Raises
    ------
    ValueError
        If *input_filename* does not have a ``.segy`` extension.

    Notes
    -----
    SEG-Y traces are assumed to be stored column-major (each trace corresponds
    to one *x*-column of the 2-D velocity grid).

    Examples
    --------
    Basic smoothing with a standard deviation of 5 grid cells:

    >>> smooth_velocity_field_file(
    ...     "marmousi.segy",
    ...     "marmousi_smooth.segy",
    ...     sigma=5,
    ... )
    """
    f, filetype = os.path.splitext(input_filename)

    if filetype == ".segy":
        with segyio.open(input_filename, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                vp[:, index] = trace
    else:
        raise ValueError("Not yet implemented!")

    vp_min = np.min(vp)
    vp_max = np.max(vp)
    print(f"Velocity model has minimum vp of {vp_min}, and max of {vp_max}")

    vp_smooth = gaussian_filter(vp, sigma)
    ni, nj = np.shape(vp)
    if i_limit is None:
        i_limit = 0
    if vp_limit is None:
        vp_limit = vp_min

    for i in range(ni):
        for j in range(nj):
            if i < i_limit:
                vp_smooth[i, j] = vp[i, j]
            if vp[i, j] <= vp_limit + tol:
                vp_smooth[i, j] = vp_min

    spec = segyio.spec()
    spec.sorting = 2  # not sure what this means
    spec.format = 1  # not sure what this means
    spec.samples = range(vp_smooth.shape[0])
    spec.ilines = range(vp_smooth.shape[1])
    spec.xlines = range(vp_smooth.shape[0])

    assert np.sum(np.isnan(vp_smooth[:])) == 0

    with segyio.create(output_filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = vp_smooth[:, tr]

    if save_fig is True or show is True:
        with segyio.open(output_filename, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            show_vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                show_vp[:, index] = trace

        fig, ax = plt.subplots()
        plt.pcolormesh(show_vp, shading="auto")
        plt.title("Guess model")
        plt.colorbar(label="P-wave velocity (km/s)")
        plt.xlabel("x-direction (m)")
        plt.ylabel("z-direction (m)")
        ax.axis("equal")
        if save_fig:
            plt.savefig(output_filename + ".png")
        if show:
            plt.show()

    if write_hdf5:
        if HAS_SEISMICMESH:
            write_velocity_model(output_filename, ofname=output_filename[:-5])
        else:
            print("Warning: SeismicMesh not available, skipping HDF5 writing.")

    return None
