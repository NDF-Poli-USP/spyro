import os
from scipy.ndimage import gaussian_filter
import segyio
import numpy as np
import matplotlib.pyplot as plt
from SeismicMesh import write_velocity_model


def smooth_velocity_field_file(input_filename, output_filename, sigma, show=False, write_hdf5=True, i_limit=None, vp_limit=None, tol=1e-5):
    """Smooths a velocity field using a Gaussian filter.

    Parameters
    ----------
    input_filename : string
        The name of the input file.
    output_filename : string
        The name of the output file.
    sigma : float
        The standard deviation of the Gaussian filter.
    show : boolean, optional
        Should the plot image appear on screen

    Returns
    -------
    None

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

    if show is True:
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
        plt.savefig(output_filename+".png")
        plt.show()

    if write_hdf5:
        write_velocity_model(output_filename, ofname=output_filename[:-5])

    return None
