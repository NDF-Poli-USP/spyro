import os
from scipy.ndimage import gaussian_filter
import segyio
import numpy as np
import matplotlib.pyplot as plt
from SeismicMesh import write_velocity_model
import sys

def smooth_field(input_filename, output_filename, show = False, sigma = 100):
    f, filetype = os.path.splitext(input_filename)

    if filetype == ".segy":
        with segyio.open(input_filename, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                vp[:, index] = trace

    vp_smooth = gaussian_filter(vp, sigma)
    ni, nj = np.shape(vp)

    for i in range(ni):
        for j in range(nj):
            if vp[i,j] < 1.51 and i < 400:
                vp_smooth[i,j] = vp[i,j]

    spec = segyio.spec()
    spec.sorting = 2 # not sure what this means
    spec.format = 1 # not sure what this means
    spec.samples = range(vp_smooth.shape[0])
    spec.ilines = range(vp_smooth.shape[1])
    spec.xlines = range(vp_smooth.shape[0])

    assert np.sum(np.isnan(vp_smooth[:])) == 0

    with segyio.create(output_filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = vp_smooth[:, tr]

    if show == True:
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
        plt.show()

    return None


# try 100:
sigma = int(sys.argv[1])
input = "velocity_models/vp_marmousi-ii.segy"
output = "velocity_models/vp_marmousi-ii_sigma"+str(sigma)+".segy"

smooth_field(input, output, show = True, sigma =sigma)

vp_filename, vp_filetype = os.path.splitext(output)

write_velocity_model(output, ofname = vp_filename)