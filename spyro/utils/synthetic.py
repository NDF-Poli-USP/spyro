import os
from scipy.ndimage import gaussian_filter
import segyio
import numpy as np
import matplotlib.pyplot as plt
import firedrake as fire
import copy
import spyro

def smooth_field(input_filename, output_filename, show = False):
    f, filetype = os.path.splitext(input_filename)

    if filetype == ".segy":
        with segyio.open(input_filename, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                vp[:, index] = trace

    sigma = 100
    vp_smooth = gaussian_filter(vp, sigma)

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

def create_shot_record(old_model, comm, show = False):
    model = copy.deepcopy(old_model)
    if model["mesh"]["truemodel"] == None:
        raise ValueError('Please insert a true model for shot record creation.')
    model["mesh"]["initmodel"] = model["mesh"]["truemodel"]

    print('Entering mesh generation', flush = True)
    M = cells_per_wavelength(model)
    mesh = build_mesh(model, vp = 'default', comm)
    element = domains.space.FE_method(mesh, method, degree)
    V = fire.FunctionSpace(mesh, element)
    vp = spyro.io.interpolate(model, mesh, V, guess=False)
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(
        dt=model["timeaxis"]["dt"],
        tf=model["timeaxis"]["tf"],
        freq=model["acquisition"]["frequency"],
    )
    p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
    spyro.io.save_shots(model, comm, p_r)
    if show == True:
        spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)

    return shot_record

