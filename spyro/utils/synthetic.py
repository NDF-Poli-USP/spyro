import os
from scipy.ndimage import gaussian_filter
import segyio
import numpy as np
import matplotlib.pyplot as plt
import firedrake as fire
import copy
import spyro
from spyro.utils.mesh_utils import cells_per_wavelength, build_mesh
from spyro.domains.space import FE_method
from SeismicMesh import write_velocity_model


def smooth_field(input_filename, output_filename, show = False, sigma = 100):
    f, filetype = os.path.splitext(input_filename)

    if filetype == ".segy":
        with segyio.open(input_filename, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                vp[:, index] = trace

    vp_smooth = gaussian_filter(vp, sigma)
    ni, nj = np.shape(vp_smooth)
    for i in range(ni):
        for j in range(nj):
            if vp[i,j]==1.5:
                vp_smooth[i,j] = 1.5

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
    
    # Creating forward model inputs
    if model["inversion"]["true_model"] == None:
        raise ValueError('Please insert a true model for shot record creation.')
    model["inversion"]["initial_guess"] = model["inversion"]["true_model"]
    
    model["mesh"]["meshfile"] = 'meshes/temp_synthetic_truemodel_mesh.msh'
    print('Entering mesh generation', flush = True)
    M = cells_per_wavelength(model["opts"]['method'],model["opts"]['degree'],model["opts"]['dimension'])
    print("Generating true model mesh")
    mesh = build_mesh(model, comm, 'meshes/temp_synthetic_truemodel_mesh', model["inversion"]["initial_guess"])
    element = FE_method(mesh, model["opts"]['method'], model["opts"]['degree'])
    V = fire.FunctionSpace(mesh, element)

    
    vpfile = model["inversion"]["true_model"]
    vp_filename, vp_filetype = os.path.splitext(vpfile)

    if vp_filetype == '.segy':
        write_velocity_model(vpfile, ofname = vp_filename)
        new_vpfile = vp_filename+'.hdf5'
        model["inversion"]["true_model"] = new_vpfile

    
    vp = spyro.io.interpolate(model, mesh, V, guess=False)
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(
        dt=model["timeaxis"]["dt"],
        tf=model["timeaxis"]["tf"],
        freq=model["acquisition"]["frequency"],
    )
    print("Running true model shot record to save.")
    p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output = True)
    spyro.io.save_shots(model, comm, p_r)
    if show == True:
        spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)

    shot_record = spyro.io.load_shots(model, comm)

    return shot_record

