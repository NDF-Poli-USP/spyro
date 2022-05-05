from firedrake import File
import matplotlib.pyplot as plt
import numpy as np
import math
import spyro
import pytest

def compare_velocity(p_r, receiver_in_source_index, receiver_comparison_index, model,dt):
    receiver_0 = p_r[:,receiver_in_source_index]
    receiver_1 = p_r[:,receiver_comparison_index]

    pos = model["acquisition"]["receiver_locations"]

    time0 = np.argmax(receiver_0)*dt
    time1 = np.argmax(receiver_1)*dt

    x0 = pos[receiver_in_source_index][1]
    x1 = pos[receiver_comparison_index][1]

    measured_velocity = np.abs(x1-x0)/(time1-time0)
    minimum_velocity = 1.5

    error_percent = 100*np.abs(measured_velocity-minimum_velocity)/minimum_velocity
    return error_percent


def test_forward_3d(tf = 0.6):
    model = {}

    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadrature": "KMV",  # Equi or KMV
        "degree": 3,  # p order
        "dimension": 3,  # dimension
    }
    model["parallelism"] = {"type": "automatic"}  # automatic",
    model["mesh"] = {
        "Lz": 5.175,  # depth in km - always positive
        "Lx": 7.50,  # width in km - always positive
        "Ly": 7.50,  # thickness in km - always positive
        "meshfile": "meshes/overthrust_3D_true_model.msh",
        "initmodel": "velocity_models/overthrust_3D_guess_model.hdf5",
        "truemodel": "velocity_models/overthrust_3D_true_model.hdf5",
    }
    model["BCs"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
        "exponent": 2,  # damping layer has a exponent variation
        "cmax": 6.0,  # maximum acoustic wave velocity in PML - km/s
        "R": 1e-6,  # theoretical reflection coefficient
        "lz": 0.75,  # thickness of the PML in the z-direction (km) - always positive
        "lx": 0.75,  # thickness of the PML in the x-direction (km) - always positive
        "ly": 0.75,  # thickness of the PML in the y-direction (km) - always positive
    }
    model["acquisition"] = {
        "source_type": "Ricker",
        "source_pos": [(-0.15, 0.25, 0.25)],
        "frequency": 5.0,
        "delay": 1.0,
        "receiver_locations": [(-0.15, 0.25, 0.25), (-0.15, 0.3, 0.25), (-0.15, 0.35, 0.25), (-0.15, 0.4, 0.25), (-0.15, 0.45, 0.25), (-0.15, 0.5, 0.25), (-0.15, 0.55, 0.25), (-0.15, 0.6, 0.25)],
    }
    model["aut_dif"] ={
        "status": False
    }
    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": tf,  # Final time for event
        "dt": 0.00075,
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool": 100,  # how frequently to output solution to pvds
        "fspool": 99999,  # how frequently to save solution to RAM
    }

    comm = spyro.utils.mpi_init(model)
    mesh, V = spyro.io.read_mesh(model, comm)
    vp = spyro.io.interpolate(model, mesh, V, guess=False)

    if comm.ensemble_comm.rank == 0:
        File("true_velocity.pvd", comm=comm.comm).write(vp)

    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(
        dt=model["timeaxis"]["dt"],
        tf=model["timeaxis"]["tf"],
        freq=model["acquisition"]["frequency"],
    )

    p, p_r = spyro.solvers.forward(
        model, mesh, comm, vp, sources, wavelet, receivers, output=False
    )

    dt=model["timeaxis"]["dt"]
    final_time=model["timeaxis"]["tf"]

    pass_error_test = True

    if comm.comm.rank == 0:
        error_percent = compare_velocity(p_r, 0, 7, model,dt)
        if error_percent < 5:
            pass_error_test = True
        else:
            pass_error_test = False
    
    assert pass_error_test