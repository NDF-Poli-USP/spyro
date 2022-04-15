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
    x0 = pos[receiver_in_source_index,1]
    x1 = pos[receiver_comparison_index,1]
    measured_velocity = np.abs(x1-x0)/(time1-time0)
    minimum_velocity = 1.5
    error_percent = 100*np.abs(measured_velocity-minimum_velocity)/minimum_velocity
    print(f"Velocity error of {error_percent}%.", flush = True)
    return error_percent

def get_receiver_in_source_location(source_id, model):
    receiver_locations = model["acquisition"]["receiver_locations"]
    source_locations = model["acquisition"]["source_pos"]
    source_x = source_locations[source_id,1]

    cont = 0
    for receiver_location in receiver_locations:
        if math.isclose(source_x, receiver_location[1]):
            return cont
        cont += 1
    return ValueError("Couldn't find a receiver whose location coincides with a source within the standard tolerance.")



@pytest.mark.mpi(min_size=15)
def test_forward_3d():
    line1 = spyro.create_transect((0.25, 0.25), (0.25, 7.25), 4)
    line2 = spyro.create_transect((2.0, 0.25), (2.0, 7.25), 4)
    line3 = spyro.create_transect((3.75, 0.25), (3.75, 7.25), 4)
    line4 = spyro.create_transect((5.5, 0.25), (5.25, 7.25), 4)
    line5 = spyro.create_transect((7.25, 0.25), (7.25, 7.25), 4)
    lines = np.concatenate((line1, line2, line3, line4, line5))

    sources = spyro.insert_fixed_value(lines, -0.10, 0)

    receivers = spyro.create_2d_grid(0.25, 7.25, 0.25, 7.25, 30)
    receivers = spyro.insert_fixed_value(receivers, -0.15, 0)

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
        "source_pos": sources,
        "frequency": 5.0,
        "delay": 1.0,
        "receiver_locations": receivers,
    }
    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": 2.00,  # Final time for event
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

