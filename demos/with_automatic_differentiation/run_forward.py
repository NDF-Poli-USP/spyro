from firedrake import *
import spyro
import matplotlib.pyplot as plt
import numpy as np

import spyro.solvers

# --- Basid setup to run a forward simulation with AD --- #
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5,  # regularization parameter
}

model["parallelism"] = {
    # options:
    # `shots_parallelism`. Shots parallelism.
    # None - no shots parallelism.
    "type": "shots_parallelism",
    "num_spacial_cores": 1,  # Number of cores to use in the spatial parallelism.
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC
    "outer_bc": "non-reflective",  # none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "source_pos": spyro.create_transect((0.2, 0.15), (0.8, 0.15), 3),
    "frequency": 7.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((0.2, 0.2), (0.8, 0.2), 10),
}
model["aut_dif"] = {
    "status": True,
}

model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 0.8,  # Final time for event (for test 7)
    "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}


# Use emsemble parallelism.
M = model["parallelism"]["num_spacial_cores"]
my_ensemble = Ensemble(COMM_WORLD, M)
mesh = UnitSquareMesh(50, 50, comm=my_ensemble.comm)
element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)
V = FunctionSpace(mesh, element)


def make_vp_circle(vp_guess=False, plot_vp=False):
    """Acoustic velocity model"""
    x, z = SpatialCoordinate(mesh)
    if vp_guess:
        vp = Function(V).interpolate(1.5 + 0.0 * x)
    else:
        vp = Function(V).interpolate(
            2.5
            + 1 * tanh(100 * (0.125 - sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2)))
        )
    if plot_vp:
        outfile = File("acoustic_cp.pvd")
        outfile.write(vp)
    return vp


forward_solver = spyro.solvers.forward_ad.ForwardSolver(
    model, mesh
)

c_true = make_vp_circle()
# Ricker wavelet
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)

if model["parallelism"]["type"] is None:
    for sn in range(len(model["acquisition"]["source_pos"])):
        rec_data, _ = forward_solver.execute(c_true, sn, wavelet)
        spyro.plots.plot_shots(
            model, my_ensemble.comm, rec_data, vmax=1e-08, vmin=-1e-08)
else:
    # source_number based on the ensemble.ensemble_comm.rank
    source_number = my_ensemble.ensemble_comm.rank
    rec_data, _ = forward_solver.execute(
        c_true, source_number, wavelet)
    spyro.plots.plot_shots(
        model, my_ensemble.comm, rec_data, vmax=1e-08, vmin=-1e-08)
