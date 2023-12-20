from firedrake import *
import spyro
import matplotlib.pyplot as plt
import numpy as np

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
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
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
    "frequency": 10.0,
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

comm = spyro.utils.mpi_init(model)
mesh = UnitSquareMesh(100, 100)  # to test FWI, mesh aligned with interface

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

wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)

vp_exact = make_vp_circle(plot_vp=True)
if comm.comm.rank == 0:
    for sn in range(len(model["acquisition"]["source_pos"])):
        receiver_data = spyro.solvers.forward_AD(model, mesh, comm, vp_exact,
                                                 wavelet, debug=True,
                                                 source_number=sn)

        # --- Plot the receiver data --- #
        data = []
        for _, rec in enumerate(receiver_data):
            data.append(rec.dat.data_ro[:])

        spyro.plots.plot_shots(model, comm, data, vmax=1e-08, vmin=-1e-08)
else:
    raise NotImplementedError("Only implemented for 1 processor")
