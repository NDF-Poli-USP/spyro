from firedrake import (
    BoxMesh,
    FunctionSpace,
    Function,
)
import time
import spyro

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 3,  # dimension
}

model["parallelism"] = {
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

model["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 1.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["PML"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 1.0,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 1.0,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 1.0,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 1.0,  # thickness of the PML in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.1, 1.0, 1.0)],
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 300,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1, 1.9), (-0.10, 0.1, 1.9), 300
    ),
}

# Simulate for 1.0 seconds.
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.0,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}

# Note: nz, nx, and ny are obtained by dividing the dimensions 3 x 4 x 4 km
# by your desired element size.
mesh = BoxMesh(60, 80, 80, 3.0, 4.0, 4.0)
# If using a PML, then the z coordinates needs to be negative,
# and on the -x and -y sides of the domain, the PML region must have negative coordinates.
mesh.coordinates.dat.data[:, 0] -= 3.0
mesh.coordinates.dat.data[:, 1] -= 1.0
mesh.coordinates.dat.data[:, 2] -= 1.0

comm = spyro.utils.mpi_init(model)

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)
V = FunctionSpace(mesh, element)

if comm.comm.rank == 0:
    print("There are " + str(V.dim()) + " degrees of freedom", flush=True)

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

vp = Function(V, name="velocity").assign(1.5)

t1 = time.time()
p_field, p_at_recv = spyro.solvers.Leapfrog(
    model,
    mesh,
    comm,
    vp,
    sources,
    receivers,
    output=False,
)
if comm.comm.rank == 0:
    spyro.plots.plot_shotrecords(
        model, p_at_recv, name="testing", vmin=-1e-5, vmax=1e-5
    )

# Record the slowest time.
print(time.time() - t1, flush=True)
