import time
import pytest


from firedrake import *

import Spyro

model = {}

model["opts"] = {
    "method": "KMV",
    "variant": None,
    "type": "SIP",  # for DG only - SIP, NIP and IIP
    "degree": 2,  # p order
    "dimension": 2,  # dimension
    "mesh_size": 0.005,  # h
    "beta": 0.0,  # for Newmark time integration only
    "gamma": 0.5,  # for Newmark time integration only
}

model["parallelism"] = {
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

model["mesh"] = {
    "Lz": 0.75,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["PML"] = {
    "status": True,  # True,  # True or false
    "outer_bc": "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 1,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.1, 0.75)],
    "frequency": 8.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": Spyro.create_receiver_transect(
        (-0.10, 0.1), (-0.10, 1.4), 100
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 2.0,  # Final time for event
    "dt": 0.00025,  # timestep size
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 50,  # how frequently to save solution to RAM
}


comm = Spyro.utils.mpi_init(model)

mesh = RectangleMesh(100, 200, 1.0, 2.0)

mesh.coordinates.dat.data[:, 0] -= 1.0
mesh.coordinates.dat.data[:, 1] -= 0.25

element = Spyro.domains.space.FE_method(mesh, "KMV", 2)
V = FunctionSpace(mesh, element)

x, y = SpatialCoordinate(mesh)
velocity = conditional(x > -0.35, 1.5, 3.0)
vp = Function(V, name="velocity").interpolate(velocity)

sources = Spyro.Sources(model, mesh, V, comm).create()
receivers = Spyro.Receivers(model, mesh, V, comm).create()

t1 = time.time()
p_field, p_recv = Spyro.solvers.Leapfrog(
    model, mesh, comm, vp, sources, receivers, source_num=0
)
print(time.time())

print(norm(p_field[-1]))
