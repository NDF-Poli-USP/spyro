import time

from firedrake import *

import spyro

# 10 outside
# 11 inside

model = {}

model["opts"] = {
    "method": "KMV",
    "variant": None,
    "type": "SIP",  # for dg only - sip, nip and iip
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "mesh_size": 0.005,  # h
    "beta": 0.0,  # for newmark time integration only
    "gamma": 0.5,  # for newmark time integration only
}

model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

model["mesh"] = {
    "Lz": 1.50,  # depth in km - always positive
    "Lx": 1.50,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/immersed_disk_v2.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["PML"] = {
    "status": False,  # true,  # true or false
    "outer_bc": None,  # "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 1,
    "cmax": 4.7,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 5,
    "source_pos": [(-0.10, 0.20), (-0.10, 0.50), (-0.10, 0.80)],
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 200,
    "receiver_locations": spyro.create_receiver_transect(
        (-0.10, 0.1), (-0.10, 0.9), 200
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 1.0,  # final time for event
    "dt": 0.0001,  # timestep size
    "nspool": 200,  # how frequently to output solution to pvds
    "fspool": 1000,  # how frequently to save solution to ram
}


vp = [4.5, 2.0]  # inside and outside subdomain respectively in km/s

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)

# Determine subdomains originally specified in the mesh
subdomains = []
subdomains.append(dx(10, rule=qr_x))
subdomains.append(dx(11, rule=qr_x))

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

for sn in range(model["acquisition"]["num_sources"]):
    if spyro.io.is_owner(comm, sn):
        t1 = time.time()
        p_field, p_field_dt, p_recv = spyro.solvers.Leapfrog_level_set(
            model, mesh, comm, vp, sources, receivers, subdomains, source_num=sn
        )
        print(time.time() - t1)
        spyro.io.save_shots("forward_exact_level_set" + str(sn) + ".dat", p_recv)
        spyro.plots.plot_shotrecords(
            model,
            p_recv,
            name="level_set_" + str(sn),
            vmin=-1e-5,
            vmax=1e-5,
            appear=False,
        )
