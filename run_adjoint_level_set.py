import time
from advect import advect

from firedrake import *

import Spyro

# 10 outside
# 11 inside

model = {}

model["opts"] = {
    "method": "KMV",
    "variant": None,
    "type": "SIP",  # for dg only - sip, nip and iip
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "_size": 0.005,  # h
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
    "meshfile": "meshes/immersed_disk_v2_guess.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["PML"] = {
    "status": False,  # true,  # true or false
    "outer_bc": None,  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 4.7,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 3,
    "source_pos": [(-0.10, 0.20), (-0.10, 0.50), (-0.10, 0.80)],
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 200,
    "receiver_locations": Spyro.create_receiver_transect(
        (-0.10, 0.1), (-0.10, 0.9), 200
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 1.0,  # final time for event
    "dt": 0.0001,  # timestep size
    "nspool": 200,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to ram
}

comm = Spyro.utils.mpi_init(model)

mesh, V = Spyro.io.read_mesh(model, comm)

vp = [4.5, 2.0]  # inside and outside subdomain respectively in km/s

comm = Spyro.utils.mpi_init(model)

qr_x, _, _ = Spyro.domains.quadrature.quadrature_rules(V)

# Determine subdomains originally specified in the mesh
subdomains = []
subdomains.append(dx(10, rule=qr_x))
subdomains.append(dx(11, rule=qr_x))

indicator = Function(V)
u = TrialFunction(V)
v = TestFunction(V)
solve(u * v * dx == 1 * v * dx(10) + -1 * v * dx(11), indicator)

sources = Spyro.Sources(model, mesh, V, comm).create()

receivers = Spyro.Receivers(model, mesh, V, comm).create()

J_local = np.zeros((1))
J_total = np.zeros((1))

for sn in range(model["acquisition"]["num_sources"]):
    if Spyro.io.is_owner(comm, sn):
        t1 = time.time()
        # run for guess model
        guess, guess_dt, p_recv = Spyro.solvers.Leapfrog_level_set(
            model, mesh, comm, vp, sources, receivers, subdomains, source_num=sn
        )
        # load exact solution
        p_exact_recv = Spyro.io.load_shots("forward_exact_level_set" + str(sn) + ".dat")
        # compute the residual between guess and exact
        residual = Spyro.utils.evaluate_misfit(model, comm, p_exact_recv, p_recv)
        # compute the functional for the current model
        J_local[0] = Spyro.utils.compute_functional(model, comm, residual)
        # sum over all processors
        COMM_WORLD.Allreduce(J_local, J_total, op=MPI.SUM)
        # run the adjoint and return the shape gradient
        theta = Spyro.solvers.Leapfrog_adjoint_level_set(
            model,
            mesh,
            comm,
            vp,
            guess,
            guess_dt,
            residual,
            subdomains,
            source_num=sn,
        )
        print(time.time() - t1, flush=True)
        # solve a transport equation to move the subdomains around
        candidate_indicator, candidate_subdomains = advect(
            mesh, indicator, theta, number_of_timesteps=10
        )
