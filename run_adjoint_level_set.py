import time
from firedrake import *

import spyro

# 10 outside
# 11 inside

model = {}

model["opts"] = {
    "method": "KMV",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

model["parallelism"] = {
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off
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
    "status": True,  # true,  # true or false
    "outer_bc": None,  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 4.5,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 4,
    "source_pos": spyro.create_receiver_transect((-0.10, 0.30), (-0.10, 1.20), 4),
    "frequency": 10.0,
    "delay": 1.0,
    "ampltiude": 1e6,
    "num_receivers": 200,
    "receiver_locations": spyro.create_receiver_transect(
        (-0.10, 0.30), (-0.10, 1.20), 200
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 0.70,  # final time for event
    "dt": 0.0001,  # timestep size
    "nspool": 200,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to ram
}

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp = [4.5, 2.0]  # inside and outside subdomain respectively in km/s

comm = spyro.utils.mpi_init(model)

qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)

# Determine subdomains originally specified in the mesh
candidate_subdomains = []
candidate_subdomains.append(dx(10, rule=qr_x))
candidate_subdomains.append(dx(11, rule=qr_x))

# create the indicator
dgV = FunctionSpace(mesh, "DG", 0)
indicator = Function(dgV)
u = TrialFunction(dgV)
v = TestFunction(dgV)
solve(u * v * dx == 1 * v * dx(10) + -1 * v * dx(11), indicator)

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

J_total = np.zeros((1))

VF = VectorFunctionSpace(mesh, "CG", 1)
theta_global = Function(VF)
outfile_theta = File("theta_global.pvd")

for fwi_itr in range(5):
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            t1 = time.time()
            # run for guess model
            guess, guess_dt, p_recv = spyro.solvers.Leapfrog_level_set(
                model, mesh, comm, vp, sources, receivers, subdomains, source_num=sn
            )
            # load exact solution for this shot
            p_exact_recv = spyro.io.load_shots(
                "forward_exact_level_set" + str(sn) + ".dat"
            )
            # compute the residual between guess and exact
            residual = spyro.utils.evaluate_misfit(model, comm, p_exact_recv, p_recv)
            # compute the functional for the current model
            J_total[0] += spyro.utils.compute_functional(model, comm, residual)
            # run the adjoint and return the shape gradient
            theta_local = spyro.solvers.Leapfrog_adjoint_level_set(
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
        theta_global += theta_local

    if comm.ensemble_comm.size > 1:
        comm.ensemble_comm.Allreduce(
            theta_global.dat.data[:], theta_global.dat.data[:], op=MPI.SUM
        )
    # scale theta so update moves the functional
    theta_global *= 10000
    outfile_theta.write(theta_global)

    # sum functional over all ensembles
    if comm.ensemble_comm.size > 1:
        comm.ensemble_comm.Allreduce(J_total, J_total, op=MPI.SUM)

    if COMM_WORLD.rank == 0:
        print(
            "At iteration " + str(fwi_itr) + " the functional is " + str(J_total[0]),
            flush=True,
        )

    # solve a transport equation to move the subdomains around
    candidate_indicator, candidate_subdomains = spyro.solvers.advect(
        mesh, indicator, theta_global, number_of_timesteps=10
    )
