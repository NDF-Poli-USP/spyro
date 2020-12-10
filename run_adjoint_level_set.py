from mpi4py import MPI
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
    "status": True,  # true,  # true or false
    "outer_bc": "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 1,
    "cmax": 4.5,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 2,
    "source_pos": [
        (-0.10, 0.25),
        (-0.10, 0.75),
    ],  # spyro.create_receiver_transect((-0.10, 0.30), (-0.10, 1.20), 4),
    "frequency": 10.0,
    "delay": 1.0,
    "ampltiude": 1,
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

#### end of options ####


def calculate_indicator_from_mesh(mesh):
    """Create an indicator function
    assumes the sudomains are labeled 10 and 11
    """
    dgV = FunctionSpace(mesh, "DG", 0)
    indicator = Function(dgV)
    u = TrialFunction(dgV)
    v = TestFunction(dgV)
    solve(u * v * dx == 1 * v * dx(10) + -1 * v * dx(11), indicator)
    return indicator


def update_velocity(q, vp):
    """Update the velocity (material properties)
    based on the indicator
    """
    sd1 = SubDomainData(q < 0)
    sd2 = SubDomainData(q > 0)

    vp.interpolate(Constant(4.5), subset=sd1)
    vp.interpolate(Constant(2.0), subset=sd2)

    # write the current status to disk
    evolution_of_velocity.write(vp, name="control")
    return vp


def calculate_functional(model, mesh, comm, vp, sources, receivers):
    """Calculate the l2-norm functional"""
    J_local = np.zeros((1))
    J_total = np.zeros((1))
    print("Computing the functional")
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            guess, guess_dt, guess_recv = spyro.solvers.Leapfrog_level_set(
                model, mesh, comm, vp, sources, receivers, source_num=sn
            )
            p_exact_recv = spyro.io.load_shots(
                "forward_exact_level_set" + str(sn) + ".dat"
            )
            residual = spyro.utils.evaluate_misfit(
                model, comm, p_exact_recv, guess_recv
            )
            J_local[0] += spyro.utils.compute_functional(model, comm, residual)
    if comm.ensemble_comm.size > 1:
        COMM_WORLD.Allreduce(J_local, J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size
    return (
        J_total[0],
        guess,
        guess_dt,
        residual,
    )


def calculate_gradient(model, mesh, comm, vp, guess, guess_dt, residual):
    """Calculate the shape gradient"""
    print("Computing the gradient", flush=True)
    # gradient is scaled because it appears very small??
    scale = 1e11
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            theta_local = spyro.solvers.Leapfrog_adjoint_level_set(
                model,
                mesh,
                comm,
                vp,
                guess,
                guess_dt,
                residual,
                source_num=sn,
            )
    # sum shape gradient if ensemble parallelism here
    if comm.ensemble_comm.size > 1:
        theta = theta_local.copy()
        comm.ensemble_comm.Allreduce(
            theta_local.dat.data[:], theta.dat.data[:], op=MPI.SUM
        )
    theta *= scale
    return theta


def model_update(mesh, indicator, theta, step):
    """Solve a transport equation to move the subdomains around based
    on the shape gradient which hopefully minimizes the functional.
    """
    print("Updating the shape...", flush=True)
    indicator_new = spyro.solvers.advect(
        mesh, indicator, step * theta, number_of_timesteps=100
    )
    return indicator_new


def optimization(model, mesh, comm, vp, sources, receivers, max_iter=10):
    """Optimization with a line search algorithm"""
    beta0 = beta0_init = 1.5
    max_ls = 3
    gamma = gamma2 = 0.8

    indicator = calculate_indicator_from_mesh(mesh)

    # the file that contains the shape gradient each iteration
    grad_file = File("theta.pvd")

    ls_iter = 0
    iter_num = 0
    # some very large number to start for the functional
    J_old = 9999999.0
    while iter_num < max_iter:
        # calculate the new functional for the new model
        J_new, guess_new, guess_dt_new, residual_new = calculate_functional(
            model, mesh, comm, vp, sources, receivers
        )
        # compute the shape gradient for the new domain
        theta = calculate_gradient(
            model, mesh, comm, vp, guess_new, guess_dt_new, residual_new
        )
        grad_file.write(theta, name="grad")
        # update the new shape...solve transport equation
        indicator_new = model_update(mesh, indicator, theta, beta0)
        # update the velocity
        vp_new = update_velocity(indicator_new, vp)
        # using some basic logic attempt to reduce the functional
        if J_new < J_old:
            print(
                "Iteration "
                + str(iter_num)
                + " : Accepting shape update...functional is: "
                + str(J_new),
                flush=True,
            )
            iter_num += 1
            # accept new domain
            J_old = J_new
            guess = guess_new
            guess_dt = guess_dt_new
            residual = residual_new
            indicator = indicator_new
            vp = vp_new
            # update step
            if ls_iter == max_ls:
                beta0 = max(beta0 * gamma2, 0.1 * beta_0_init)
            elif ls_iter == 0:
                beta0 = min(beta0 / gamma2, 1.0)
            else:
                # no change to step
                beta0 = beta0
            ls_iter = 0
        elif ls_iter < 3:
            print("Line search " + str(ls_iter) + "...reducing step...", flush=True)
            # advance the line search counter
            ls_iter += 1
            # reduce step length by gamma
            beta0 *= gamma
            # now solve the transport equation over again
            # but with the reduced step
        else:
            raise ValueError("failed to reduce the functional...")

    return vp


# run the script

# visualize the updates
evolution_of_velocity = File("evolution_of_velocity.pvd")

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

# the "velocity model"
vp = Function(V)

# create initial velocity field
q = calculate_indicator_from_mesh(mesh)

# initial velocity field
vp = update_velocity(q, vp)

# spyro stuff
sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

# run the optimization based on a line search
vp = optimization(model, mesh, comm, vp, sources, receivers)
