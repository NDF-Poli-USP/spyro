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

# the "velocity model"
vp = [4.5, 2.0]  # inside and outside subdomain respectively in km/s

#### end of options ####

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

comm = spyro.utils.mpi_init(model)

qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)

# Determine subdomains originally specified in the mesh
candidate_subdomains = []
candidate_subdomains.append(dx(10, rule=qr_x))
candidate_subdomains.append(dx(11, rule=qr_x))

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

def calculate_indicator_from_mesh(mesh):
    dgV = FunctionSpace(mesh, "DG", 0)
    indicator = Function(dgV)
    u = TrialFunction(dgV)
    v = TestFunction(dgV)
    solve(u * v * dx == 1 * v * dx(10) + -1 * v * dx(11), indicator)
    return indicator


def calculate_functional(model, mesh, comm, vp, sources, receivers, subdomains):
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            guess, guess_dt, guess_recv = spyro.solvers.Leapfrog_level_set(
                model, mesh, comm, vp, sources, receivers, subdomains, source_num=sn
            )
            p_exact_recv = spyro.io.load_shots(
                "forward_exact_level_set" + str(sn) + ".dat"
            )
            residual = spyro.utils.evaluate_misfit(model, comm, p_exact_recv, p_recv)
            J = spyro.utils.compute_functional(model, comm, residual)
    # todo: sum functional if ensemble parallelism here
    return (
        J,
        guess,
        guess_dt,
        residual,
    )


def calculate_gradient(model, mesh, comm, vp, guess, guess_dt, residual, subdomains):
    """Calculate the shape gradient"""
    scale = 1e9
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            theta= spyro.solvers.Leapfrog_adjoint_level_set(
                model,
                mesh,
                comm,
                vp,
                guess,
                guess_dt,
                residual,
                subdomains,
                source_num=sn,
    # todo: sum shape gradient if ensemble parallelism here
    # scale theta so update moves the functional
    theta *= scale
    return theta


def model_update(mesh, indicator, theta, step):
    """Solve a transport equation to move the subdomains around based
    on the shape gradient.
    """
    new_indicator, new_subdomains = spyro.solvers.advect(
        mesh, indicator, step*theta, number_of_timesteps=10
    )
    return new_indicator, new_subdomains

def line_search(model, mesh, comm, vp, sources, receivers, subdomains, max_number_of_iterations=10):
    """Optimization"""
    beta0=beta0_init=1.5
    max_line_search = 3
    gamma = gamma2 = 0.8

    # compute the functional
    J_old, guess, guess_dt, residual  = calculate_functional(model, mesh, comm, vp, sources, receivers, subdomains)

    iter_num = 0
    while iter_num < max_number_of_iterations:
        line_search_iter = 0
        # compute the shape gradient
        theta = calculate_gradient(model, mesh, comm, vp, guess, guess_dt, residual)
        # update the new shape
        indicator_new, subdomains_new = model_update(mesh, indicator, theta, beta0)
        # calculate the new functional for the new model
        J_new, guess_new, guess_new_dt, residual_new  = calculate_functional(model, mesh, comm, vp, sources, receivers, subdomains_new)
        # using some basic logic reduce the functional
        if J_new < J_old:
            print('Iteration '+str(iter_num)+' : Accepting shape update...functional is '+str(J_new))

            iter_num += 1

            # accept new domain
            J_old = J_new
            guess = guess_new
            guess_dt = guess_dt_new
            residual = residual_new
            subdomains = subdomains_new
            indicator = indicator_new
            # update step
            if line_search_iter == max_line_search:
                beta0 = max(beta0*gamma2, 0.1*beta_0_init)
            elif line_search_iter == 0:
                beta0 = min(beta0/gamma2, 1.0)
            else:
                # no change to step
                beta0 = beta0
        else:
            print('Reducing step...')

            line_search_iter += 1
            # reduce step length by gamma
            beta0 *= gamma
            # solve the transport equation over again but changing the step


    return subdomains, indicator
