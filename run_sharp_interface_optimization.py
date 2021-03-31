from mpi4py import MPI
import numpy as np
from firedrake import *

import spyro
import gc

gc.disable()

model = {}
model["opts"] = {
    "method": "KMV",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "quadrature": "KMV",
}
model["mesh"] = {
    "Lz": 0.65,  # depth in km - always positive
    "Lx": 1.00,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/immersed_disk_guess_vp.msh",
    "initmodel": "velocity_models/immersed_disk_guess_vp.hdf5",
    "truemodel": "velocity_models/immersed_disk_true_vp.hdf5",
}
model["PML"] = {
    "status": True,  # true,  # true or false
    "outer_bc": "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 5.0,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.50,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.50,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}
recvs = spyro.create_transect((-0.01, 0.01), (-0.01, 0.99), 100)
sources = spyro.create_transect((-0.01, 0.01), (-0.01, 0.99), 4)
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": len(sources),
    "source_pos": sources,
    "frequency": 5.0,
    "delay": 1.0,
    "amplitude": 1.0,
    "num_receivers": len(recvs),
    "receiver_locations": recvs,
}
model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 1.0,  # final time for event
    "dt": 0.0005,  # timestep size
    "nspool": 9999,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to ram
    "skip": 2,
}
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}
#### end of options ####

VP_1 = 4.5  # inside subdomain to be optimized
VP_2 = 2.0  # outside subdomain to be optimized


def calculate_indicator_from_vp(vp):
    """Create an indicator function"""
    dgV = FunctionSpace(mesh, "DG", 0)
    cond = conditional(vp > (VP_1 - 0.1), 1, 2)
    indicator = Function(dgV, name="indicator").interpolate(cond)
    return indicator


def update_velocity(V, q, vp):
    """Update the velocity (material properties)
    based on the indicator function
    """
    sd1 = SubDomainData(q < 1.5)
    # sd2 = SubDomainData(q > 1.5)

    vp_new = Function(V, name="velocity")

    vp_new.assign(Constant(VP_2))
    vp_new.interpolate(Constant(VP_1), subset=sd1)

    return vp_new


def create_weighting_function(V, const=100.0, M=5, width=0.1, show=False):
    """Create a weighting function g, which is large near the
    boundary of the domain and a constant smaller value in the interior

    Inputs
    ------
       V: Firedrake.FunctionSpace
    const: the weight function is equal to this constant value, except close to the boundary
    M:   maximum value on the boundary will be M**2
    width:  the decimal fraction of the domain where the weight is > constant
    show: Visualize the weighting function

    Outputs
    -------
    wei: a Firedrake.Function containing the weights

    """

    # get coordinates of DoFs
    m = V.ufl_domain()
    W2 = VectorFunctionSpace(m, V.ufl_element())
    coords = interpolate(m.coordinates, W2)
    Z, X = coords.dat.data_ro_with_halos[:, 0], coords.dat.data_ro_with_halos[:, 1]

    a0 = np.amin(X)
    a1 = np.amax(X)
    b0 = np.amin(Z)
    b1 = np.amax(Z)

    cx = a1 - a0  # x-coordinate of center of rectangle
    cz = b1 - b0  # z-coordinate of center of rectangle

    def h(t, d):
        L = width * d  # fraction of the domain where the weight is > constant
        return (np.maximum(0.0, M / L * t + M)) ** 2

    w = const * (
        1.0
        + np.maximum(
            h(X - a1, a1 - a0) + h(a0 - X, a1 - a0),
            h(b0 - Z, b1 - b0) + h(Z - b1, b1 - b0),
        )
    )
    if show:
        import matplotlib.pyplot as plt

        plt.scatter(Z, X, 5, c=w)
        plt.colorbar()
        plt.show()

    wei = Function(V, w, name="weighting_function")
    return wei


def calculate_functional(model, mesh, comm, vp, sources, receivers, iter_num):
    """Calculate the l2-norm functional"""
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Computing the cost functional", flush=True)
    J_local = np.zeros((1))
    J_total = np.zeros((1))
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            guess, guess_dt, guess_recv = spyro.solvers.Leapfrog_level_set(
                model, mesh, comm, vp, sources, receivers, source_num=sn
            )
            f = "shots/forward_exact_level_set" + str(sn) + ".dat"
            p_exact_recv = spyro.io.load_shots(f)
            # DEBUG
            # viz the signal at receiver # 100
            if comm.comm.rank == 0:
                import matplotlib.pyplot as plt

                plt.plot(p_exact_recv[:-1:2, 50], "k-")
                plt.plot(guess_recv[:, 50], "r-")
                plt.ylim(-5e-5, 5e-5)
                plt.title("Receiver #100")
                plt.savefig(
                    "comparison_"
                    + str(comm.ensemble_comm.rank)
                    + "_iter_"
                    + str(iter_num)
                    + ".png"
                )
                plt.close()
            ## END DEBUG

            residual = spyro.utils.evaluate_misfit(
                model,
                comm,
                guess_recv,
                p_exact_recv,
            )
            J_local[0] += spyro.utils.compute_functional(model, comm, residual)
    if comm.ensemble_comm.size > 1:
        COMM_WORLD.Allreduce(J_local, J_total, op=MPI.SUM)
        if comm.comm.size > 1:
            J_total[0] /= COMM_WORLD.size
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(f"The cost functional is: {J_total[0]}", flush=True)
    return (
        J_total[0],
        guess,
        guess_dt,
        residual,
    )


def calculate_gradient(model, mesh, comm, vp, guess, guess_dt, weighting, residual):
    """Calculate the shape gradient"""
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Computing the shape derivative...", flush=True)
    VF = VectorFunctionSpace(mesh, model["opts"]["method"], model["opts"]["degree"])
    theta = Function(VF, name="gradient")
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            theta_local = spyro.solvers.Leapfrog_adjoint_level_set(
                model,
                mesh,
                comm,
                vp,
                guess,
                guess_dt,
                weighting,
                residual,
                source_num=sn,
                output=False,
            )
    # sum shape gradient if ensemble parallelism here
    if comm.ensemble_comm.size > 1:
        comm.allreduce(theta_local, theta)
    else:
        theta = theta_local
    return theta


def model_update(mesh, indicator, theta, step):
    """Solve a transport equation to move the subdomains around based
    on the shape gradient which hopefully minimizes the functional.
    """
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Updating the shape...", flush=True)
    indicator_new = spyro.solvers.advect(
        mesh,
        indicator,
        step * theta,
        number_of_timesteps=50,
        output=False,
    )
    gc.collect()
    return indicator_new


def optimization(model, mesh, V, comm, vp, sources, receivers, max_iter=10):
    """Optimization with steepest descent using a line search algorithm"""
    beta0 = beta0_init = 1.5
    max_ls = 3
    gamma = gamma2 = 0.8

    # the file that contains the shape gradient each iteration
    if comm.ensemble_comm.rank == 0:
        grad_file = File("theta.pvd", comm=comm.comm)

    weighting = create_weighting_function(V, width=0.1, M=10, const=1e-9)

    ls_iter = 0
    iter_num = 0
    # calculate the new functional for the new model
    J_old, guess, guess_dt, residual = calculate_functional(
        model, mesh, comm, vp, sources, receivers, iter_num
    )
    while iter_num < max_iter:
        if comm.ensemble_comm.rank == 0 and iter_num == 0 and ls_iter == 0:
            print("Commencing the inversion...", flush=True)

        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print(f"The step size is: {beta0}", flush=True)

        # compute the shape gradient for the new domain (only on the first line search)
        if ls_iter == 0:
            theta = calculate_gradient(
                model, mesh, comm, vp, guess, guess_dt, weighting, residual
            )
            # write the gradient to a vtk file
            if comm.ensemble_comm.rank == 0:
                grad_file.write(theta, name="gradient")
        # calculate the so-called indicator function by thresholding vp
        indicator = calculate_indicator_from_vp(vp)
        # update the new shape by solving the transport equation with the indicator field
        indicator_new = model_update(mesh, indicator, theta, beta0)
        # update the velocity according to the new indicator
        vp_new = update_velocity(V, indicator_new, vp)
        # write ALL velocity updates to a vtk file
        if comm.ensemble_comm.rank == 0:
            evolution_of_velocity.write(vp_new)
        # compute the new functional
        J_new, guess_new, guess_dt_new, residual_new = calculate_functional(
            model, mesh, comm, vp_new, sources, receivers, iter_num
        )
        # write the new velocity to a vtk file
        # using a line search to attempt to reduce the functional
        if J_new < J_old:
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print(
                    f"Iteration {iter_num}: Functional was {J_old}. Accepting shape update...new functional is: {J_new}",
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
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print(J_old, J_new, flush=True)
                print(
                    f"Line search number {ls_iter}...reducing step size...", flush=True
                )
            # advance the line search counter
            ls_iter += 1
            # reduce step length by gamma
            beta0 *= gamma
            # now solve the transport equation over again
            # but with the reduced step
            # Need to recompute guess_dt since was discarded above
            # compute the new functional (using old velocity field)
            J, guess, guess_dt, residual = calculate_functional(
                model, mesh, comm, vp, sources, receivers, iter_num
            )
        else:
            raise ValueError(
                f"Failed to reduce the functional after {ls_iter} line searches..."
            )

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print(f"Termination: Reached {max_iter} iterations...")
    return vp


# run the script

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp = spyro.io.interpolate(model, mesh, V, guess=True)

# visualize the updates with this file
if comm.ensemble_comm.rank == 0:
    evolution_of_velocity = File("evolution_of_velocity.pvd", comm=comm.comm)
    evolution_of_velocity.write(vp, name="velocity")

# Configure the sources and receivers
sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

# run the optimization based on a line search for max_iter iterations
vp = optimization(model, mesh, V, comm, vp, sources, receivers, max_iter=50)
