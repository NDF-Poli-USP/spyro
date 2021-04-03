import numpy as np
from mpi4py import MPI

from .. import solvers
from .. import utils
from .. import io


import from firedrake import Constant, dx, dot, div, ds, dS, COMM_WORLD
import firedrake as fd


import gc

gc.disable()


def _advect(mesh, q, u, number_of_timesteps=10, output=False):
    """Advect a mesh with an indicator function based on the shape gradient `theta`
    solves a transport equation for `number_of_timesteps` using an upwinding DG scheme
    explictly marching in time with a 4th order RK scheme.
    """

    V = fd.FunctionSpace(mesh, "DG", 0)

    dt = 0.0001
    T = dt * number_of_timesteps
    dtc = Constant(dt)
    q_in = Constant(1.0)

    dq_trial = fd.TrialFunction(V)
    phi = fd.TestFunction(V)
    a = phi * dq_trial * dx

    n = fd.FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))

    L1 = dtc * (
        q * div(phi * u) * dx
        - fd.conditional(dot(u, n) < 0, phi * dot(u, n) * q_in, 0.0) * ds
        - fd.conditional(dot(u, n) > 0, phi * dot(u, n) * q, 0.0) * ds
        - (phi("+") - phi("-")) * (un("+") * q("+") - un("-") * q("-")) * dS
    )

    q1 = fd.Function(V)
    q2 = fd.Function(V)
    L2 = fd.replace(L1, {q: q1})
    L3 = fd.replace(L1, {q: q2})

    dq = fd.Function(V)

    params = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}
    prob1 = fd.LinearVariationalProblem(a, L1, dq)
    solv1 = fd.LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = fd.LinearVariationalProblem(a, L2, dq)
    solv2 = fd.LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = fd.LinearVariationalProblem(a, L3, dq)
    solv3 = fd.LinearVariationalSolver(prob3, solver_parameters=params)

    t = 0.0

    if output:
        indicator = fd.File("indicator.pvd")

    step = 0
    while t < T - 0.5 * dt:

        solv1.solve()
        q1.assign(q + dq)

        solv2.solve()
        q2.assign(0.75 * q + 0.25 * (q1 + dq))

        solv3.solve()
        q.assign((1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dq))

        if step % 5 == 0 and output:
            indicator.write(q)

        step += 1
        t += dt

    return q


def _create_weighting_function(V, const=100.0, M=5, width=0.1, show=False):
    """Create a weighting function g, which is large near the
    boundary of the domain and a constant smaller value in the interior

    Inputs
    ------
       V: Firedrake.FunctionSpace
    const: the weight function is equal to this constant value,
           except close to the boundary
    M: maximum value on the boundary will be M**2
    width:  the decimal fraction of the domain where the weight is > constant
    show: Visualize the weighting function

    Outputs
    -------
    wei: a Firedrake.Function containing the weights

    """

    # get coordinates of DoFs
    m = V.ufl_domain()
    W2 = fd.VectorFunctionSpace(m, V.ufl_element())
    coords = fd.interpolate(m.coordinates, W2)
    Z, X = coords.dat.data_ro_with_halos[:,0], coords.dat.data_ro_with_halos[:, 1]

    a0 = np.amin(X)
    a1 = np.amax(X)
    b0 = np.amin(Z)
    b1 = np.amax(Z)

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

    wei = fd.Function(V, w, name="weighting_function")
    return wei


def _calculate_indicator_from_vp(mesh, vp, VP_1=4.5, VP_2=2.0):
    """Create an indicator function assuming two velocities
    4.5 km/s and 2.0 km/s
    """
    dgV = fd.FunctionSpace(mesh, "DG", 0)
    cond = fd.conditional(vp > (VP_1 - 0.1), 1, 2)
    indicator = fd.Function(dgV, name="indicator").interpolate(cond)
    return indicator


def _update_velocity(V, q, vp, VP_1=4.5, VP_2=2.0):
    """Update the velocity (material properties)
    based on the indicator function which assumes two
    velocities 4.5 km./s and 2.0 km/s
    """
    sd1 = fd.SubDomainData(q < 1.5)
    vp_new = fd.Function(V, name="velocity")
    vp_new.assign(Constant(VP_2))
    vp_new.interpolate(Constant(VP_1), subset=sd1)
    return vp_new


def _calculate_functional(
    model, mesh, comm, vp, sources, receivers, iter_num, exact_shot_prefix
):
    """Calculate the l2-norm functional"""
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Computing the cost functional", flush=True)
    J_local = np.zeros((1))
    J_total = np.zeros((1))
    for sn in range(model["acquisition"]["num_sources"]):
        if io.is_owner(comm, sn):
            guess, guess_dt, guess_recv = solvers.Leapfrog_level_set(
                model, mesh, comm, vp, sources, receivers, source_num=sn
            )
            f = exact_shot_prefix + str(sn) + ".dat"
            p_exact_recv = io.load_shots(f)
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
            # END DEBUG

            residual = utils.evaluate_misfit(
                model,
                comm,
                guess_recv,
                p_exact_recv,
            )
            J_local[0] += utils.compute_functional(model, comm, residual)
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


def _calculate_gradient(model, mesh, comm, vp, guess, guess_dt, weighting, residual):
    """Calculate the shape gradient"""
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Computing the shape derivative...", flush=True)
    VF = fd.VectorFunctionSpace(mesh, model["opts"]["method"], model["opts"]["degree"])
    theta = fd.Function(VF, name="gradient")
    for sn in range(model["acquisition"]["num_sources"]):
        if io.is_owner(comm, sn):
            theta_local = solvers.Leapfrog_adjoint_level_set(
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


def _model_update(mesh, comm, indicator, theta, step_size, timesteps):
    """Solve a transport equation to move the subdomains around based
    on the shape gradient which hopefully minimizes the functional.
    """
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("Updating the shape...", flush=True)
    indicator_new = _advect(
        mesh,
        indicator,
        step_size * theta,
        number_of_timesteps=timesteps,
        output=False,
    )
    gc.collect()
    return indicator_new


def optimization(model, mesh, V, comm, vp, sources, receivers):
    """Optimization with steepest descent using a line search algorithm"""

    exact_shot_prefix = model["optimization"]["exact_shot_prefix"]

    beta0 = model["optimization"]["beta0"]
    beta0_init = beta0
    max_ls = model["optimization"]["max_ls"]
    gamma = model["optimization"]["gamma"]
    gamma2 = gamma
    advect_timesteps = model["optimization"]["advect_timesteps"]
    max_iter = model["optimization"]["max_iter"]

    # visualize the updates with this file
    if comm.ensemble_comm.rank == 0:
        evolution_of_velocity = fd.File("evolution_of_velocity.pvd", comm=comm.comm)
        evolution_of_velocity.write(vp, name="velocity")

        # the file that contains the shape gradient each iteration
        grad_file = fd.File("theta.pvd", comm=comm.comm)

    weighting = _create_weighting_function(V, width=0.1, M=10, const=1e-9)

    ls_iter = 0
    iter_num = 0
    # calculate the new functional for the new model
    J_old, guess, guess_dt, residual = _calculate_functional(
        model, mesh, comm, vp, sources, receivers, iter_num, exact_shot_prefix
    )
    while iter_num < max_iter:
        if comm.ensemble_comm.rank == 0 and iter_num == 0 and ls_iter == 0:
            print("Commencing the inversion...", flush=True)

        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print(f"The step size is: {beta0}", flush=True)

        # compute the shape gradient for the new domain
        # (only on the first line search)
        if ls_iter == 0:
            theta = _calculate_gradient(
                model, mesh, comm, vp, guess, guess_dt, weighting, residual
            )
            # write the gradient to a vtk file
            if comm.ensemble_comm.rank == 0:
                grad_file.write(theta, name="gradient")
        # calculate the so-called indicator function by thresholding vp
        indicator = _calculate_indicator_from_vp(mesh, vp)
        # update the new shape by solving the transport equation
        # with the indicator field
        indicator_new = _model_update(
            mesh, comm, indicator, theta, beta0, advect_timesteps
        )
        # update the velocity according to the new indicator
        vp_new = _update_velocity(V, indicator_new, vp)
        # write ALL velocity updates to a vtk file
        if comm.ensemble_comm.rank == 0:
            evolution_of_velocity.write(vp_new)
        # compute the new functional
        J_new, guess_new, guess_dt_new, residual_new = _calculate_functional(
            model, mesh, comm, vp_new, sources, receivers, iter_num, exact_shot_prefix
        )
        # write the new velocity to a vtk file
        # using a line search to attempt to reduce the functional
        if J_new < J_old:
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print(
                    f"Iteration {iter_num}: Cost functional was {J_old}. Accepting shape update...new cost functional is: {J_new}",
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
                beta0 = max(beta0 * gamma2, 0.1 * beta0_init)
            elif ls_iter == 0:
                beta0 = min(beta0 / gamma2, 1.0)
            else:
                # no change to step
                beta0 = beta0
            ls_iter = 0
        elif ls_iter < max_ls:
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print(
                    f"Old cost functional was: {J_old} and the new cost functional is {J_new}",
                    flush=True,
                )
            if abs(J_new - J_old) < 1e-16:
                # increase the step by 1/gamma
                print(
                    f"Line search number {ls_iter}...increasing step size...", flush=True
                )
                beta0 /= gamma
            else:
                print(
                    f"Line search number {ls_iter}...reducing step size...", flush=True
                )
                # reduce step length by gamma
                beta0 *= gamma

            # advance the line search counter
            ls_iter += 1
       
            # now solve the transport equation over again
            # but with the reduced step
            # Need to recompute guess_dt since was discarded above
            # compute the new functional (using old velocity field)
            J, guess, guess_dt, residual = _calculate_functional(
                model, mesh, comm, vp, sources, receivers, iter_num, exact_shot_prefix
            )
        else:
            raise ValueError(
                f"Failed to reduce the functional after {ls_iter} line searches..."
            )

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print(f"Termination: Reached {max_iter} iterations...")
    return vp
