import os
import numpy as np

from firedrake import *

import spyro

from .inputfiles.Model1_Leapfrog_adjoint_2d import model


outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")


def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
    )
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)
    return vp_guess


def _simulate_exact(model, mesh, comm, vp_exact, sources, receivers):
    """Simulate the observed data"""
    p_exact, p_exact_recv = spyro.solvers.Leapfrog(
        model, mesh, comm, vp_exact, sources, receivers, output=False
    )
    return p_exact, p_exact_recv


def _simulate_guess(model, mesh, comm, vp_guess, sources, receivers):
    """Simulate the guess data"""
    p_guess, p_guess_recv = spyro.solvers.Leapfrog(
        model, mesh, comm, vp_guess, sources, receivers, output=False
    )
    return p_guess, p_guess_recv


def _compute_misfit(model, mesh, comm, guess, exact):
    """Compute the residual"""
    misfit = spyro.utils.evaluate_misfit(model, comm, guess, exact)
    return misfit


def _compute_functional(model, mesh, comm, misfit):
    """Compute the L2 norm functional"""
    J = spyro.utils.compute_functional(model, comm, misfit)
    return J


def _compute_gradient(model, mesh, comm, vp_guess, misfit, p_guess):
    """"Compute the gradient of the functional and the functional"""
    grad = spyro.solvers.Leapfrog_adjoint(model, mesh, comm, vp_guess, p_guess, misfit)
    outfile_total_gradient.write(grad, name="TotalGradient")
    return grad


def test_gradient_talyor_remainder():

    comm = spyro.utils.mpi_init(model)

    mesh, V = spyro.io.read_mesh(model, comm)

    vp_exact = _make_vp_exact(V, mesh)

    vp_guess = _make_vp_guess(V, mesh)

    sources = spyro.Sources(model, mesh, V, comm).create()

    receivers = spyro.Receivers(model, mesh, V, comm).create()

    # simulate the exact model
    _, p_exact_recv = _simulate_exact(model, mesh, comm, vp_exact, sources, receivers)

    # simulate the guess model
    p_guess, p_guess_recv = _simulate_guess(
        model, mesh, comm, vp_guess, sources, receivers
    )

    # compute the misfit
    misfit = _compute_misfit(model, mesh, comm, p_guess_recv, p_exact_recv)

    # compute the gradient of the control (to be verified)
    grad = _compute_gradient(model, mesh, comm, vp_guess, misfit, p_guess)

    # compute the functional
    J = []
    J.append(_compute_functional(model, mesh, comm, misfit))

    delta_m = Function(V).assign(0.50)
    step = 0.50  #

    remainder = []
    for i in range(3):
        vp_guess = _make_vp_guess(V, mesh)
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        vp_guess.dat.data[:] += step * delta_m.dat.data[:]
        # simulate the guess model with the new vp_guess
        _, p_guess_recv = _simulate_guess(
            model, mesh, comm, vp_guess, sources, receivers
        )
        # misfit calculation
        misfit = _compute_misfit(model, mesh, comm, p_guess_recv, p_exact_recv)
        # compute the functional (again)
        J.append(_compute_functional(model, mesh, comm, misfit))
        # compute the second-order Taylor remainder
        remainder.append(
            J[i + 1] - J[0] - step * np.dot(grad.dat.data[:], delta_m.dat.data[:])
        )
        # assemble(grad * delta_m * dx))
        # halve the step and repeat
        step /= 2.0

    # remainder should decrease at a second order rate
    remainder = np.array(np.abs(remainder))
    l2conv = np.log2(remainder[:-1] / remainder[1:])
    # print(remainder)
    print(l2conv)
    assert (l2conv > 1.8).all()


if __name__ == "__main__":
    test_gradient_talyor_remainder()
