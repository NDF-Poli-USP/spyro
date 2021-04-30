import numpy as np

from firedrake import *

import spyro

from .inputfiles.Model1_gradient_2d import model


# outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")

forward = spyro.solvers.forwrd
gradient = spyro.solvers.gradient
functional = spyro.utils.compute_functional


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


def test_gradient_talyor_remainder():
    """2nd order Taylor remainder test"""

    comm = spyro.utils.mpi_init(model)

    mesh, V = spyro.io.read_mesh(model, comm)

    vp_exact = _make_vp_exact(V, mesh)

    vp_guess = _make_vp_guess(V, mesh)

    sources = spyro.Sources(model, mesh, V, comm).create()

    receivers = spyro.Receivers(model, mesh, V, comm).create()

    wavelet = spyro.sources.FullRickerWavelet(
        model["timeaxis"]["dt"],
        model["timeaxis"]["tf"],
        model["acquisition"]["frequency"],
    )

    # simulate the exact model
    p_exact, p_exact_recv = forward(
        model,
        mesh,
        comm,
        vp_exact,
        sources,
        wavelet,
        receivers,
    )

    # simulate the guess model
    p_guess, p_guess_recv = forward(
        model,
        mesh,
        comm,
        vp_guess,
        sources,
        wavelet,
        receivers,
    )

    misfit = p_exact_recv - p_guess_recv

    Jm = functional(model, comm, misfit)

    # compute the gradient of the control (to be verified)
    dJ = gradient(model, mesh, comm, vp_guess, receivers, p_guess, misfit)

    step = 0.01  # step length

    delta_m = Function(V)  # model direction (random)
    delta_m.vector()[:] = np.random.rand(V.dim())

    remainders = []
    steps = []
    # this deepcopy is important otherwise pertubations accumulate
    vp_original = vp_guess.copy(deepcopy=True)
    for i in range(4):
        steps.append(step)
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        vp_guess = vp_original + step * delta_m
        _, p_guess_recv = forward(
            model, mesh, comm, vp_guess, sources, wavelet, receivers
        )
        Jp = functional(model, comm, p_exact_recv - p_guess_recv)
        # compute the second-order Taylor remainder
        remainder = np.abs(Jp - Jm - step * np.dot(dJ.vector(), delta_m.vector()))
        remainders.append(remainder)
        # halve the step and repeat
        step /= 2.0

    # remainder should decrease at a second order rate
    remainders = np.array(remainders)
    print(remainders)
    r = []
    for i in range(1, len(steps)):
        r.append(
            np.log(remainders[i] / remainders[i - 1]) / np.log(steps[i] / steps[i - 1])
        )
    print("Computed convergence rates: {}".format(r))
    assert r[-1] > 1.8


if __name__ == "__main__":
    test_gradient_talyor_remainder()
