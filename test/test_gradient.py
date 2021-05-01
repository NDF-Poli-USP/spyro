import numpy as np

from firedrake import *

import spyro
from spyro.domains import quadrature

from .inputfiles.Model1_gradient_2d import model
from .inputfiles.Model1_gradient_2d_pml import model as model_pml


# outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")

forward = spyro.solvers.forward
gradient = spyro.solvers.gradient
functional = spyro.utils.compute_functional


def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0
        + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
        # 5.0 + 0.5 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
    )
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)
    return vp_guess


def test_gradient():
    inputs = [model, model_pml]
    for d in inputs:
        _test_gradient(d)


def _test_gradient(options):

    comm = spyro.utils.mpi_init(options)

    mesh, V = spyro.io.read_mesh(options, comm)

    vp_exact = _make_vp_exact(V, mesh)

    vp_guess = _make_vp_guess(V, mesh)

    sources = spyro.Sources(options, mesh, V, comm).create()

    receivers = spyro.Receivers(options, mesh, V, comm).create()

    wavelet = spyro.full_ricker_wavelet(
        options["timeaxis"]["dt"],
        options["timeaxis"]["tf"],
        options["acquisition"]["frequency"],
    )

    # simulate the exact model
    p_exact, p_exact_recv = forward(
        options,
        mesh,
        comm,
        vp_exact,
        sources,
        wavelet,
        receivers,
    )

    # simulate the guess model
    p_guess, p_guess_recv = forward(
        options,
        mesh,
        comm,
        vp_guess,
        sources,
        wavelet,
        receivers,
    )

    misfit = p_exact_recv - p_guess_recv

    qr_x, _, _ = quadrature.quadrature_rules(V)

    Jm = functional(options, misfit)
    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # compute the gradient of the control (to be verified)
    dJ = gradient(options, mesh, comm, vp_guess, receivers, p_guess, misfit)
    File("gradient.pvd").write(dJ)

    step = 0.01  # step length

    delta_m = Function(V)  # model direction (random)
    delta_m.vector()[:] = 0.2  # np.random.rand(V.dim())

    # this deepcopy is important otherwise pertubations accumulate
    vp_original = vp_guess.copy(deepcopy=True)

    steps = []
    errors = []
    for i in range(4):
        steps.append(step)
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        vp_guess = vp_original + step * delta_m
        _, p_guess_recv = forward(
            options,
            mesh,
            comm,
            vp_guess,
            sources,
            wavelet,
            receivers,  # True
        )

        Jp = functional(model, p_exact_recv - p_guess_recv)
        projnorm = assemble(dJ * delta_m * dx(rule=qr_x))
        fd_grad = (Jp - Jm) / step
        print(
            "\n Cost functional for step "
            + str(step)
            + " : "
            + str(Jp)
            + ", percent. var.: "
            + str(fd_grad)
            + ", theor. value : "
            + str(projnorm)
            + " \n ",
        )

        errors.append(100 * ((fd_grad - projnorm) / projnorm))
        step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 1.0).all()


if __name__ == "__main__":
    test_gradient(model)
