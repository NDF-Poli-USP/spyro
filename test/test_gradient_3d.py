import numpy as np
import pytest 

from firedrake import *

import spyro
from spyro.domains import quadrature

from .inputfiles.Model1_gradient_3d_pml import model_pml

# outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")

forward = spyro.solvers.forward
gradient = spyro.solvers.gradient
functional = spyro.utils.compute_functional


def _make_vp_exact_pml(V, mesh):
    """Create a half space"""
    z, x, y = SpatialCoordinate(mesh)
    velocity = conditional(z > -0.5, 1.5, 4.0)
    vp_exact = Function(V, name="vp").interpolate(velocity)
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x, y = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)
    return vp_guess

@pytest.mark.skip(reason="no way of currently testing this")
def test_gradient_3d():
    _test_gradient(model_pml, pml=True)

def _test_gradient(options, pml=False):

    comm = spyro.utils.mpi_init(options)

    mesh, V = spyro.io.read_mesh(options, comm)


    vp_exact = _make_vp_exact_pml(V, mesh)

    #dt = spyro.estimate_timestep(mesh, V, vp_exact)
    #print(dt, flush=True)

    z, x, y = SpatialCoordinate(mesh)
    Lx = model_pml["mesh"]["Lx"]
    Lz = model_pml["mesh"]["Lz"]
    Ly = model_pml["mesh"]["Ly"]
    x1 = 0.0
    x2 = Lx
    z1 = 0.0
    z2 = -Lz
    y1 = 0.0 
    y2 = Ly 
    boxx1     = Function(V).interpolate(conditional(x>x1 ,1.0,0.0))
    boxx2     = Function(V).interpolate( conditional(x<Lx,1.0,0.0) )

    boxy1     = Function(V).interpolate(conditional(y>y1 ,1.0,0.0))
    boxy2     = Function(V).interpolate( conditional(y<Ly,1.0,0.0) )

    boxz1     = Function(V).interpolate( conditional(z>z2,1.0,0.0) )
    box1      = Function(V).interpolate( boxx1 * boxx2 * boxz1 * boxy1 * boxy2 )
    File("inner-product-energy.pvd").write(box1)

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

    steps = [1e-3, 1e-4, 1e-5]#, 1e-6]  # step length

    delta_m = Function(V)  # model direction (random)
    #delta_m.vector()[:] = 0.2  # np.random.rand(V.dim())
    delta_m.assign(dJ)

    # this deepcopy is important otherwise pertubations accumulate
    vp_original = vp_guess.copy(deepcopy=True)

    errors = []
    for step in steps: #range(3):
        #steps.append(step)
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
            receivers,
        )

        Jp = functional(options, p_exact_recv - p_guess_recv)
        projnorm = assemble(box1*dJ * delta_m * dx(rule=qr_x))
        fd_grad = (Jp - Jm) / step
        print(
            "\n Cost functional for step "
            + str(step)
            + " : "
            + str(Jp)
            + ", fd approx.: "
            + str(fd_grad)
            + ", grad'*dir : "
            + str(projnorm)
            + " \n ",
        )

        errors.append(100 * ((fd_grad - projnorm) / projnorm))
        #step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 1.0).all()


if __name__ == "__main__":
    test_gradient_3d(model)
