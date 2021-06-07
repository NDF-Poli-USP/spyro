import numpy as np
from pyadjoint import enlisting

from firedrake import *

import spyro
from spyro.domains import quadrature
from firedrake_adjoint import *
from inputfiles.Model1_gradient_2d import model
from inputfiles.Model1_gradient_2d_pml import model_pml

import copy
# outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")
OMP_NUM_THREADS=1
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


def _make_vp_exact_pml(V, mesh):
    """Create a half space"""
    z, x = SpatialCoordinate(mesh)
    velocity = conditional(z > -0.5, 1.5, 4.0)
    vp_exact = Function(V, name="vp").interpolate(velocity)
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)
    return vp_guess


def test_gradient():
    _test_gradient(model)


def test_gradient_pml():
    _test_gradient(model_pml, pml=True)


def _test_gradient(options, pml=False):

    comm = spyro.utils.mpi_init(options)

    mesh, V = spyro.io.read_mesh(options, comm)

    if pml:
        vp_exact = _make_vp_exact_pml(V, mesh)
        z, x = SpatialCoordinate(mesh)
        Lx = model_pml["mesh"]["Lx"]
        Lz = model_pml["mesh"]["Lz"]
        x1 = 0.0
        x2 = Lx
        z1 = 0.0
        z2 = -Lz
        boxx1 = Function(V).interpolate(conditional(x > x1, 1.0, 0.0))
        boxx2 = Function(V).interpolate(conditional(x < Lx, 1.0, 0.0))
        boxz1 = Function(V).interpolate(conditional(z > z2, 1.0, 0.0))
        mask = Function(V).interpolate(boxx1 * boxx2 * boxz1)
        File("mask.pvd").write(mask)
    else:
        vp_exact = _make_vp_exact(V, mesh)

        mask = Function(V).assign(1.0)

    num_rec = model["acquisition"]["num_receivers"]
    # create_transect((0.1, -2.90), (2.9, -2.90), 100)
    δs = np.linspace(0.1, 2.9, num_rec)
    X, Y = np.meshgrid(δs,-2.90)

    xs = np.vstack((X.flatten(), Y.flatten())).T
    
    vp_guess = _make_vp_guess(V, mesh)

    sources = spyro.Sources(options, mesh, V, comm)

    control = Control(vp_guess)
    
    solver  = spyro.solver_AD()
    
    # simulate the exact model
    solver.p_true_rec = solver.forward_AD(model, mesh, comm,
                               vp_exact, sources, xs)
    solver.Calc_Jfunctional = True
    p_rec_guess = solver.forward_AD(model, mesh, comm,
                               vp_guess, sources, xs)
    spyro.plots.plot_shots(
            model,comm,p_rec_guess,show=True,file_name=str(0 + 1),legend=True,save=False
        )
   
    J  = solver.obj_func
    
    dJ   = compute_gradient(J, control)
    Jhat = ReducedFunctional(J, control) 

    with stop_annotating():
        Jm = copy.deepcopy(J)

        qr_x, _, _ = quadrature.quadrature_rules(V)

        print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

        dJ *= mask
        File("gradient.pvd").write(dJ)

        steps = [1e-3, 1e-4, 1e-5]  # , 1e-6]  # step length


        delta_m = Function(V)  # model direction (random)
        delta_m.assign(dJ)
        derivative = enlisting.Enlist(Jhat.derivative())
        hs = enlisting.Enlist(delta_m)

        projnorm = sum(hi._ad_dot(di) for hi, di in zip(hs, derivative))
        
        # this deepcopy is important otherwise pertubations accumulate
        vp_original = vp_guess.copy(deepcopy=True)

        errors = []
        for step in steps:  # range(3):
            
            solver.obj_func   = 0.
            # J(m + delta_m*h)
            vp_guess = vp_original + step*delta_m
            p_rec_guess = solver.forward_AD(model, mesh, comm,
                                vp_guess, sources, xs)  
            Jp = solver.obj_func
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
            # step /= 2

        # all errors less than 1 %
        errors = np.array(errors)
        assert (np.abs(errors) < 5.0).all()


if __name__ == "__main__":
    test_gradient()


