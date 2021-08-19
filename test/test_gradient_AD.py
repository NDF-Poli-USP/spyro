from spyro.pml import damping
import numpy as np
from pyadjoint import enlisting
import matplotlib.pyplot as plt
from firedrake import *

import spyro
from spyro.domains import quadrature
from firedrake_adjoint import *
from .inputfiles.Model1_gradient_2d import model
from .inputfiles.Model1_gradient_2d_pml import model_pml


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


def _make_vp_exact_damping(V, mesh):
    """Create a half space"""
    z, x = SpatialCoordinate(mesh)
    velocity = conditional(z > -0.5, 2, 4)
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


def test_gradient_damping():
    _test_gradient(model_pml, damping=True)


def _test_gradient(options, damping=False):
    with stop_annotating():
        comm = spyro.utils.mpi_init(options)

        mesh, V = spyro.io.read_mesh(options, comm)
        num_rec = options["acquisition"]["num_receivers"]
        if damping:
            vp_exact = _make_vp_exact_damping(V, mesh)
            δs = np.linspace(0.1, 0.9, num_rec)
            X, Y = np.meshgrid(-0.1, δs)
        else:
            vp_exact = _make_vp_exact(V, mesh)
            # create_transect((0.1, -2.90), (2.9, -2.90), 100)
            δs = np.linspace(0.1, 2.9, num_rec)
            X, Y = np.meshgrid(δs,-2.90)

        xs = np.vstack((X.flatten(), Y.flatten())).T
           
        sources = spyro.Sources(options, mesh, V, comm)  
        solver  = spyro.solver_AD()
    
        # simulate the exact options
        solver.p_true_rec = solver.forward_AD(options, mesh, comm,
                                vp_exact, sources, xs)

    
    vp_guess = _make_vp_guess(V, mesh)
    control = Control(vp_guess)
    solver.Calc_Jfunctional = True
    p_rec_guess = solver.forward_AD(options, mesh, comm,
                               vp_guess, sources, xs)
     
    J  = solver.obj_func
    
    dJ   = compute_gradient(J, control)
    Jhat = ReducedFunctional(J, control) 

    with stop_annotating():
        Jm = copy.deepcopy(J)

        qr_x, _, _ = quadrature.quadrature_rules(V)

        print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

        File("gradient.pvd").write(dJ)

        steps = [1e-3] #, 1e-4, 1e-5]  # , 1e-6]  # step length


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
            p_rec_guess = solver.forward_AD(options, mesh, comm,
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
    #or test_gradient_damping() #when the damping is employed

