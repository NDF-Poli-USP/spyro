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
forward = spyro.solvers.forward_AD

def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0
        + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
        # 5.0 + 0.5 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
    )
    # File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    # File("guess_vel.pvd").write(vp_guess)
    return vp_guess


def test_gradient():
    _test_gradient(model)


def _test_gradient(options):

    comm     = spyro.utils.mpi_init(model)
    mesh, V  = spyro.io.read_mesh(model, comm)
    vp_exact = _make_vp_exact(V, mesh)
    vp_guess = _make_vp_guess(V, mesh)
    with stop_annotating():
        print('######## Starting gradient test ########')

        sources   = spyro.Sources(model, mesh, V, comm)
        receivers = spyro.Receivers(model, mesh, V, comm)

        wavelet = spyro.full_ricker_wavelet(
            model["timeaxis"]["dt"],
            model["timeaxis"]["tf"],
            model["acquisition"]["frequency"],
        )

        point_cloud = receivers.setPointCloudRec(comm,paralel_z=True)

        # simulate the exact model
        print('######## Running the exact model ########')
        p_exact_recv = forward(
            model, mesh, comm, vp_exact, sources, wavelet, point_cloud
        )
        print(p_exact_recv)
    # simulate the guess model
    print('######## Running the guess model ########')
    p_guess_recv, J = forward(
        model, mesh, comm, vp_guess, sources, wavelet, 
        point_cloud, fwi=True, true_rec=p_exact_recv
    )
    
    control  = Control(vp_guess)
    print('######## Computing the gradient ########')
    dJ   = compute_gradient(J, control)
    Jhat = ReducedFunctional(J, control) 
    # File("gradient.pvd").write(dJ)
    with stop_annotating():
        Jm = copy.deepcopy(J)

        print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

        # File("gradient.pvd").write(dJ)
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
            # J(m + delta_m*h)
            vp_guess = vp_original + step * delta_m
            p_guess_recv, Jp = forward(
                model, mesh, comm, vp_guess, sources, wavelet, 
                point_cloud, fwi=True, true_rec=p_exact_recv
            )
 
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
