import numpy as np
from firedrake import *
from firedrake_adjoint import *
from pyadjoint import enlisting
import spyro
from spyro.domains import quadrature
import matplotlib.pyplot as plt
import sys

forward = spyro.solvers.forward_AD

def gradient_test_acoustic(model, mesh, V, comm, vp_exact, vp_guess, mask=None): #{{{
    with stop_annotating():
        print('######## Starting gradient test ########')

        sources = spyro.Sources(model, mesh, V, comm)
        receivers = spyro.Receivers(model, mesh, V, comm)

        wavelet = spyro.full_ricker_wavelet(
            model["timeaxis"]["dt"],
            model["timeaxis"]["tf"],
            model["acquisition"]["frequency"],
        )
        point_cloud = receivers.set_point_cloud(comm,paralel_z=True)
        # simulate the exact model
        print('######## Running the exact model ########')
        p_exact_recv = forward(
            model, mesh, comm, vp_exact, sources, wavelet, point_cloud
        )
    

    # simulate the guess model
    print('######## Running the guess model ########')
    p_guess_recv, Jm = forward(
        model, mesh, comm, vp_guess, sources, wavelet, 
        point_cloud, fwi=True, true_rec=p_exact_recv
    )

    qr_x, _, _ = quadrature.quadrature_rules(V)

    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # compute the gradient of the control (to be verified)
    print('######## Computing the gradient by adjoint method ########')
    control = Control(vp_guess)
    dJ      = compute_gradient(Jm, control)
    if mask:
        dJ *= mask
    
    # File("gradient.pvd").write(dJ)

    #steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # step length
    #steps = [1e-4, 1e-5, 1e-6, 1e-7]  # step length
    steps = [1e-5, 1e-6, 1e-7, 1e-8]  # step length
    with stop_annotating():
        delta_m = Function(V)  # model direction (random)
        delta_m.assign(dJ)
        Jhat    = ReducedFunctional(Jm, control) 
        derivative = enlisting.Enlist(Jhat.derivative())
        hs = enlisting.Enlist(delta_m)
     
        projnorm = sum(hi._ad_dot(di) for hi, di in zip(hs, derivative))
     

        # this deepcopy is important otherwise pertubations accumulate
        vp_original = vp_guess.copy(deepcopy=True)

        print('######## Computing the gradient by finite diferences ########')
        errors = []
        for step in steps:  # range(3):
            # steps.append(step)
            # perturb the model and calculate the functional (again)
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

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 5.0).all()
#}}}
