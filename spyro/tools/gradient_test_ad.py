import numpy as np
from firedrake import *
from firedrake_adjoint import *
from pyadjoint import enlisting
import spyro
from spyro.domains import quadrature
import matplotlib.pyplot as plt
import sys

forward = spyro.solvers.forward_AD
forward_elastic_waves = spyro.solvers.forward_elastic_waves_AD


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
        point_cloud = receivers.setPointCloudRec(comm,paralel_z=True)
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

    if False:
        ue=[]
        ug=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            ue.append(p_exact_recv[ti][rn])
            ug.append(p_guess_recv[ti][rn])
        plt.title("p")
        plt.plot(ue,label='exact')
        plt.plot(ug,label='guess')
        plt.legend()
        plt.savefig('/home/santos/Desktop/grad_test_acoustic.png')
        plt.close()


    qr_x, _, _ = quadrature.quadrature_rules(V)

    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # compute the gradient of the control (to be verified)
    print('######## Computing the gradient by adjoint method ########')
    control = Control(vp_guess)
    dJ      = compute_gradient(Jm, control)
    if mask:
        dJ *= mask
    
    File("gradient.pvd").write(dJ)

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
def gradient_test_elastic(model, mesh, V, comm, rho, lamb_exact, mu_exact, lamb_guess, mu_guess, mask=None): #{{{
    with stop_annotating():
        print('######## Starting gradient test ########')

        dim = model["opts"]["dimension"]
        sources = spyro.Sources(model, mesh, V, comm)
        receivers = spyro.Receivers(model, mesh, V, comm)
        # Automatic Differentiation status
    
        point_cloud = receivers.setPointCloudRec(comm, paralel_z=True)
        
        wavelet = spyro.full_ricker_wavelet(
            model["timeaxis"]["dt"],
            model["timeaxis"]["tf"],
            model["acquisition"]["frequency"],
            amp=model["timeaxis"]["amplitude"]
        )

        # simulate the exact model
        print('######## Running the exact model ########')
        uz_exact_recv, ux_exact_recv, uy_exact_recv = forward_elastic_waves(
            model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, point_cloud, output=True
        )
        true_rec = [uz_exact_recv, ux_exact_recv]
    
    # simulate the guess model
    print('######## Running the guess model ########')
    uz_guess_recv, ux_guess_recv, uy_guess_recv, Jm = forward_elastic_waves(
        model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, point_cloud, output=False, 
        true_rec=true_rec, fwi=True
    )
   
    if True:
        ue=[]
        ug=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            ue.append(uz_exact_recv[ti][rn])
            ug.append(uz_guess_recv[ti][rn])
        plt.title("u_z")
        plt.plot(ue,label='exact')
        plt.plot(ug,label='guess')
        plt.legend()
        plt.savefig('/home/santos/Desktop/grad_test_elastic_AD.png')
        plt.close()
    
    qr_x, _, _ = quadrature.quadrature_rules(V)
    
    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")
    #sys.exit("sys.exit called")

    # compute the gradient of the control (to be verified)
    print('######## Computing the gradient by automatic differentiation ########')

    control_l  = Control(lamb_guess)
    dJdl       = compute_gradient(Jm, control_l, options={"riesz_representation": "L2"})
 
    # control_m  = Control(mu_guess) 
    # dJdm       = compute_gradient(Jm, control_m)
    
    Jhat_l     = ReducedFunctional(Jm, control_l) 
    # Jhat_m     = ReducedFunctional(Jm, control_m) 
    with stop_annotating():
        #sys.exit("sys.exit called")
        
        File("dJdl_AD.pvd").write(dJdl)
        #File("dJdm_AD.pvd").write(dJdm)

        #steps = [1e-3, 1e-4, 1e-5, 1e-6]  # step length
        steps = [1e-3, 1e-4]  # step length

        delta_l = Function(V)  # model direction (random)
        delta_l.assign(dJdl)
        # delta_m = Function(V)  # model direction (random)
        # delta_m.assign(dJdm)
        derivative_l = enlisting.Enlist(Jhat_l.derivative())
        # derivative_m = enlisting.Enlist(Jhat_m.derivative())
        hs_l = enlisting.Enlist(delta_l)
        # hs_m = enlisting.Enlist(delta_m)

        projnorm_lamb = sum(hi._ad_dot(di) for hi, di in zip(hs_l, derivative_l))
        # projnorm_mu   = sum(hi._ad_dot(di) for hi, di in zip(hs_m, derivative_m))
        # this deepcopy is important otherwise pertubations accumulate
        lamb_original = lamb_guess.copy(deepcopy=True)
        # mu_original   = mu_guess.copy(deepcopy=True)

        print('######## Computing the gradient by finite diferences ########')
        errors = []
        for step in steps:  # range(3):
            # steps.append(step)
            # perturb the model and calculate the functional (again)
            # J(m + delta_m*h)
            lamb_guess = lamb_original + step * delta_l
            # mu_guess   = mu_original #+ step * delta_m FIXME
            
            uz_guess_recv, ux_guess_recv, uy_guess_recv, Jp = forward_elastic_waves(
                model, mesh, comm, rho, lamb_guess, mu_guess, sources, wavelet, point_cloud,
                true_rec=true_rec, fwi=True
            )
                                   
            fd_grad = (Jp - Jm) / step
            print(
                "\n Cost functional for step "
                + str(step)
                + " : "
                + str(Jp)
                + ", fd approx.: "
                + str(fd_grad)
                + ", grad'*dir (lambda) : "
                + str(projnorm_lamb)
                # + ", grad'*dir (mu) : "
                # + str(projnorm_mu)
                + " \n ",
            )

            #errors.append(100 * ((fd_grad - projnorm) / projnorm))
            # step /= 2

        # all errors less than 1 %
        errors = np.array(errors)
        #assert (np.abs(errors) < 5.0).all()
#}}}
