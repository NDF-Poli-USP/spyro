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
    
        point_cloud = receivers.setPointCloudRec(comm, paralel_z=True)
         
        wavelet = spyro.full_ricker_wavelet(
            model["timeaxis"]["dt"],
            model["timeaxis"]["tf"],
            model["acquisition"]["frequency"],
            amp=model["timeaxis"]["amplitude"]
        )

        # simulate the exact model
        print('######## Running the exact model ########')
        u_exact, uz_exact_recv, ux_exact_recv, uy_exact_recv = forward_elastic_waves(
            model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, point_cloud, output=True
        )
        true_rec = [uz_exact_recv, ux_exact_recv]
         
    # simulate the guess model
    print('######## Running the guess model (lambda) ########')
    u_guess_lamb, uz_guess_recv, ux_guess_recv, uy_guess_recv, J_l = forward_elastic_waves(
        model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, point_cloud, output=False, 
        true_rec=true_rec, fwi=True
    )
    
    print('######## Running the guess model (mu) ########')
    u_guess_mu, _, _, _, J_m = forward_elastic_waves(
        model, mesh, comm, rho, lamb_exact, mu_guess, sources, wavelet, point_cloud, output=False, 
        true_rec=true_rec, fwi=True
    )
   
    if True: # print u exact and guess at receiver {{{
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
    #}}}
    if False: # testing J as computed by u_exact-u_guess over the entire domain {{{
        # this generates result similar to adjoint method
        J_new = 0
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        for step in range(nt): 
            J_new += assemble( (0.5 * inner(u_guess[step]-u_exact[step], u_guess[step]-u_exact[step])) * dx)
        
        print("J_new (AD)="+str(J_new))
        control_l  = Control(lamb_guess)
        dJdl       = compute_gradient(J_new, control_l, options={"riesz_representation": "L2"})
        
        File("dJdl_AD.pvd").write(dJdl)    
        sys.exit("sys.exit called")
    #}}}
    if True: # testing J as computed by u_exact-u_guess over a receiver modeled by Gaussian function {{{
        # this generates result similar to adjoint method
        J_new = 0
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        element = spyro.domains.space.FE_method(
            mesh, model["opts"]["method"], model["opts"]["degree"]
        )
        
        V2 = VectorFunctionSpace(mesh, element)
        u_guess_rec = Function(V2)
        u_exact_rec = Function(V2)
        gaussian_mask = Function(V)

        def delta_expr(x0, z, x, sigma_x=500.0):
            return np.exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2))

        p  = receivers.receiver_locations[0]
        nz = receivers.node_locations[:, 0]
        nx = receivers.node_locations[:, 1]
        gaussian_mask.dat.data[:] = delta_expr(p, nz, nx)
        #gaussian_mask.dat.data[:] = 0.5 #FIXME testing
       
        #File("u_exact_AD.pvd").write(gaussian_mask * u_exact[-1])
        J_l=0
        for step in range(nt): 
            u_guess_rec = gaussian_mask * u_guess_lamb[step]
            u_exact_rec = gaussian_mask * u_exact[step]
            J_l += assemble( (0.5 * inner(u_guess_rec-u_exact_rec, u_guess_rec-u_exact_rec)) * dx)
        
        J_m=0
        for step in range(nt): 
            u_guess_rec = gaussian_mask * u_guess_mu[step]
            u_exact_rec = gaussian_mask * u_exact[step]
            J_m += assemble( (0.5 * inner(u_guess_rec-u_exact_rec, u_guess_rec-u_exact_rec)) * dx)
        
        print("J_l (AD)="+str(J_l))
        print("J_m (AD)="+str(J_m))
        control_l  = Control(lamb_guess)
        control_m  = Control(mu_guess)
        dJdl       = compute_gradient(J_l, control_l, options={"riesz_representation": "L2"})
        dJdm       = compute_gradient(J_m, control_m, options={"riesz_representation": "L2"})
        
        File("dJdl_AD.pvd").write(dJdl)    
        File("dJdm_AD.pvd").write(dJdm)    
        sys.exit("sys.exit called")
    #}}}
        
    qr_x, _, _ = quadrature.quadrature_rules(V)
    #sys.exit("sys.exit called")

    # compute the gradient of the control (to be verified)
    print('######## Computing the gradient by automatic differentiation ########')
    control_l  = Control(lamb_guess)
    dJdl       = compute_gradient(J_l, control_l, options={"riesz_representation": "L2"})
 
    control_m  = Control(mu_guess) 
    dJdm       = compute_gradient(J_m, control_m, options={"riesz_representation": "L2"})
   
    Jhat_l     = ReducedFunctional(J_l, control_l) 
    Jhat_m     = ReducedFunctional(J_m, control_m) 
    with stop_annotating():
        if True:
            File("dJdl_AD.pvd").write(dJdl)
            File("dJdm_AD.pvd").write(dJdm)
            sys.exit("sys.exit called")

        steps = [1e-3, 1e-4, 1e-5]  # step length

        delta_l = Function(V)  # model direction (random)
        delta_l.assign(dJdl)
        delta_m = Function(V)  # model direction (random)
        delta_m.assign(dJdm)
        derivative_l = enlisting.Enlist(Jhat_l.derivative())
        derivative_m = enlisting.Enlist(Jhat_m.derivative())
        hs_l = enlisting.Enlist(delta_l)
        hs_m = enlisting.Enlist(delta_m)

        projnorm_lamb = sum(hi._ad_dot(di) for hi, di in zip(hs_l, derivative_l))
        projnorm_mu   = sum(hi._ad_dot(di) for hi, di in zip(hs_m, derivative_m))
        # this deepcopy is important otherwise pertubations accumulate
        lamb_original = lamb_guess.copy(deepcopy=True)
        mu_original   = mu_guess.copy(deepcopy=True)

        print('######## Computing the gradient by finite diferences ########')
        errors_lamb = []
        errors_mu   = []
        for step in steps:  # range(3):
            # perturb the model and calculate the functional (again)
            # J(m + delta_m*h)
            lamb_guess = lamb_original + step * delta_l
            mu_guess   = mu_original + step * delta_m 
            
            _, _, _, Jp_l = forward_elastic_waves(
                model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, point_cloud,
                true_rec=true_rec, fwi=True
            )
            
            _, _, _, Jp_m = forward_elastic_waves(
                model, mesh, comm, rho, lamb_exact, mu_guess, sources, wavelet, point_cloud,
                true_rec=true_rec, fwi=True
            )
                                   
            fd_grad_lamb = (Jp_l - J_l) / step
            fd_grad_mu   = (Jp_m - J_m) / step
            print(
            "\n Step " + str(step) + "\n"
            + "\t lambda:\n"
            + "\t cost functional (exact):\t" + str(J_l) + "\n"
            + "\t cost functional (FD):\t\t" + str(Jp_l) + "\n"
            + "\t grad'*dir (AD):\t\t" + str(projnorm_lamb) + "\n"
            + "\t grad'*dir (FD):\t\t" + str(fd_grad_lamb) + "\n"
            + "\n"
            + "\t mu:\n"
            + "\t cost functional (exact):\t" + str(J_m) + "\n"
            + "\t cost functional (FD):\t\t" + str(Jp_m) + "\n"
            + "\t grad'*dir (AD):\t\t" + str(projnorm_mu) + "\n"
            + "\t grad'*dir (FD):\t\t" + str(fd_grad_mu) + "\n"
            )

            errors_lamb.append(100 * ((fd_grad_lamb - projnorm_lamb) / projnorm_lamb))
            errors_mu.append(100 * ((fd_grad_mu - projnorm_mu) / projnorm_mu))

        # all errors less than 5 %
        errors_lamb = np.array(errors_lamb)
        errors_mu   = np.array(errors_mu)
        assert (np.abs(errors_lamb) < 5.0).all()
        assert (np.abs(errors_mu) < 5.0).all()
#}}}
