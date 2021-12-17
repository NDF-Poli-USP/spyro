import numpy as np
from firedrake import *
import spyro
from spyro.domains import quadrature
import matplotlib.pyplot as plt
import sys

forward = spyro.solvers.forward
gradient = spyro.solvers.gradient
forward_elastic_waves = spyro.solvers.forward_elastic_waves
gradient_elastic_waves = spyro.solvers.gradient_elastic_waves
functional = spyro.utils.compute_functional
calc_misfit = spyro.utils.evaluate_misfit

def gradient_test_acoustic(model, mesh, V, comm, vp_exact, vp_guess, mask=None): #{{{
    
    print('######## Starting gradient test ########')

    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)

    wavelet = spyro.full_ricker_wavelet(
        model["timeaxis"]["dt"],
        model["timeaxis"]["tf"],
        model["acquisition"]["frequency"],
    )

    # simulate the exact model
    print('######## Running the exact model ########')
    p_exact, p_exact_recv = forward(
        model, mesh, comm, vp_exact, sources, wavelet, receivers
    )

    # simulate the guess model
    print('######## Running the guess model ########')
    p_guess, p_guess_recv = forward(
        model, mesh, comm, vp_guess, sources, wavelet, receivers
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

    misfit = p_exact_recv - p_guess_recv

    qr_x, _, _ = quadrature.quadrature_rules(V)

    Jm = functional(model, misfit)
    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # compute the gradient of the control (to be verified)
    print('######## Computing the gradient by adjoint method ########')
    dJ = gradient(model, mesh, comm, vp_guess, receivers, p_guess, misfit)
    if mask:
        dJ *= mask
    
    File("gradient.pvd").write(dJ)

    #steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # step length
    #steps = [1e-4, 1e-5, 1e-6, 1e-7]  # step length
    steps = [1e-5, 1e-6, 1e-7, 1e-8]  # step length

    delta_m = Function(V)  # model direction (random)
    delta_m.assign(dJ)

    # this deepcopy is important otherwise pertubations accumulate
    vp_original = vp_guess.copy(deepcopy=True)

    print('######## Computing the gradient by finite diferences ########')
    errors = []
    for step in steps:  # range(3):
        # steps.append(step)
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        vp_guess = vp_original + step * delta_m
        _, p_guess_recv = forward(
            model, mesh, comm, vp_guess, sources, wavelet, receivers
        )
        
        Jp = functional(model, p_exact_recv - p_guess_recv)
        if mask:
            projnorm = assemble(mask * dJ * delta_m * dx(rule=qr_x))
        else:
            projnorm = assemble(dJ * delta_m * dx(rule=qr_x))
        
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
  
    #FIXME check the difference when sigma_x>500

    #J_scale = sqrt(1.e10) #FIXME set it as input
    J_scale = sqrt(1.) #FIXME set it as input

    use_AD_type_interp = True
    apply_gaussian_func = True # apply a Gaussian function to compute the misfit and the functional (used to compare)

    print('######## Starting gradient test ########')

    dim = model["opts"]["dimension"]
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)

    if use_AD_type_interp:
        receivers = receivers.setPointCloudRec(comm, paralel_z=True) 

    wavelet = spyro.full_ricker_wavelet(
        model["timeaxis"]["dt"],
        model["timeaxis"]["tf"],
        model["acquisition"]["frequency"],
        amp=model["timeaxis"]["amplitude"]
    )

    # simulate the exact model
    print('######## Running the exact model ########')
    u_exact, uz_exact_recv, ux_exact_recv, uy_exact_recv = forward_elastic_waves(
        model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, receivers, output=False,
        use_AD_type_interp=use_AD_type_interp # use to debug
    )
    
    # simulate the guess model for a perturbation in lambda
    print('######## Running the guess model (lambda) ########')
    u_guess_lamb, uz_guess_recv_lamb, ux_guess_recv_lamb, uy_guess_recv_lamb = forward_elastic_waves(
        model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, receivers, output=False,
        use_AD_type_interp=use_AD_type_interp # use to debug
    )
    
    # simulate the guess model for a perturbation in mu
    print('######## Running the guess model (mu) ########')
    u_guess_mu, uz_guess_recv_mu, ux_guess_recv_mu, uy_guess_recv_mu = forward_elastic_waves(
        model, mesh, comm, rho, lamb_exact, mu_guess, sources, wavelet, receivers, output=False,
        use_AD_type_interp=use_AD_type_interp # use to debug
    )
    
    if True: # print u exact and guess at receiver {{{
        ue=[]
        ug=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            ue.append(uz_exact_recv[ti][rn])
            ug.append(uz_guess_recv_lamb[ti][rn])
        plt.title("u_z")
        plt.plot(ue,label='exact')
        plt.plot(ug,label='guess')
        plt.legend()
        plt.savefig('/home/santos/Desktop/grad_test_elastic.png')
        plt.close()
    #}}}
    if False: # testing J as computed by u_exact-u_guess over the entire domain {{{
        # This generates the same result produced by AD
        J_new = 0
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        for step in range(nt): 
            J_new += assemble( (0.5 * inner(u_guess_lamb[step]-u_exact[step], u_guess_lamb[step]-u_exact[step])) * dx)
    
        print("J_new (adj)="+str(J_new))
        misfit_uz_lamb = []
        misfit_ux_lamb = []
        misfit_uy_lamb = []
        dJdl, _ = gradient_elastic_waves(
            model, mesh, comm, rho, lamb_guess, mu_exact, receivers,
            u_guess_lamb, misfit_uz_lamb, misfit_ux_lamb, misfit_uy_lamb, output=False,
            exact=u_exact # to debug
        )
        
        File("dJdl_adj.pvd").write(dJdl)
        sys.exit("sys.exit called")
    #}}}
    if False: # testing J as computed by u_exact-u_guess over a receiver modeled by a gaussian function {{{
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
    
        receivers = spyro.Receivers(model, mesh, V, comm)
        p  = receivers.receiver_locations[0]
        nz = receivers.node_locations[:, 0]
        nx = receivers.node_locations[:, 1]
        gaussian_mask.dat.data[:] = delta_expr(p, nz, nx)
        #gaussian_mask.dat.data[:] = 0.5 #FIXME testing
 
        #File("u_exact_adj.pvd").write(u_exact[-1])
        J_lamb = 0
        for step in range(nt): 
            u_guess_rec = gaussian_mask * u_guess_lamb[step]
            u_exact_rec = gaussian_mask * u_exact[step]
            J_lamb += assemble( (0.5 * inner(u_guess_rec-u_exact_rec, u_guess_rec-u_exact_rec)) * dx)
       
        step=0
        u_guess_copy = [Function(V2) for t in range(nt)]
        u_exact_copy = [Function(V2) for t in range(nt)]
        residual = [Function(V2) for t in range(nt)]
        J_lamb2=0
        for step in range(nt):
            u_guess_copy[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_lamb[step].sub(0).dat.data[:]
            u_guess_copy[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_lamb[step].sub(1).dat.data[:]
            u_exact_copy[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(0).dat.data[:]
            u_exact_copy[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(1).dat.data[:]
            residual[step] = u_exact_copy[step] - u_guess_copy[step]
            #residual[step] = (gaussian_mask**2) * (u_exact[step] - u_guess_lamb[step]) it returns an error
            J_lamb2+=assemble(
                    (gaussian_mask**2)*(0.5*inner(u_guess_lamb[step]-u_exact[step],u_guess_lamb[step]-u_exact[step]))*dx
                    )
    
        print("J_lamb (adj)="+str(J_lamb))
        print("J_lamb2 (adj)="+str(J_lamb2))
        misfit_uz_lamb = []
        misfit_ux_lamb = []
        misfit_uy_lamb = []
        dJdl, _ = gradient_elastic_waves(
            model, mesh, comm, rho, lamb_guess, mu_exact, receivers,
            u_guess_lamb, misfit_uz_lamb, misfit_ux_lamb, misfit_uy_lamb, output=False,
            residual=residual # to debug
            #exact=u_exact_copy, guess_rec=u_guess_copy # to debug
        )
        del residual

        step=0
        J_mu2=0
        u_guess_copy2 = [Function(V2) for t in range(nt)]
        u_exact_copy2 = [Function(V2) for t in range(nt)]
        residual = [Function(V2) for t in range(nt)]
        for step in range(nt):
            u_guess_copy2[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_mu[step].sub(0).dat.data[:]
            u_guess_copy2[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_mu[step].sub(1).dat.data[:]
            u_exact_copy2[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(0).dat.data[:]
            u_exact_copy2[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(1).dat.data[:]
            residual[step] = u_exact_copy2[step] - u_guess_copy2[step]
            J_mu2+=assemble(
                   (gaussian_mask**2)*(0.5*inner(u_guess_mu[step]-u_exact[step],u_guess_mu[step]-u_exact[step]))*dx
                   )
        
        _, dJdm = gradient_elastic_waves(
            model, mesh, comm, rho, lamb_exact, mu_guess, receivers,
            u_guess_mu, misfit_uz_lamb, misfit_ux_lamb, misfit_uy_lamb, output=False,
            residual=residual # to debug
            #exact=u_exact_copy2, guess_rec=u_guess_copy2 # to debug
        )
        
        print("J_mu2 (adj)="+str(J_mu2))
        
        File("dJdl_adj.pvd").write(dJdl)
        File("dJdm_adj.pvd").write(dJdm)
        sys.exit("sys.exit called")
    #}}}

    if use_AD_type_interp: # using functional employed in AD {{{
        p_true_rec = [uz_exact_recv, ux_exact_recv]
        p_rec      = [uz_guess_recv_lamb, ux_guess_recv_lamb]
        #p_rec      = [uz_guess_recv_mu, ux_guess_recv_mu]
        
        P = VectorFunctionSpace(receivers, "DG", 0) #
        J = 0
        dt = model["timeaxis"]["dt"]
        tf = model["timeaxis"]["tf"]
        nt = int(tf / dt)
        for step in range(nt):
            true_rec = Function(P)
            true_rec.sub(0).dat.data[:] = p_true_rec[0][step]
            true_rec.sub(1).dat.data[:] = p_true_rec[1][step]
            rec = Function(P)
            rec.sub(0).dat.data[:] = p_rec[0][step]
            rec.sub(1).dat.data[:] = p_rec[1][step]
            J += 0.5 * assemble(inner(true_rec-rec, true_rec-rec) * dx) # this generates the same output of functional 
        print("J="+str(J))
        #sys.exit("sys.exit called")

        uz_guess_recv_lamb = np.array(uz_guess_recv_lamb)
        uz_exact_recv = np.array(uz_exact_recv)
        ux_guess_recv_lamb = np.array(ux_guess_recv_lamb)
        ux_exact_recv = np.array(ux_exact_recv)
    #}}}

    J_lamb = 0 
    J_mu   = 0
    if apply_gaussian_func:
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        element = spyro.domains.space.FE_method(
            mesh, model["opts"]["method"], model["opts"]["degree"]
        )
        V2 = VectorFunctionSpace(mesh, element)
        u_guess_rec = Function(V2)
        u_exact_rec = Function(V2)
        gaussian_mask = Function(V)
    
        def delta_expr(x0, z, x, sigma_x=20000.0):
            return np.exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2))
        
        receivers = spyro.Receivers(model, mesh, V, comm)
        p  = receivers.receiver_locations[0]
        nz = receivers.node_locations[:, 0]
        nx = receivers.node_locations[:, 1]
        gaussian_mask.dat.data[:] = delta_expr(p, nz, nx)
        
        u_guess_copy = [Function(V2) for t in range(nt)]
        u_exact_copy = [Function(V2) for t in range(nt)]
        residual = [Function(V2) for t in range(nt)]
        for step in range(nt):
            u_guess_copy[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_lamb[step].sub(0).dat.data[:]
            u_guess_copy[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_lamb[step].sub(1).dat.data[:]
            u_exact_copy[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(0).dat.data[:]
            u_exact_copy[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(1).dat.data[:]
            residual[step] = u_exact_copy[step] - u_guess_copy[step]
            #residual[step] = (gaussian_mask**2) * (u_exact[step] - u_guess_lamb[step]) it returns an error
            J_lamb+=assemble(
                    (gaussian_mask**2)*(0.5*inner(u_guess_lamb[step]-u_exact[step],u_guess_lamb[step]-u_exact[step]))*dx
                    )

        misfit_uz_lamb = []
        misfit_ux_lamb = []
        misfit_uy_lamb = []
        print('######## Computing the gradient by adjoint method (lambda) ########')
        dJdl, _ = gradient_elastic_waves(
            model, mesh, comm, rho, lamb_guess, mu_exact, receivers,
            u_guess_lamb, misfit_uz_lamb, misfit_ux_lamb, misfit_uy_lamb, output=False,
            residual=residual 
        )
        del residual
        del u_guess_copy
        del u_exact_copy

        u_guess_copy = [Function(V2) for t in range(nt)]
        u_exact_copy = [Function(V2) for t in range(nt)]
        residual = [Function(V2) for t in range(nt)]
        for step in range(nt):
            u_guess_copy[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_mu[step].sub(0).dat.data[:]
            u_guess_copy[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_guess_mu[step].sub(1).dat.data[:]
            u_exact_copy[step].sub(0).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(0).dat.data[:]
            u_exact_copy[step].sub(1).dat.data[:] = (gaussian_mask.dat.data[:]**2) * u_exact[step].sub(1).dat.data[:]
            residual[step] = u_exact_copy[step] - u_guess_copy[step]
            J_mu+=assemble(
                   (gaussian_mask**2)*(0.5*inner(u_guess_mu[step]-u_exact[step],u_guess_mu[step]-u_exact[step]))*dx
                   )

        print('######## Computing the gradient by adjoint method (mu) ########')
        _, dJdm = gradient_elastic_waves(
            model, mesh, comm, rho, lamb_exact, mu_guess, receivers,
            u_guess_mu, misfit_uz_lamb, misfit_ux_lamb, misfit_uy_lamb, output=False,
            residual=residual
        )
    
        if use_AD_type_interp:
            receivers = receivers.setPointCloudRec(comm, paralel_z=True)

    else:
        misfit_uz_lamb = J_scale * calc_misfit(model, uz_guess_recv_lamb, uz_exact_recv) # exact - guess
        misfit_ux_lamb = J_scale * calc_misfit(model, ux_guess_recv_lamb, ux_exact_recv) # exact - guess
        misfit_uz_mu   = J_scale * calc_misfit(model, uz_guess_recv_mu, uz_exact_recv) # exact - guess
        misfit_ux_mu   = J_scale * calc_misfit(model, ux_guess_recv_mu, ux_exact_recv) # exact - guess
        if dim==3:
            misfit_uy_lamb = J_scale * calc_misfit(model, uy_guess_recv_lamb, uy_exact_recv) # exact - guess
            misfit_uy_mu   = J_scale * calc_misfit(model, uy_guess_recv_mu, uy_exact_recv) # exact - guess
        else:
            misfit_uy_lamb = []
            misfit_uy_mu   = []
        

        J_lamb += functional(model, misfit_uz_lamb) # J_scale is already imposed in the misfit
        J_lamb += functional(model, misfit_ux_lamb) # J_scale is already imposed in the misfit
        J_mu   += functional(model, misfit_uz_mu) # J_scale is already imposed in the misfit
        J_mu   += functional(model, misfit_ux_mu) # J_scale is already imposed in the misfit
        if dim==3:
            J_lamb += functional(model, misfit_uy_lamb) # J_scale is already imposed in the misfit
            J_mu   += functional(model, misfit_uy_mu) # J_scale is already imposed in the misfit
   
        # compute the gradient of the control (to be verified)
        print('######## Computing the gradient by adjoint method (lambda) ########')
        dJdl, _ = gradient_elastic_waves(
            model, mesh, comm, rho, lamb_guess, mu_exact, receivers, 
            u_guess_lamb, misfit_uz_lamb, misfit_ux_lamb, misfit_uy_lamb, output=True
        )
        
        print('######## Computing the gradient by adjoint method (mu) ########')
        _, dJdm = gradient_elastic_waves(
            model, mesh, comm, rho, lamb_exact, mu_guess, receivers, 
            u_guess_mu, misfit_uz_mu, misfit_ux_mu, misfit_uy_mu, output=False
        )
    
    if False:
        print("J_lamb=" + str(J_lamb)) 
        print("J_mu=" + str(J_mu)) 
        sys.exit("sys.exit called")
    
    if True:
        File("dJdl_adj.pvd").write(dJdl)
        File("dJdm_adj.pvd").write(dJdm)
        #sys.exit("sys.exit called")

    delta_lamb = Function(V) # model direction (random)
    delta_lamb.assign(dJdl)
    delta_mu = Function(V)   # model direction (random)
    delta_mu.assign(dJdm)
        
    qr_x, _, _ = quadrature.quadrature_rules(V)
        
    projnorm_lamb = assemble(dJdl * delta_lamb * dx(rule=qr_x))
    projnorm_mu   = assemble(dJdm * delta_mu * dx(rule=qr_x))

    # this deepcopy is important otherwise pertubations accumulate
    lamb_original = lamb_guess.copy(deepcopy=True)
    mu_original   = mu_guess.copy(deepcopy=True)


    print('######## Computing the gradient by finite diferences ########')
    errors_lamb = []
    errors_mu   = []

    h_steps = [1e-3, 1e-4, 1e-5]  # step length
    for h in h_steps: 
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        lamb_guess = lamb_original + h * delta_lamb
        mu_guess   = mu_original + h * delta_mu 
        
        u_guess_lamb, uz_fd_recv_lamb, ux_fd_recv_lamb, uy_fd_recv_lamb = forward_elastic_waves(
            model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, receivers,
            use_AD_type_interp=use_AD_type_interp # use to debug
        )
        
        u_guess_mu, uz_fd_recv_mu, ux_fd_recv_mu, uy_fd_recv_mu = forward_elastic_waves(
            model, mesh, comm, rho, lamb_exact, mu_guess, sources, wavelet, receivers,
            use_AD_type_interp=use_AD_type_interp # use to debug
        )
       
        Jp_lamb = 0
        Jp_mu   = 0
        if apply_gaussian_func:
            for step in range(nt):
                Jp_lamb+=assemble(
                    (gaussian_mask**2)*(0.5*inner(u_guess_lamb[step]-u_exact[step],u_guess_lamb[step]-u_exact[step]))*dx
                    )
                Jp_mu+=assemble(
                    (gaussian_mask**2)*(0.5*inner(u_guess_mu[step]-u_exact[step],u_guess_mu[step]-u_exact[step]))*dx
                    )
        else:
            misfit_uz_lamb = J_scale * calc_misfit(model, uz_fd_recv_lamb, uz_exact_recv) # exact - guess
            misfit_ux_lamb = J_scale * calc_misfit(model, ux_fd_recv_lamb, ux_exact_recv) # exact - guess
            misfit_uz_mu   = J_scale * calc_misfit(model, uz_fd_recv_mu, uz_exact_recv) # exact - guess
            misfit_ux_mu   = J_scale * calc_misfit(model, ux_fd_recv_mu, ux_exact_recv) # exact - guess
            if dim==3:
                misfit_uy_lamb = J_scale * calc_misfit(model, uy_fd_recv_lamb, uy_exact_recv) # exact - guess
                misfit_uy_mu   = J_scale * calc_misfit(model, uy_fd_recv_mu, uy_exact_recv) # exact - guess
            else:
                misfit_uy_lamb = []
                misfit_uy_mu   = []
            
            Jp_lamb += functional(model, misfit_uz_lamb) # J_scale is already imposed in the misfit
            Jp_lamb += functional(model, misfit_ux_lamb) # J_scale is already imposed in the misfit
            Jp_mu   += functional(model, misfit_uz_mu) # J_scale is already imposed in the misfit
            Jp_mu   += functional(model, misfit_ux_mu) # J_scale is already imposed in the misfit
            if dim==3:
                Jp_lamb += functional(model, misfit_uy_lamb) # J_scale is already imposed in the misfit
                Jp_mu   += functional(model, misfit_uy_mu) # J_scale is already imposed in the misfit

        fd_grad_lamb = (Jp_lamb - J_lamb) / h
        fd_grad_mu   = (Jp_mu - J_mu) / h
       
        print(
            "\n Step " + str(h) + "\n"
            + "\t lambda:\n"
            + "\t cost functional (exact):\t" + str(J_lamb) + "\n"
            + "\t cost functional (FD):\t\t" + str(Jp_lamb) + "\n"
            + "\t grad'*dir (adj):\t\t" + str(projnorm_lamb) + "\n"
            + "\t grad'*dir (FD):\t\t" + str(fd_grad_lamb) + "\n"
            + "\n"
            + "\t mu:\n"
            + "\t cost functional (exact):\t" + str(J_mu) + "\n"
            + "\t cost functional (FD):\t\t" + str(Jp_mu) + "\n"
            + "\t grad'*dir (adj):\t\t" + str(projnorm_mu) + "\n"
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
