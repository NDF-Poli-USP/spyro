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
   
    #J_scale = sqrt(1.e10) #FIXME set it as input
    J_scale = sqrt(1.) #FIXME set it as input

    use_AD_type_interp = True

    print('######## Starting gradient test ########')

    dim = model["opts"]["dimension"]
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)

    if use_AD_type_interp:
        receivers = receivers.setPointCloudRec(comm, paralel_z=True) # FIXME testing

    wavelet = spyro.full_ricker_wavelet(
        model["timeaxis"]["dt"],
        model["timeaxis"]["tf"],
        model["acquisition"]["frequency"],
        amp=model["timeaxis"]["amplitude"]
    )

    # simulate the exact model
    print('######## Running the exact model ########')
    u_exact, uz_exact_recv, ux_exact_recv, uy_exact_recv = forward_elastic_waves(
        model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, receivers, output=True
    )
    
    # simulate the guess model for a perturbation in lambda
    print('######## Running the guess model (lambda) ########')
    u_guess_lamb, uz_guess_recv_lamb, ux_guess_recv_lamb, uy_guess_recv_lamb = forward_elastic_waves(
        model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, receivers, output=False 
    )
    
    # simulate the guess model for a perturbation in mu
    print('######## Running the guess model (mu) ########')
    u_guess_mu, uz_guess_recv_mu, ux_guess_recv_mu, uy_guess_recv_mu = forward_elastic_waves(
        model, mesh, comm, rho, lamb_exact, mu_guess, sources, wavelet, receivers, output=False 
    )
    
    if True:
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

    if use_AD_type_interp: # using functional employed in AD
        p_true_rec = [uz_exact_recv, ux_exact_recv]
        p_rec      = [uz_guess_recv_lamb, ux_guess_recv_lamb]
        
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
            J += 0.5 * assemble(inner(true_rec-rec, true_rec-rec) * dx)
        print(J)
        sys.exit("sys.exit called")

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
    
    qr_x, _, _ = quadrature.quadrature_rules(V)

    J_lamb = np.zeros((1))
    J_mu   = np.zeros((1))
    J_lamb[0] += functional(model, misfit_uz_lamb) # J_scale is already imposed in the misfit
    J_lamb[0] += functional(model, misfit_ux_lamb) # J_scale is already imposed in the misfit
    J_mu[0]   += functional(model, misfit_uz_mu) # J_scale is already imposed in the misfit
    J_mu[0]   += functional(model, misfit_ux_mu) # J_scale is already imposed in the misfit
    if dim==3:
        J_lamb[0] += functional(model, misfit_uy_lamb) # J_scale is already imposed in the misfit
        J_mu[0]   += functional(model, misfit_uy_mu) # J_scale is already imposed in the misfit
    
    # compute the gradient of the control (to be verified)
    print('######## Computing the gradient by adjoint method (lambda) ########')
    dJdl_lamb, dJdm_lamb = gradient_elastic_waves(
        model, mesh, comm, rho, lamb_guess, mu_exact, receivers, 
        u_guess_lamb, misfit_uz_lamb, misfit_ux_lamb, misfit_uy_lamb, output=True
    )
    
    print('######## Computing the gradient by adjoint method (mu) ########')
    dJdl_mu, dJdm_mu = gradient_elastic_waves(
        model, mesh, comm, rho, lamb_exact, mu_guess, receivers, 
        u_guess_mu, misfit_uz_mu, misfit_ux_mu, misfit_uy_mu, output=False
    )
    
    if True:
        File("dJdl_lamb.pvd").write(dJdl_lamb)
        File("dJdm_lamb.pvd").write(dJdm_lamb)
        File("dJdl_mu.pvd").write(dJdl_mu)
        File("dJdm_mu.pvd").write(dJdm_mu)

    #steps = [1e-3, 1e-4, 1e-5, 1e-6]  # step length
    steps = [1e-3, 1e-4]  # step length

    delta_lamb = Function(V)
    delta_mu   = Function(V)
    delta_lamb.assign(lamb_guess)
    delta_mu.assign(mu_guess)
        
    projnorm_dJdl_lamb = assemble(dJdl_lamb * delta_lamb * dx(rule=qr_x))
    projnorm_dJdm_mu   = assemble(dJdm_mu * delta_mu * dx(rule=qr_x))

    # this deepcopy is important otherwise pertubations accumulate
    lamb_original = lamb_guess.copy(deepcopy=True)
    mu_original   = mu_guess.copy(deepcopy=True)

    print('######## Computing the gradient by finite diferences ########')
    errors = []
    for step in steps: 
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        lamb_guess = lamb_original + step * delta_lamb
        mu_guess   = mu_original + step * delta_mu 
        
        _, uz_fd_recv_lamb, ux_fd_recv_lamb, uy_fd_recv_lamb = forward_elastic_waves(
            model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, receivers,
        )
        
        _, uz_fd_recv_mu, ux_fd_recv_mu, uy_fd_recv_mu = forward_elastic_waves(
            model, mesh, comm, rho, lamb_exact, mu_guess, sources, wavelet, receivers,
        )
        
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
        
        Jp_lamb = np.zeros((1))
        Jp_mu   = np.zeros((1))
        Jp_lamb[0] += functional(model, misfit_uz_lamb) # J_scale is already imposed in the misfit
        Jp_lamb[0] += functional(model, misfit_ux_lamb) # J_scale is already imposed in the misfit
        Jp_mu[0]   += functional(model, misfit_uz_mu) # J_scale is already imposed in the misfit
        Jp_mu[0]   += functional(model, misfit_ux_mu) # J_scale is already imposed in the misfit
        if dim==3:
            Jp_lamb[0] += functional(model, misfit_uy_lamb) # J_scale is already imposed in the misfit
            Jp_mu[0]   += functional(model, misfit_uy_mu) # J_scale is already imposed in the misfit

        fd_grad_lamb = (Jp_lamb - J_lamb) / step
        fd_grad_mu   = (Jp_mu - J_mu) / step
       
        print(
            "\n Step " + str(step) + "\n"
            + "\t lambda:\n"
            + "\t cost functional (exact):\t" + str(J_lamb[0]) + "\n"
            + "\t cost functional (FD):\t\t" + str(Jp_lamb[0]) + "\n"
            + "\t grad'*dir (adj):\t\t" + str(projnorm_dJdl_lamb) + "\n"
            + "\t grad'*dir (FD):\t\t" + str(fd_grad_lamb[0]) + "\n"
            + "\n"
            + "\t mu:\n"
            + "\t cost functional (exact):\t" + str(J_mu[0]) + "\n"
            + "\t cost functional (FD):\t\t" + str(Jp_mu[0]) + "\n"
            + "\t grad'*dir (adj):\t\t" + str(projnorm_dJdm_mu) + "\n"
            + "\t grad'*dir (FD):\t\t" + str(fd_grad_mu[0]) + "\n"
        )
    
        #errors.append(100 * ((fd_grad - projnorm) / projnorm))
        # step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    #assert (np.abs(errors) < 5.0).all()
#}}}
