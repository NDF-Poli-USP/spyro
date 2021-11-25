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
    
    print('######## Starting gradient test ########')

    dim = model["opts"]["dimension"]
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    AD = True
    if AD:
        point_cloud = receivers.setPointCloudRec(comm,paralel_z=True)
    
    wavelet = spyro.full_ricker_wavelet(
        model["timeaxis"]["dt"],
        model["timeaxis"]["tf"],
        model["acquisition"]["frequency"],
        amp=model["timeaxis"]["amplitude"]
    )

    # simulate the exact model
    print('######## Running the exact model ########')
    u_exact, uz_exact_recv, ux_exact_recv, uy_exact_recv = forward_elastic_waves(
        model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, point_cloud, output=False
    )
    true_rec = [uz_exact_recv, ux_exact_recv]
    # simulate the guess model
    print('######## Running the guess model ########')
    u_guess, uz_guess_recv, ux_guess_recv, uy_guess_recv, J = forward_elastic_waves(
        model, mesh, comm, rho, lamb_guess, mu_exact, sources, wavelet, point_cloud, output=False, 
        true_rec=true_rec, fwi=True
    )
   
    if False:
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
        plt.savefig('/home/santos/Desktop/grad_test_elastic.png')
        plt.close()
    
    # misfit_uz = calc_misfit(model, uz_guess_recv, uz_exact_recv) # exact - guess
    # misfit_ux = calc_misfit(model, ux_guess_recv, ux_exact_recv) # exact - guess
    # if dim==3:
    #     misfit_uy = calc_misfit(model, uy_guess_recv, uy_exact_recv) # exact - guess
    # else:
    #     misfit_uy = []
    
    if False:
        plt.title("misfits (uz, ux)")
        plt.plot(uz_exact_recv-uz_guess_recv,label='misfit uz 1')
        plt.plot(misfit_uz,label='misfit uz 2',linestyle='--')
        plt.plot(ux_exact_recv-ux_guess_recv,label='misfit ux 1')
        plt.plot(misfit_ux,label='misfit ux 2',linestyle='--')
        plt.legend()
        plt.savefig('/home/santos/Desktop/grad_test_elastic_misfit.png')
        plt.close()
    
    qr_x, _, _ = quadrature.quadrature_rules(V)

    # Jm = np.zeros((1))
    # Jm[0] += functional(model, misfit_uz)
    # Jm[0] += functional(model, misfit_ux)
    # if dim==3:
    #     Jm[0] += functional(model, misfit_uy)
    
    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")
    #sys.exit("sys.exit called")

    # compute the gradient of the control (to be verified)
    print('######## Computing the gradient by adjoint method ########')
    # dJdl, dJdm = gradient_elastic_waves(
    #     model, mesh, comm, rho, lamb_guess, mu_guess, receivers, u_guess, misfit_uz, misfit_ux, misfit_uy, output=False
    # )
    #sys.exit("sys.exit called")
    if mask: # water mask 
        dJdl *= mask
        dJdm *= mask
    
    File("dJdl.pvd").write(dJdl)
    File("dJdm.pvd").write(dJdm)

    #steps = [1e-3, 1e-4, 1e-5, 1e-6]  # step length
    steps = [1e-4, 1e-5, 1e-6]  # step length

    delta_l = Function(V)  # model direction (random)
    delta_l.assign(dJdl)
    delta_m = Function(V)  # model direction (random)
    delta_m.assign(dJdm)

    # this deepcopy is important otherwise pertubations accumulate
    lamb_original = lamb_guess.copy(deepcopy=True)
    mu_original = mu_guess.copy(deepcopy=True)

    print('######## Computing the gradient by finite diferences ########')
    errors = []
    for step in steps:  # range(3):
        # steps.append(step)
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        lamb_guess = lamb_original + step * delta_l
        mu_guess = mu_original #+ step * delta_m FIXME
        
        _, uz_guess_recv, ux_guess_recv, uy_guess_recv = forward_elastic_waves(
            model, mesh, comm, rho, lamb_guess, mu_guess, sources, wavelet, receivers,
        )
        
        misfit_uz = calc_misfit(model, uz_guess_recv, uz_exact_recv) # exact - guess
        misfit_ux = calc_misfit(model, ux_guess_recv, ux_exact_recv) # exact - guess
        if dim==3:
            misfit_uy = calc_misfit(model, uy_guess_recv, uy_exact_recv) # exact - guess
        
        Jp = np.zeros((1))
        Jp[0] += functional(model, misfit_uz)
        Jp[0] += functional(model, misfit_ux)
        if dim==3:
            Jp[0] += functional(model, misfit_uy)
        
        if mask:
            projnorm_m = assemble(mask * dJdm * delta_m * dx(rule=qr_x))
        else:
            projnorm_lamb = assemble(dJdl * delta_l * dx(rule=qr_x))
            projnorm_mu = assemble(dJdm * delta_m * dx(rule=qr_x))
        
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
            + ", grad'*dir (mu) : "
            + str(projnorm_mu)
            + " \n ",
        )

        #errors.append(100 * ((fd_grad - projnorm) / projnorm))
        # step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    #assert (np.abs(errors) < 5.0).all()
#}}}
