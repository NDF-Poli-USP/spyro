import numpy as np
import firedrake as fire
from pyadjoint import enlisting
import spyro


def gradient_test_acoustic(model, mesh, V, comm, vp_exact, vp_guess, mask=None): #{{{
    import firedrake_adjoint as fire_adj
    solver_ad = spyro.solvers.solver_ad.solver_ad(model, mesh, V)
    wp = solver_ad.wave_propagate
    with fire_adj.stop_annotating():
        if comm.comm.rank == 0:
            print('######## Starting gradient test ########', flush = True)

        sources = spyro.Sources(model, mesh, V, comm)
        receivers = spyro.Receivers(model, mesh, V, comm)

        wavelet = spyro.full_ricker_wavelet(
            model["timeaxis"]["dt"],
            model["timeaxis"]["tf"],
            model["acquisition"]["frequency"],
        )
        
        # simulate the exact model
        if comm.comm.rank == 0:
            print('######## Running the exact model ########', flush = True)
        output = wp(
                comm, vp_exact, sources, receivers, wavelet,
                output=True, save_rec_data=True
                )
        p_exact_recv = output[0]
        

    # simulate the guess model
    if comm.comm.rank == 0:
        print('######## Running the guess model ########', flush = True)
    out = wp(
                comm, vp_guess, sources, receivers, wavelet,
                calc_functional=True, true_rec=p_exact_recv,
                output=True
                )
    Jm = out[0]
    if comm.comm.rank == 0:
        print("\n Cost functional at fixed point : " + str(Jm) + " \n ", flush = True)

    # compute the gradient of the control (to be verified)
    if comm.comm.rank == 0:
        print('######## Computing the gradient by automatic differentiation ########', flush = True)
    control = fire_adj.Control(vp_guess)
    dJ = fire_adj.compute_gradient(Jm, control)
    fire.File("grad.pvd").write(dJ)
    if mask:
        dJ *= mask
    
    # File("gradient.pvd").write(dJ)

    #steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # step length
    #steps = [1e-4, 1e-5, 1e-6, 1e-7]  # step length
    steps = [1e-5]  # step length
    with fire_adj.stop_annotating():
        delta_m = fire.Function(V)  # model direction (random)
        delta_m.assign(dJ)
        Jhat = fire_adj.ReducedFunctional(Jm, control) 
        derivative = enlisting.Enlist(Jhat.derivative())
        hs = enlisting.Enlist(delta_m)
     
        projnorm = sum(hi._ad_dot(di) for hi, di in zip(hs, derivative))
     

        # this deepcopy is important otherwise pertubations accumulate
        vp_original = vp_guess.copy(deepcopy=True)

        if comm.comm.rank == 0:
            print('######## Computing the gradient by finite diferences ########', flush = True)
        errors = []
        for step in steps:  # range(3):
            # steps.append(step)
            # perturb the model and calculate the functional (again)
            # J(m + delta_m*h)
            vp_guess = vp_original + step * delta_m
            out = wp(
                comm, vp_guess, sources, receivers, wavelet,
                calc_functional=True, true_rec=p_exact_recv,
                output=True
                )
            Jp = out[0]
            
            fd_grad = (Jp - Jm) / step
            if comm.comm.rank == 0:
                print(
                    "\n Cost functional for step "
                    + str(step)
                    + " : "
                    + str(Jp)
                    + ", fd approx.: "
                    + str(fd_grad)
                    + ", grad'*dir : "
                    + str(projnorm)
                    + " \n ", flush=True
                )

            errors.append(100 * ((fd_grad - projnorm) / projnorm))
    
    fire_adj.get_working_tape().clear_tape()

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 5.0).all()
#}}}