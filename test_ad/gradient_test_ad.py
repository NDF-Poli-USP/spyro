import numpy as np
import firedrake as fire
from pyadjoint import enlisting
import spyro


def gradient_test_acoustic_ad(
                model, mesh, V, comm, vp_exact, vp_guess, mask=None
                ):
    print('######## Starting gradient test ########')

    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)

    wavelet = spyro.full_ricker_wavelet(
        model["timeaxis"]["dt"],
        model["timeaxis"]["tf"],
        model["acquisition"]["frequency"],
    )

    from spyro.solvers.solver_ad import solver_ad
    # solver_ad = solver_ad(
    #         model, mesh, sources, point_cloud, wavelet, V
    #         )
    solver_ad = solver_ad(model, mesh, V)
    solver_ad.source_num = 0
    wp = solver_ad.wave_propagate
    # simulate the exact model
    print('######## Running the exact model ########')
    solver_ad.model["aut_dif"]["status"] = False
    output = wp(
                comm, vp_exact, sources, receivers, wavelet,
                save_rec_data=True
                )
    p_exact_recv = output[0]
    # p_exact_recv = solver_ad.wave_propagate(comm, vp_exact)

    # simulate the guess model
    print('######## Running the guess model ########')
    solver_ad.model["aut_dif"]["status"] = True
    import firedrake_adjoint as fire_adj
    out = wp(
            comm, vp_guess, sources, receivers, wavelet,
            calc_functional=True, true_rec=p_exact_recv,
            )
    Jm = out[0]
    # Jm = solver_ad.wave_propagate(
    #                 comm, vp_guess, source_num=0,
    #                 calc_misfit=True, true_rec=p_exact_recv
    #                 )
    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # compute the gradient of the control (to be verified)
    print('#### Computing the gradient by automatic differentiation ####')
   
    control = fire_adj.Control(vp_guess)
    dJ = fire_adj.compute_gradient(Jm, control)
    fire.File("grad.pvd", comm=comm.comm).write(dJ)
 
    # steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # step length
    steps = [1e-4, 1e-5, 1e-6, 1e-7]  # step length
    # steps = [1e-3, 1e-4, 1e-5]  # step length
    with fire_adj.stop_annotating():
        delta_m = fire.Function(V)  # model direction (random)
        delta_m.assign(dJ)
        Jhat = fire_adj.ReducedFunctional(Jm, control) 
        derivative = enlisting.Enlist(Jhat.derivative())
        hs = enlisting.Enlist(delta_m)
        fire_adj.get_working_tape().clear_tape()     

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
            out = wp(
                    comm, vp_guess, sources, receivers, wavelet,
                    calc_functional=True, true_rec=p_exact_recv,
                    )
            Jp = out[0]
            # Jp = solver_ad.wave_propagate(
            #                     comm, vp_guess, source_num=0,
            #                     calc_misfit=True, true_rec=p_exact_recv
            #                     )
            
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

    errors = np.array(errors)
    assert (np.abs(errors) < 5.0).all()

