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
            print('######## Starting gradient test ########', flush=True)

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

    h = fire.Function(V)
    h.assign(0.05)
    Jhat = fire_adj.ReducedFunctional(Jm, control)
    conv_rate = fire_adj.taylor_test(Jhat, vp_guess, h)
    # # all errors less than 1 %
    errors = abs(100*(2.0 - conv_rate)/2.0)
 
    assert (errors < 5).all()
