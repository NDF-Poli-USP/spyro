import numpy as np
from firedrake import *
from pyadjoint import enlisting
import spyro

forward = spyro.solvers.forward_AD


def gradient_test_acoustic(
    model, mesh, V, comm, vp_exact, vp_guess, mask=None
):
    """Gradient test for the acoustic FWI problem

    Parameters
    ----------
    model : `dictionary`
        Contains simulation parameters and options.
    mesh : a Firedrake.mesh
        2D/3D simplicial mesh read in by Firedrake.Mesh
    V : Firedrake.FunctionSpace object
        The space of the finite elements
    comm : Firedrake.ensemble_communicator
        An ensemble communicator
    vp_exact : Firedrake.Function
        The exact velocity model
    vp_guess : Firedrake.Function
        The guess velocity model
    mask : Firedrake.Function, optional
        A mask for the gradient test

    Returns
    -------
    None
    """
    import firedrake_adjoint as fire_adj

    with fire_adj.stop_annotating():
        if comm.comm.rank == 0:
            print("######## Starting gradient test ########", flush=True)

        sources = spyro.Sources(model, mesh, V, comm)
        receivers = spyro.Receivers(model, mesh, V, comm)

        wavelet = spyro.full_ricker_wavelet(
            model["timeaxis"]["dt"],
            model["timeaxis"]["tf"],
            model["acquisition"]["frequency"],
        )
        point_cloud = receivers.set_point_cloud(comm)
        # simulate the exact model
        if comm.comm.rank == 0:
            print("######## Running the exact model ########", flush=True)
        p_exact_recv = forward(
            model, mesh, comm, vp_exact, sources, wavelet, point_cloud
        )

    # simulate the guess model
    if comm.comm.rank == 0:
        print("######## Running the guess model ########", flush=True)
    p_guess_recv, Jm = forward(
        model,
        mesh,
        comm,
        vp_guess,
        sources,
        wavelet,
        point_cloud,
        fwi=True,
        true_rec=p_exact_recv,
    )
    if comm.comm.rank == 0:
        print(
            "\n Cost functional at fixed point : " + str(Jm) + " \n ",
            flush=True,
        )

    # compute the gradient of the control (to be verified)
    if comm.comm.rank == 0:
        print(
            "######## Computing the gradient by automatic differentiation ########",
            flush=True,
        )
    control = fire_adj.Control(vp_guess)
    dJ = fire_adj.compute_gradient(Jm, control)
    if mask:
        dJ *= mask

    # File("gradient.pvd").write(dJ)

    # steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # step length
    # steps = [1e-4, 1e-5, 1e-6, 1e-7]  # step length
    steps = [1e-5, 1e-6, 1e-7, 1e-8]  # step length
    with fire_adj.stop_annotating():
        delta_m = Function(V)  # model direction (random)
        delta_m.assign(dJ)
        Jhat = fire_adj.ReducedFunctional(Jm, control)
        derivative = enlisting.Enlist(Jhat.derivative())
        hs = enlisting.Enlist(delta_m)

        projnorm = sum(hi._ad_dot(di) for hi, di in zip(hs, derivative))

        #  this deepcopy is important otherwise pertubations accumulate
        vp_original = vp_guess.copy(deepcopy=True)

        if comm.comm.rank == 0:
            print(
                "######## Computing the gradient by finite diferences ########",
                flush=True,
            )
        errors = []
        for step in steps:  # range(3):
            # steps.append(step)
            # perturb the model and calculate the functional (again)
            # J(m + delta_m*h)
            vp_guess = vp_original + step * delta_m
            p_guess_recv, Jp = forward(
                model,
                mesh,
                comm,
                vp_guess,
                sources,
                wavelet,
                point_cloud,
                fwi=True,
                true_rec=p_exact_recv,
            )

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
                    + " \n ",
                    flush=True,
                )

            errors.append(100 * ((fd_grad - projnorm) / projnorm))

    fire_adj.get_working_tape().clear_tape()

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 5.0).all()
