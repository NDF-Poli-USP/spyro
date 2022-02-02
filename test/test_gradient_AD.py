from firedrake import *
import spyro
import numpy as np
import meshio
import SeismicMesh
import finat
import pytest

#from ..domains import quadrature, space
@pytest.mark.skip(reason="no way of currently testing this")
def test_gradient_AD():
    model = {}

    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadratrue": "KMV", # Equi or KMV
        "degree": 1,  # p order
        "dimension": 2,  # dimension
        "regularization": False,  # regularization is on?
        "gamma": 1e-5, # regularization parameter
    }

    model["parallelism"] = {
        "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the ABL.
    model["mesh"] = {
        "Lz": 1.5,  # depth in km - always positive
        "Lx": 1.5,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "not_used.msh",
        "initmodel": "not_used.hdf5",
        "truemodel": "not_used.hdf5",
    }

    # Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
    model["BCs"] = {
        "status": False,  # True or False, used to turn on any type of BC
        "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
        "abl_bc": "none",  # none, gaussian-taper, or alid
        "lz": 0.25,  # thickness of the ABL in the z-direction (km) - always positive
        "lx": 0.25,  # thickness of the ABL in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": [(0.75, 0.75)],
        "frequency": 10.0,
        "delay": 1.0,
        "num_receivers": 10,
        "receiver_locations": spyro.create_transect(
        (0.9, 0.2), (0.9, 0.8), 10
        ),
    }
    model["aut_dif"] = {
        "status": True, 
    }

    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": 0.8,  # Final time for event (for test 7)
        "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
        "fspool": 1,  # how frequently to save solution to RAM
    }

    comm = spyro.utils.mpi_init(model)
    mesh = RectangleMesh(100, 100, 1.5, 1.5) # to test FWI, mesh aligned with interface

    element = spyro.domains.space.FE_method(
        mesh, model["opts"]["method"], model["opts"]["degree"]
    )

    V    = FunctionSpace(mesh, element)
    z, x = SpatialCoordinate(mesh)

    vp_exact = Function(V).interpolate( 1.0 + 0.0*x)
    vp_guess = Function(V).interpolate( 0.8 + 0.0*x)

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
            point_cloud = receivers.set_point_cloud(comm)
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

        print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

        # compute the gradient of the control (to be verified)
        print('######## Computing the gradient by automatic differentiation ########')
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

    gradient_test_acoustic(
                            model, 
                            mesh, 
                            V, 
                            comm, 
                            vp_exact, 
                            vp_guess)

