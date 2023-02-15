import firedrake as fire
import spyro
import gradient_test_ad as grad_ad
import model_set
OMP_NUM_THREADS = 1
# from ..domains import quadrature, space
# @pytest.mark.skip(reason="no way of currently testing this")


def test_gradient_AD():
    vel_model = "horizont_layers"
    model = model_set.model_settings(vel_model)
    comm = spyro.utils.mpi_init(model)
  
    if vel_model == "marmousi":
        mesh, V = spyro.io.read_mesh(model, comm)
 
        vp_exact = spyro.io.interpolate(model, mesh, V)          
        vp_guess = spyro.io.interpolate(
                            model, mesh, V, guess=True)    
    
        fire.File("exact_vel.pvd").write(vp_exact)
        fire.File("guess_vel.pvd").write(vp_guess)

    if vel_model == "horizont_layers":
        mesh = model_set.meshing(model)
        element = spyro.domains.space.FE_method(
                                        mesh, model["opts"]["method"], 
                                        model["opts"]["degree"]
                                        )
        
        V = fire.FunctionSpace(mesh, element)
        vp_exact = model_set._make_vp_pml(V, mesh)
        vp_guess = model_set._make_vp_pml(V, mesh, v0=1.5, v1=1.5)
        fire.File("exact_vel.pvd").write(vp_exact)
        fire.File("exact_guess.pvd").write(vp_guess)
      
    grad_ad.gradient_test_acoustic(
                                model, mesh, 
                                V, comm, vp_exact, 
                                vp_guess
                                )


test_gradient_AD()