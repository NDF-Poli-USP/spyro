from spyro.pml import damping
import numpy as np
from pyadjoint import enlisting
import matplotlib.pyplot as plt
from firedrake import *

import spyro
from spyro.domains import quadrature
from firedrake_adjoint import *
from .inputfiles.Model1_gradient_2d import model
from .inputfiles.Model1_gradient_2d_pml import model_pml


import copy
# outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")
OMP_NUM_THREADS=1
forward = spyro.solvers.forward
gradient = spyro.solvers.gradient
functional = spyro.utils.compute_functional


def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0
        + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
        # 5.0 + 0.5 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
    )
    # File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    # File("guess_vel.pvd").write(vp_guess)
    return vp_guess

def createPointCloud(model,mesh):
    num_rec = model["acquisition"]["num_receivers"]
    x0 = model["acquisition"]["receiver_locations"][0,0]
    xn = model["acquisition"]["receiver_locations"][num_rec-1,0]
    δs = np.linspace(x0, xn, num_rec)
    X, Y = np.meshgrid(-0.2, δs)
    xs = np.vstack((X.flatten(), Y.flatten())).T

    point_cloud = VertexOnlyMesh(mesh, xs) 
    P = FunctionSpace(point_cloud, "DG", 0)  
    return P


def test_gradient():
    _test_gradient(model)


def _test_gradient(options):

    comm = spyro.utils.mpi_init(model)
    mesh, V = spyro.io.read_mesh(model, comm)
    vp_exact = _make_vp_exact(V, mesh)
    
    P = createPointCloud(model,mesh)
    source_pos =  model["acquisition"]["source_pos"]
    with stop_annotating(): 
        solver  = spyro.solver_AD(Aut_Dif=False)
    
        i=0
        rec = []
        for sn in source_pos:
            rec.append(solver.wave_propagation(model,mesh,comm,vp_exact,P,sn))
            i+=1

    
    vp_guess = _make_vp_guess(V, mesh)
    control  = Control(vp_guess)

    solver  = spyro.solver_AD(fwi=True,Aut_Dif=True)
    J       = 0
    p_rec,J = solver.wave_propagation(model,mesh,comm,vp_guess,P,source_pos[0],p_true_rec=rec[0],obj_func=J)

    
    dJ   = compute_gradient(J, control)
    Jhat = ReducedFunctional(J, control) 

    with stop_annotating():
        Jm = copy.deepcopy(J)

        print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

        # File("gradient.pvd").write(dJ)
        steps = [1e-3, 1e-4, 1e-5]  # , 1e-6]  # step length
        
        delta_m = Function(V)  # model direction (random)
        delta_m.assign(dJ)
        derivative = enlisting.Enlist(Jhat.derivative())
        hs = enlisting.Enlist(delta_m)

        projnorm = sum(hi._ad_dot(di) for hi, di in zip(hs, derivative))
        
        # this deepcopy is important otherwise pertubations accumulate
        vp_original = vp_guess.copy(deepcopy=True)

        errors = []
        for step in steps:  # range(3):
            # J(m + delta_m*h)
            vp_guess = vp_original + step*delta_m
            J = 0.
            p_rec, Jp = solver.wave_propagation(model,mesh,comm,
                        vp_guess,P,source_pos[0],p_true_rec=rec[0],obj_func=J)
 
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
            # step /= 2

        # all errors less than 1 %
        errors = np.array(errors)
        assert (np.abs(errors) < 5.0).all()




if __name__ == "__main__":
    test_gradient() 
