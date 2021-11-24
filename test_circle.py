from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import time as tm
import copy
import finat
from scipy import optimize
import sys
from mpi4py import MPI
from dask.distributed    import Client, wait
OMP_NUM_THREADS=1

import spyro

import os
from spyro import create_transect

# Choose method and parameters
# Choose method and parameters
opts = {
    "method": "KMV",
    "quadrature": "KMV",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

parallelism = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor), custom, off.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    "num_cores_per_shot": 4
}

mesh = {
    "Lz": 2.,  # depth in km - always positive
    "Lx": 2.,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/circle.msh",
}

BCs = {
    "status": False,  # True,  # True or false
    "outer_bc": "non-reflective",  # "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "method": "damping", # damping, pml
    "exponent": 2,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the pml in the z-direction (km) - always positive
    "lx":0.5,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}
acquisition = {
    "source_type": "Ricker",
    "frequency": 7.0,
    "delay": 1.0,
    # "num_sources": 1,
    "num_sources": 4,
    "source_pos": create_transect((0.6, -0.1), (1.4, -0.1), 4),
    "amplitude": 1.0,
    "num_receivers": 10,
    "receiver_locations": create_transect((0.6, -0.2), (1.4, -0.2), 10),
}

timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.0,  # Final time for event
    "dt": 0.001,  # timestep size
    "nspool": 1001,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}  # how freq. to output to files and screen

inversion = {
    "freq_bands": [None]
}  # cutoff frequencies (Hz) for Ricker source and to low-pass the observed shot record


# Create your model with all the options
model = {
    "self": None,
    "inversion": inversion,
    "opts": opts,
    "BCs": BCs,
    "parallelism": parallelism,
    "mesh": mesh,
    "acquisition": acquisition,
    "timeaxis": timeaxis,
}

def _make_vp(V, mesh, vp_guess=False):
    """creating velocity models"""
    x,z = SpatialCoordinate(mesh)
    if vp_guess:
        vp   = Function(V).interpolate(1.5 + 0.0 * x)
        File("guess_vel.pvd").write(vp)
    else:
        vp  = Function(V).interpolate(
            2.5
            + 1 * tanh(20 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2)))
            # 5.0 + 0.5 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
        )
      
        File("exact_vel.pvd").write(vp)
      
    
    return vp

def makeMask(V,mesh):
    # Update the local vp_guess function
    x,z = SpatialCoordinate(mesh)
    boxx1 = Function(V).interpolate(conditional(x > 0.5, 1.0, 0.0))
    boxx2 = Function(V).interpolate(conditional(x < 1.5, 1.0, 0.0))
    boxz1 = Function(V).interpolate(conditional(z < -0.1, 1.0, 0.0))
    boxz2 = Function(V).interpolate(conditional(z > -0.8, 1.0, 0.0))
    mask  = Function(V).interpolate(boxx1*boxx2*boxz1*boxz2)

    return mask


comm    = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

source_pos  =  model["acquisition"]["source_pos"]
receivers   = spyro.Receivers(model,mesh,V,comm)
point_cloud = receivers.setPointCloudRec(comm)


mask = makeMask(V,mesh)
rec  = []
for sn in range(len(source_pos)): 
    rec.append(spyro.io.load_shots(model, comm, file_name=str(sn)))

solver  = spyro.solver_AD(fwi=True,Aut_Dif=True)
def runFWI(xi):
    vp = xi
    if solver.Aut_Dif:
        import firedrake_adjoint
        control = firedrake_adjoint.Control(vp) 
    
    J_total = 0
    for sn in range(len(source_pos)):
        J = 0
        if solver.Aut_Dif:
            J       = solver.wave_propagation(
                            model,
                            mesh,
                            comm,
                            vp,point_cloud,
                            source_pos[sn],
                            p_true_rec=rec[sn],
                            obj_func=J)
        else:
            p, misfit, J= solver.wave_propagation(
                model,
                mesh,
                comm,
                vp,
                point_cloud,
                source_pos[sn],
                p_true_rec=rec,
                obj_func=J)
      
        J_total += J

        # quit()

    # spyro.plots.plot_shots(model,comm,misfit,show=True) 
    if solver.Aut_Dif:   
        # dJdm  = firedrake_adjoint.compute_gradient(J, control) 
        Ĵ     = firedrake_adjoint.ReducedFunctional(J_total, control) 
        q_min = firedrake_adjoint.minimize(Ĵ, options={'disp': True,"eps": 1e-15, "gtol": 1e-15,"maxiter": 10},bounds=(1.5,3.5))
        firedrake_adjoint.get_working_tape().clear_tape()    
    else:
        dJdm   = solver.wave_propagation(
            model,
            mesh,
            comm,
            vp,
            point_cloud,
            receivers.receiver_locations,
            type="adjoint",
            misfit=misfit,
            guess=p)
    
    if comm.ensemble_comm.rank == 0:
        control_file = File( "control.pvd", comm=comm.comm)
        control_file.write(q_min)
    
end   = tm.time()       
init_vel = _make_vp(V,mesh,vp_guess=True)
runFWI(init_vel)
final = tm.time()
print(final-end)