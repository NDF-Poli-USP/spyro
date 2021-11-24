from firedrake import *
import numpy as np
import finat
from scipy import optimize
import sys
import h5py
import time as tm
from mpi4py import MPI
OMP_NUM_THREADS=1
from memory_profiler import profile

import spyro

import os
from spyro import create_transect

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
    "num_cores_per_shot": 1
}

mesh = {
    "Lz": 7.5,  # depth in km - always positive
    "Lx": 17.312,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/gm.msh",
#     "initmodel": initmodel + ".hdf5",
    "truemodel": "velocity_models/gm_2020.hdf5",
}

BCs = {
    "status": True,  # True,  # True or falset
    "outer_bc": "non-reflective",  # "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "method": "damping", # damping, pml
    "exponent": 2,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 1.0,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 1.0,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 1.0,  # thickness of the pml in the y-direction (km) - always positive
}

acquisition = {
    "source_type": "Ricker",
    "frequency": 7.0,
    "delay": 1.0,
    # "num_sources": 1,
    "num_sources": 10,
    "source_pos": spyro.create_transect((-0.1, 2.0), (-0.1, 16.0), 10),
    "amplitude": 1.0,
    "num_receivers": 200, 
    "receiver_locations": create_transect((-0.2, 2.0), (-0.2, 16.0), 200),
}

timeaxis = {
    "t0": 0.0,  #  Initial time for event
    "tf": 7.0,  # Final time for event
    "dt": 0.001,  # timestep size
    "nspool": 100,  # how frequently to output solution to pvds
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

fname = model["mesh"]["truemodel"]
with h5py.File(fname, "r") as f:
    print(f.get("velocity_model")[()])
    Z = np.asarray(f.get("velocity_model")[()]) 
    Z = Z*1000
    from   scipy.ndimage     import gaussian_filter
    Z = gaussian_filter(Z,sigma=5)

def makeMask(V,mesh):
    # Update the local vp_guess function
    x,z = SpatialCoordinate(mesh)
    boxx1 = Function(V).interpolate(conditional(x < -2.0, 1.0, 0.0))
    boxx2 = Function(V).interpolate(conditional(x > -7.5, 1.0, 0.0))
    boxz1 = Function(V).interpolate(conditional(z > 0.0, 1.0, 0.0))
    boxz2 = Function(V).interpolate(conditional(z < 17.312, 1.0, 0.0))
    mask = Function(V).interpolate(boxx1*boxx2*boxz1*boxz2)

    File("mask.pvd").write(mask)
    return mask


comm    = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

init_vel = spyro.io.interpolate(model,Z,mesh, V, guess=False)
File("guess_vel_gm.pvd").write(init_vel)

source_pos  =  model["acquisition"]["source_pos"]
receivers   = spyro.Receivers(model,mesh,V,comm)
point_cloud = receivers.setPointCloudRec(comm,paralel_z=False)


mask = makeMask(V,mesh)
rec  = []
for sn in range(len(source_pos)): 
    rec.append(spyro.io.load_shots(model, comm, file_name="mm/shotgm_"+str(sn)))

solver  = spyro.solver_AD(fwi=True,Aut_Dif=True)
def runFWI(sn,xi):
    rank = comm.comm.rank
    size = comm.comm.size
    
    vp   = Function(V) 
    n    = len(vp.dat.data[:])
    N    = [comm.comm.bcast(n, r) for r in range(size)]
    indices = np.insert(np.cumsum(N), 0, 0)
    vp.dat.data[:] = xi[indices[rank] : indices[rank+1]]

    if solver.Aut_Dif:
        import firedrake_adjoint
        control = firedrake_adjoint.Control(vp) 
    
    J = 0
    J = solver.wave_propagation(
                    model,
                    mesh,
                    comm,
                    vp,point_cloud,
                    source_pos[sn],
                    p_true_rec=rec[sn],
                    obj_func=J)
    dJdm  = firedrake_adjoint.compute_gradient(J, control) 
    firedrake_adjoint.get_working_tape().clear_tape()    
    dJdm *= mask
    if comm.ensemble_comm.rank == 0:
        control_file = File( "controlgm.pvd", comm=comm.comm)
        control_file.write(vp)
        File("grad_gm.pvd").write(dJdm)

    return J, dJdm.dat.data[:]


obj_ad = []
import multiprocessing as mp
def getGradObg(xi):
    nshots = len(source_pos)
    process_pool = mp.Pool(int(nshots/1))
    data   = [(0, xi),(1, xi),(2, xi),(3, xi),(4, xi),(5, xi),(6, xi),(7, xi),(8, xi),(9, xi)]
    output = process_pool.starmap(runFWI, data)
    process_pool.close()
    process_pool.join()
 
    
    obj  = 0
    grad = 0
    
    for j in range(0,int(nshots)):
        obj  += output[j][0]
        grad += output[j][1]

    obj_ad.append(obj)
    np.savetxt('mm/obj_adgm', obj_ad)
    return obj, grad



vmax       = 5.5
vmin       = 1.5
m0         = init_vel.vector().gather()
bounds     = [(vmin,vmax) for _ in range(len(m0))] 
result_da  = optimize.minimize(getGradObg, m0,method='L-BFGS-B', jac=True, tol = 1e-15, bounds=bounds, options={"disp": True,"eps": 1e-15, "gtol": 1e-15,"maxiter": 50})
