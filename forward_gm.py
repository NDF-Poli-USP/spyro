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
    "truemodel": "velocity_models/gm_2019.hdf5",
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
    "num_sources": 1,
    "source_pos": spyro.create_transect((-0.1, 2), (-0.1, 16.0), 1),
    "amplitude": 1.0,
    "num_receivers": 200,
    "receiver_locations": create_transect((-0.2, 2), (-0.2, 16.0), 200),
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
    # from   scipy.ndimage     import gaussian_filter
    # Z = gaussian_filter(Z,sigma=5)
comm    = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

#velocity model
vp = spyro.io.interpolate(model,Z,mesh, V, guess=False)
File("guess_vel_gm.pvd").write(vp)
quit()
model["acquisition"]["source_pos"] = [[-0.1,8.5]]
source_pos  =  model["acquisition"]["source_pos"]
receivers   = spyro.Receivers(model,mesh,V,comm)
point_cloud = receivers.setPointCloudRec(comm,paralel_z=False)
print(source_pos)
def runForward(sn):
    solver  = spyro.solver_AD(Aut_Dif=False)
    end   = tm.time()
    rec   = solver.wave_propagation(model,mesh,comm,vp,point_cloud,source_pos[sn])
    final = tm.time()
    print(final-end)
    spyro.plots.plot_shots(model, comm, rec,show=True,save=False)
    spyro.io.save_shots(model, comm, rec,file_name="mm/shotgm_"+str(sn))

nshots = len(source_pos)
import multiprocessing as mp
p = mp.Pool(nshots)
p.map(runForward, [0])
p.close()
p.join()


