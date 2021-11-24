from matplotlib.pyplot import show
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

from spyro import utils
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

comm    = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

#velocity model
vp          = _make_vp(V,mesh)
source_pos  =  model["acquisition"]["source_pos"]
receivers   = spyro.Receivers(model,mesh,V,comm)
point_cloud = receivers.setPointCloudRec()


solver  = spyro.solver_AD(Aut_Dif=False)
for sn in range(len(source_pos)):
    end   = tm.time()
    rec   = solver.wave_propagation(model,mesh,comm,vp,point_cloud,source_pos[sn])
    final = tm.time()
    print(final-end)
    spyro.plots.plot_shots(model, comm, rec,show=True,save=False)
    spyro.io.save_shots(model, comm, rec,file_name=str(sn))