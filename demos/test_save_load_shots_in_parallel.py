from firedrake import *
from scipy.optimize import * 
from movement import *
import spyro
import time
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import h5py
import meshio
#import SeismicMesh
import weakref
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from spyro.io import write_function_to_grid, create_segy
#from ..domains import quadrature, space

# define the model parameters {{{
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV", # Equi or KMV #FIXME it will be removed
    "degree": 2,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    "type": "automatic", # options: automatic (same number of cores for evey processor) or spatial
    #"type": "spatial",  # 
    #"custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    #"num_cores_per_shot": 1 #FIXME this is not used
}

model["mesh"] = {
    #"Lz": 1.5,  # depth in km - always positive
    "Lz": 2.00-.45,  # depth in km - always positive (waterbottom at z=-0.45 km)
    #"Lx": 2.0,  # width in km - always positive
    "Lx": 4.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "method": "damping", # damping, pml
    "exponent": 2,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the pml in the z-direction (km) - always positive
    "lx":0.5,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "frequency": 3.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    #"frequency": 5.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    #"frequency": 7.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    "delay": 1.0, # FIXME check this
    "num_sources": 1, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((2, -0.01-0.45), (1.5, -0.1-0.45), 1), # waterbottom at z=-0.45 km
    #"source_pos": spyro.create_transect((2, -0.01-0.45), (1.5, -0.1-0.45), 2), # waterbottom at z=-0.45 km
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 2, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((2, -0.02-0.45), (1.5, -0.2-0.45), 2), # waterbottom at z=-0.45 km REC1
    #"receiver_locations": spyro.create_transect((0.1, -0.10-0.45), (3.9, -0.10-0.45), 100), # waterbottom at z=-0.45 km REC1
    #"receiver_locations": spyro.create_transect((0.1, -1.9), (3.9, -1.9), 100), # receivers at the bottom of the domain (z=-1.9 km) REC2 
    #"receiver_locations": spyro.create_2d_grid(1, 3, -1.4, -1, 10) # 10^2 points REC3
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 2.5, # Final time for event 
    "tf": 1.0, # Final time for event 
    "dt": 0.00025,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
}
#}}}
# make vp {{{
def _make_vp(V, vp_guess=False, field="velocity_model"):
    
    path = "./velocity_models/elastic-marmousi-model/model/"
    if vp_guess: # interpolate from a smoothed field
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy.hdf5" # domain 4 x 2 km2 (x, y) 
    else: # interpolate from the exact field
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy.hdf5" # domain 4 x 2 km2 (x, y) 
        
    with h5py.File(fname, "r") as f:
        Zo = np.asarray(f.get(field)[()]) # original Marmousi data/domain
        nrow, ncol = Zo.shape
        Lz = 2 # Value defined by the velocity model
        Lx = 4 # Value defined by the velocity model
        zo = np.linspace(-Lz, 0.0, nrow) # original Marmousi data/domain
        xo = np.linspace(0.0,  Lx, ncol) # original Marmousi data/domain
        interpolant = RegularGridInterpolator((xo, zo), np.transpose(Zo))

        m = V.ufl_domain()
        W = VectorFunctionSpace(m, V.ufl_element())
        coords = interpolate(m.coordinates, W)
        xq, zq = coords.dat.data[:, 0], coords.dat.data[:, 1]

        _vp = interpolant((xq, zq))
        vp = Function(V)
        vp.dat.data[:] = _vp / 1000 # m/s -> km/s

        if vp_guess:
            File("guess_vp.pvd").write(vp)
        else:
            File("exact_vp.pvd").write(vp)
    
    return vp
#}}}
comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 20)}

#nx = 200
#ny = math.ceil( 100*(model["mesh"]["Lz"]-0.45)/model["mesh"]["Lz"] ) # (Lz-0.45)/Lz
nx = 75 # FIXME
ny = 25

mesh_ref = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                        distribution_parameters=distribution_parameters)
mesh_ref.coordinates.dat.data[:, 0] -= 0.0 
mesh_ref.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.450 km

element = spyro.domains.space.FE_method(mesh_ref, model["opts"]["method"], model["opts"]["degree"])
V_ref = FunctionSpace(mesh_ref, element)

element_DG = spyro.domains.space.FE_method(mesh_ref, "DG", 2) # here, it could be 2 too
V_DG = FunctionSpace(mesh_ref, element_DG)

sources = spyro.Sources(model, mesh_ref, V_ref, comm)
receivers = spyro.Receivers(model, mesh_ref, V_ref, comm)
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

vp_ref = _make_vp(V_DG, vp_guess=False)
#sys.exit("exit")
print("Starting forward computation of the exact model",flush=True) 
start = time.time()
p_ref, p_ref_recv = spyro.solvers.forward(
    model, mesh_ref, comm, vp_ref, sources, wavelet, receivers, output=True
)
end = time.time()
print(round(end - start,2),flush=True)
File("p_ref.pvd").write(p_ref[-1])


pr_ref = []
nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
rn = 0
for ti in range(nt):
    pr_ref.append(p_ref_recv[ti][rn])
plt.title("p at receiver (z=-0.5 km)")
plt.plot(pr_ref,label='p ref')
plt.legend()
plt.savefig('/home/santos/Desktop/p_recv_1.png')
plt.close()

pr_ref = []
nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
rn = 1
for ti in range(nt):
    pr_ref.append(p_ref_recv[ti][rn])
plt.title("p at receiver (z=-1.99 km)")
plt.plot(pr_ref,label='p ref')
plt.legend()
plt.savefig('/home/santos/Desktop/p_recv_2.png')
plt.close()

#spyro.io.save_shots(model, comm, p_ref_recv, file_name="./shots/test_p_ref_recv") # only CPU0 saves the results
#spyro.io.save_shots(model, comm, p_ref_recv, file_name=None) # only CPU0 saves the results
#sys.exit("exit")

J_total = np.zeros((1)) 

#print(p_ref_recv,flush=True)
p_ref_recv_loaded = spyro.io.load_shots(model, comm, file_name="./shots/test_p_ref_recv") 
#p_ref_recv_loaded = spyro.io.load_shots(model, comm, file_name=None) 
#print(p_ref_recv_loaded,flush=True)
misfit = spyro.utils.evaluate_misfit(model, p_ref_recv_loaded, p_ref_recv) # ds_exact[:ll] - guess
J_total[0] += spyro.utils.compute_functional(model, misfit) # J += residual[ti][rn] ** 2 (and J *= 0.5)
J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
J_total[0] /= comm.ensemble_comm.size # ensemble parallelism (sources)
if comm.comm.size > 1: 
    J_total[0] /= comm.comm.size # spatial parallelism
print(J_total[0], flush=True)


