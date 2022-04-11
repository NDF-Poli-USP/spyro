from firedrake import *
from scipy.optimize import *
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import meshio
import SeismicMesh
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
#from ..domains import quadrature, space

model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG", # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 0.8,  # depth in km - always positive
    "Lx": 0.8,  # width in km - always positive
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
    "source_pos": [(0.4, 0.4)],
    "frequency": 10.0,
    "delay": 0.0,
    "num_receivers": 1,
    "receiver_locations": spyro.create_transect(
       (0.5, 0.4), (0.5, 0.4), 1
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.001*400,  # Final time for event (for test 7)
    "dt": 0.0010,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}
model["Aut_Dif"] = {
    "status": False, 
}

comm = spyro.utils.mpi_init(model)

mesh = RectangleMesh(50, 50, model["mesh"]["Lz"], model["mesh"]["Lx"], diagonal="crossed") 

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)

V = FunctionSpace(mesh, element)

lamb_exact = Function(V).interpolate(Constant(1./2))  # exact
mu_exact   = Function(V).interpolate(Constant(1./4.))

lamb_guess = Function(V).interpolate(Constant(3.*1./2.)) # guess
mu_guess   = Function(V).interpolate(Constant(1.5*1./4.))

rho = Constant(1.) 

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
point_cloud = receivers.setPointCloudRec(comm, paralel_z=True)

wavelet = spyro.full_ricker_wavelet(
    model["timeaxis"]["dt"],
    model["timeaxis"]["tf"],
    model["acquisition"]["frequency"],
    amp=model["timeaxis"]["amplitude"]
)

forward_elastic_waves_adj = spyro.solvers.forward_elastic_waves
forward_elastic_waves_AD = spyro.solvers.forward_elastic_waves_AD

print('######## Running the exact model - AD ########')
start = time.time()
u_exact_AD, uz_exact_recv_AD, ux_exact_recv_AD, uy_exact_recv_AD = forward_elastic_waves_AD(
    model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, point_cloud, output=False
)
end = time.time()
print("simulation time="+str(round(end - start,2)),flush=True)

print('######## Running the exact model - adj ########')
start = time.time()
use_AD_type_interp=False  # if True, it uses the same linear solver definition employed in the AD version
if use_AD_type_interp==True:
    receivers=point_cloud

u_exact_adj, uz_exact_recv_adj, ux_exact_recv_adj, uy_exact_recv_adj = forward_elastic_waves_adj(
    model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, receivers, output=False, 
    use_AD_type_interp=use_AD_type_interp
)
end = time.time()
print("simulation time="+str(round(end - start,2)),flush=True)

# print u_adj and u_AD
File("u_AD.pvd").write(u_exact_AD[-1])
File("u_adj.pvd").write(u_exact_adj[-1])

V2 = VectorFunctionSpace(mesh, element)
du = Function(V2,name="Delta Displacement").assign( u_exact_adj[-1]-u_exact_AD[-1] )
File("du.pvd").write(du)

if True:
    u_AD =[] 
    u_adj=[]
    du=[]
    du_rel=[]
    nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
    rn = 0
    for ti in range(nt):
        u_AD.append(uz_exact_recv_AD[ti][rn])
        u_adj.append(uz_exact_recv_adj[ti][rn])
        du.append(uz_exact_recv_adj[ti][rn]-uz_exact_recv_AD[ti][rn])
        du_rel.append( abs( (uz_exact_recv_adj[ti][rn]-uz_exact_recv_AD[ti][rn])/(uz_exact_recv_adj[ti][rn]+1.e-12) ) )

    plt.title("u")
    plt.plot(u_AD,label='u (AD)')
    plt.plot(u_adj,label='u (adj)',linestyle='--')
    plt.legend() 
    plt.savefig('/home/santos/Desktop/u.png')
    plt.close()

    plt.title("du")
    plt.plot(du,label='u (adj) - u (AD)')
    plt.legend() 
    plt.savefig('/home/santos/Desktop/du.png')
    plt.close()

    plt.title("du (rel)")
    plt.plot(du_rel,label='(u (adj) - u (AD))/u (adj)')
    plt.legend() 
    plt.savefig('/home/santos/Desktop/du_rel.png')
    plt.close()

sys.exit("sys.exit called")


