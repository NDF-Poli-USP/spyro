# Marmousi model used to generate results for the STMI presentation (8/March/2022)
# Run just to compare the processing time of the forward model
# Elastic model
from firedrake import *
import spyro
import sys
import time

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    #"type": "automatic",
    "type": "spatial",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/marmousi_elastic_with_water_layer_adapted.msh",
    #"meshfile": "meshes/marmousi_elastic_with_water_layer_adapted_refined.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "gaussian-taper",  # none, gaussian-taper, or alid
    "lz": 0.9,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.10, 5.0)], # Z and X # with water layer (for wave propagation comparison)
    #"source_pos": [(-0.50, 5.0)], # Z and X # with water layer (maybe for FWI)
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.05, 0.0), (-0.05, 17.0), 100),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 1.6,  # Final time for event (used to reach the bottom of the domain)
    "tf": 0.100,  # Final time for event (used to measure the time)
    #"dt": 0.00025, # default
    "dt": 0.0001, # needs for P=5
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

# interpolate Vs, Vp, and Density onto the mesh FIXME check the units or create a new interpolation method
#model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s 
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.smoothed.segy.hdf5"# m/s 
vs = spyro.io.interpolate(model, mesh, V, guess=False)
#model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed.segy.hdf5"# m/s
vp = spyro.io.interpolate(model, mesh, V, guess=False)
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.hdf5"       # g/cm3
rho = spyro.io.interpolate(model, mesh, V, guess=False)

vs.dat.data[:] = vs.dat.data[:] / 1000. # only vs needs unit recast for now
# vs and vp in km/s
# rho in 1000 x Gt/km3

mu = Function(V, name="mu").interpolate(rho * vs ** 2.)
lamb = Function(V, name="lamb").interpolate(rho * (vp ** 2. - 2. * vs ** 2.))

write_files=0
if comm.ensemble_comm.rank == 0 and write_files==1: #{{{
    rho.rename("rho")
    vp.rename("p-wave vel")
    vs.rename("s-wave vel")
    lamb.rename("lambda")
    mu.rename("mu")
    File("density.pvd", comm=comm.comm).write(rho)
    File("p-wave_velocity.pvd", comm=comm.comm).write(vp)
    File("s-wave_velocity.pvd", comm=comm.comm).write(vs)
    File("lambda.pvd", comm=comm.comm).write(lamb)
    File("mu.pvd", comm=comm.comm).write(mu)
    sys.exit("Exit without running")
#}}}

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)

if 0:
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V2 = VectorFunctionSpace(mesh, element)
    print("DOF: "+str(V2.dof_count))
    sys.exit("Exit without running")

print("Starting forward computation",flush=True)
start = time.time()
u, uz_rec, ux_rec, uy_rec = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=False
)
end = time.time()
print(round(end - start,2), flush=True)

sys.exit("Exit after running")
