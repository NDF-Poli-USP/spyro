from firedrake import File
from firedrake import Function
import spyro
import sys
import time

model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "Equi",  # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "automatic",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/marmousi_elastic.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5",
}
    #"meshfile": "meshes/marmousi_elastic.msh",
    #"meshfile": "meshes/marmousi_elastic_with_water_layer.msh",
    #"truemodel": "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5",
    #"truemodel": "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5",
    #"truemodel": "velocity_models/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.hdf5",
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
    "source_pos": [(-0.45, 5.0)], # Z and X
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 10,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 500),
}
    #"source_pos": [(-0.45, 5.0)], # Z and X
    #"source_pos": [(0.0, 5.0)], # Z and X
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.5,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

# interpolate Vs, Vp, and Density onto the mesh FIXME check the units or create a new interpolation method
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5" 
vs = spyro.io.interpolate(model, mesh, V, guess=False)
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5" 
vp = spyro.io.interpolate(model, mesh, V, guess=False)
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.hdf5" 
rho = spyro.io.interpolate(model, mesh, V, guess=False)

vs.dat.data[:] = vs.dat.data[:] / 1000. # only vs needs unit recast for now
# vs and vp in km/s
# rho in 1000 x Gt/km3

mu = Function(V, name="mu").interpolate(rho * vs ** 2.)
lamb = Function(V, name="lamb").interpolate(rho * (vp ** 2. - 2. * vs ** 2.))

write_files=1
if comm.ensemble_comm.rank == 0 and write_files==1:
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

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)

#sys.exit("Exit without running")
start = time.time()
p, p_r = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=True
)
end = time.time()
print(end - start)

sys.exit("Exit without plotting shots")

spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
spyro.io.save_shots(model, comm, p_r)
