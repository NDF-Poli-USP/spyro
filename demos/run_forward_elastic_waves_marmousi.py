from firedrake import File
from firedrake import Function
import spyro
import sys
import time

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 3,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "automatic",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/marmousi_elastic_with_water_layer_adapted.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5",
}
    #"meshfile": "meshes/marmousi_elastic.msh",
    #"meshfile": "meshes/marmousi_elastic_with_water_layer.msh",
    #"meshfile": "meshes/marmousi_elastic_with_water_layer_adapted.msh",
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
    "source_pos": [(-0.10, 5.0)], # Z and X # with water layer (for wave propagation comparison)
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.05, 0.0), (-0.05, 17.0), 100),
}
    #"source_pos": [(-0.10, 7.1)], # Z and X # with water layer (for shot record comparison)
    #"source_pos": [(-0.10, 5.0)], # Z and X # with water layer (for wave propagation comparison)
    #"source_pos": [(-0.45, 5.0)], # Z and X # without water layer 
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.6,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)

# interpolate Vs, Vp, and Density onto the mesh FIXME check the units or create a new interpolation method
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s 
vs = spyro.io.interpolate(model, mesh, V, guess=False)
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s
vp = spyro.io.interpolate(model, mesh, V, guess=False)
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.hdf5"       # g/cm3
rho = spyro.io.interpolate(model, mesh, V, guess=False)

vs.dat.data[:] = vs.dat.data[:] / 1000. # only vs needs unit recast for now
# vs and vp in km/s
# rho in 1000 x Gt/km3

mu = Function(V, name="mu").interpolate(rho * vs ** 2.)
lamb = Function(V, name="lamb").interpolate(rho * (vp ** 2. - 2. * vs ** 2.))

write_files=0
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

post_process=False
if post_process==False:
    start = time.time()
    p, p_r = spyro.solvers.forward_elastic_waves(
        model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=True
    )
    end = time.time()
    print(end - start)
    spyro.plots.plot_shots(model, comm, p_r, show=True, vmin=-1.e-4, vmax=1.e-4)
    spyro.io.save_shots(model, comm, p_r, file_name="test_marmousi.dat")

else:
    p_r = spyro.io.load_shots(model, comm, file_name="test_marmousi.dat")
    spyro.plots.plot_shots(model, comm, p_r, show=True, vmin=-1.e-4, vmax=1.e-4)
