# Marmousi model used to generate results for the STMI presentation (8/March/2022)
# Run just to compare the processing time of the forward model
# Acoustic model
from firedrake import File
from firedrake import Function
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
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s
vp = spyro.io.interpolate(model, mesh, V, guess=False)

write_files=0
if comm.ensemble_comm.rank == 0 and write_files==1: #{{{
    vp.rename("p-wave vel (acoustic)")
    File("p-wave_velocity_acoustiuc.pvd", comm=comm.comm).write(vp)
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
    print("DOF: "+str(V.dof_count))
    sys.exit("Exit without running")

print("Starting forward computation",flush=True)
start = time.time()
p, p_rec = spyro.solvers.forward(
    model, mesh, comm, vp, sources, wavelet, receivers, output=True # FIXME keep false to take the processing time
)
end = time.time()
print(round(end - start,2), flush=True)

