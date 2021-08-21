from firedrake import File
import spyro
import sys

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
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(0.0, 0.0)],
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 10,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 500),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.00,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)
vp = spyro.io.interpolate(model, mesh, V, guess=False)
if comm.ensemble_comm.rank == 0:
    #File("density.pvd", comm=comm.comm).write(vp)
    #File("p-wave_velocity.pvd", comm=comm.comm).write(vp)
    File("s-wave_velocity.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)

sys.exit("Exit without running")

p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
spyro.io.save_shots(model, comm, p_r)
