from firedrake import File
import spyro
import time

# Adding model parameters:
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 4,  # p order use 2 or 3 here
    "dimension": 2,  # dimension
}

model["parallelism"] = {
    "type": "automatic",  # options: automatic, spatial, or custom.
}

model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "/home/olender/common_files/meshes/vp_marmousi_real_7p2Hz_ml4.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "/home/olender/common_files/velocity_models/vp_marmousi-ii.hdf5",
}

model["BCs"] = {
    "status": True,  # True or false
    # None or non-reflective (outer boundary condition)
    "outer_bc": "non-reflective",
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in z-direction (km)-always positive
    "lx": 0.9,  # thickness of the PML in x-direction (km)-always positive
    "ly": 0.0,  # thickness of the PML in y-direction (km)-always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 5,
    "source_pos": spyro.create_transect((-0.01, 1.0), (-0.01, 15.0), 5),
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 500,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 500),
}

# Simulate for 1.0 seconds.
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 4.0,  # Final time for event
    "dt": 0.00025,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 400,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}

comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)
vp = spyro.io.interpolate(model, mesh, V, guess=False)
if comm.ensemble_comm.rank == 0:
    File("true_velocity.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
t1 = time.time()
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
print(time.time() - t1, flush=True)

