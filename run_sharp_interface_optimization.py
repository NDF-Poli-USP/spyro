import spyro

model = {}
model["opts"] = {
    "method": "KMV",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "quadrature": "KMV",
}
model["mesh"] = {
    "Lz": 0.65,  # depth in km - always positive
    "Lx": 1.00,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/immersed_disk_guess_vp.msh",
    "initmodel": "velocity_models/immersed_disk_guess_vp.hdf5",
    "truemodel": "velocity_models/immersed_disk_true_vp.hdf5",
}
model["PML"] = {
    "status": True,  # true,  # true or false
    "outer_bc": "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 5.0,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.50,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.50,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}
recvs = spyro.create_transect((-0.01, 0.01), (-0.01, 0.99), 100)
sources = spyro.create_transect((-0.01, 0.01), (-0.01, 0.99), 4)
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,  # len(sources),
    "source_pos": [(-0.01, 0.5)],  # sources,
    "frequency": 5.0,
    "delay": 1.0,
    "amplitude": 1.0,
    "num_receivers": len(recvs),
    "receiver_locations": recvs,
}
model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 1.0,  # final time for event
    "dt": 0.0005,  # timestep size
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to ram
    "skip": 1,
}
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
}
model["optimization"] = {
    "exact_shot_prefix": "shots/forward_exact_level_set",
    "beta0": 1.5,
    "max_ls": 10,
    "gamma": 0.8,
    "advect_timesteps": 10,
    "max_iter": 100,
}
#### end of options ####


comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp = spyro.io.interpolate(model, mesh, V, guess=True)

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

vp = spyro.optimization(model, mesh, V, comm, vp, sources, receivers)
