from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
)

import spyro

model = {}

# Choose method and parameters
model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV", # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
model["mesh"] = {
    "Lz": 0.75,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.1, 0.75)],
    "frequency": 8.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 1.4), 100
    ),
}

# Simulate for 2.0 seconds.
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 2.00,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}


# Create a simple mesh of a rectangle ∈ [1 x 2] km with ~100 m sized elements
# and then create a function space for P=1 Continuous Galerkin FEM
mesh = RectangleMesh(100, 200, 1.0, 2.0)

# We edit the coordinates of the mesh so that it's in the (z, x) plane
# and has a domain padding of 250 m on three sides, which will be used later to show
# the Perfectly Matched Layer (PML). More complex 2D/3D meshes can be automatically generated with
# SeismicMesh https://github.com/krober10nd/SeismicMesh
mesh.coordinates.dat.data[:, 0] -= 1.0
mesh.coordinates.dat.data[:, 1] -= 0.25


# Create the computational environment
comm = spyro.utils.mpi_init(model)

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)
V = FunctionSpace(mesh, element)

# Manually create a simple two layer seismic velocity model `vp`.
# Note: the user can specify their own velocity model in a HDF5 file format
# in the above two lines using SeismicMesh.
# If so, the HDF5 file has to contain an array with
# the velocity data and it is linearly interpolated onto the mesh nodes at run-time.
x, y = SpatialCoordinate(mesh)
velocity = conditional(x > -0.35, 1.5, 3.0)
vp = Function(V, name="velocity").interpolate(velocity)
# These pvd files can be easily visualized in ParaView!
File("simple_velocity_model.pvd").write(vp)


# Now we instantiate both the receivers and source objects.
sources = spyro.Sources(model, mesh, V, comm)

receivers = spyro.Receivers(model, mesh, V, comm)

# Create a wavelet to force the simulation
wavelet = spyro.full_ricker_wavelet(dt=0.0005, tf=2.0, freq=8.0)

# And now we simulate the shot using a 2nd order central time-stepping scheme
# Note: simulation results are stored in the folder `~/results/` by default
p_field, p_at_recv = spyro.solvers.forward(
    model, mesh, comm, vp, sources, wavelet, receivers
)

# Visualize the shot record
spyro.plots.plot_shots(model, comm, p_at_recv)

# Save the shot (a Numpy array) as a pickle for other use.
spyro.io.save_shots(model, comm, p_at_recv)

# can be loaded back via
my_shot = spyro.io.load_shots(model, comm)
