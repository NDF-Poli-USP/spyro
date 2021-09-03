from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
    Constant,
    exp
)
import spyro
import time
import sys

model = {}

# Choose method and parameters
model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG", # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the ABL. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Absorbing Boundary Layer (ABL) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
model["mesh"] = {
    "Lz": 1.5,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "abl_bc": "gaussian-taper",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(0.0, 0.75)],
    "frequency": 8.0,
    "delay": 1.0,
    "num_receivers": 0,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 1.4), 1
    ),
}

# Simulate for 2.0 seconds.
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.0005*1000,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}

# Create a simple mesh of a rectangle ∈ [1 x 2] km with ~100 m sized elements
# and then create a function space for P=1 Continuous Galerkin FEM
mesh = RectangleMesh(50, 50, 1.5, 1.5)

# We edit the coordinates of the mesh so that it's in the (z, x) plane
# and has a domain padding of 250 m on three sides, which will be used later to show
# the Absorbing Boundary Layer (PML). More complex 2D/3D meshes can be automatically generated with
# SeismicMesh https://github.com/krober10nd/SeismicMesh
mesh.coordinates.dat.data[:, 0] -= 1.5
mesh.coordinates.dat.data[:, 1] -= 0.0

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
#x, y = SpatialCoordinate(mesh) # actually z, x
#velocity = conditional(x > -0.35, 1.5, 3.0)
#vp = Function(V, name="velocity").interpolate(velocity)
# These pvd files can be easily visualized in ParaView!
#File("simple_velocity_model.pvd").write(vp)
# FIXME rho not defined yet
#lamb = Constant(1.5/2.) # FIXME
#mu = Constant(1.5/4.)  # FIXME
lamb = Constant(1.5) # FIXME
mu = Constant(0.0)  # FIXME
rho = Constant(1.)  # FIXME

#sys.exit("exiting without running")

# Now we instantiate both the receivers and source objects.
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)

# Create a wavelet to force the simulation
#wavelet = spyro.full_ricker_wavelet(dt=0.0005, tf=2.0, freq=8.0)
wavelet = spyro.full_ricker_wavelet(
                        dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
                        )

# And now we simulate the shot using a 2nd order central time-stepping scheme
# Note: simulation results are stored in the folder `~/results/` by default
#p_field, p_at_recv = spyro.solvers.forward(
start = time.time()
p_field, p_at_recv = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=True
)
#p_field, p_at_recv = spyro.solvers.forward(
#    model, mesh, comm, lamb, sources, wavelet, receivers, output=True
#)
end = time.time()
print(end - start)

# Visualize the shot record
#spyro.plots.plot_shots(model, comm, p_at_recv)

# Save the shot (a Numpy array) as a pickle for other use.
#spyro.io.save_shots(model, comm, p_at_recv)

# can be loaded back via
#my_shot = spyro.io.load_shots(model, comm)
