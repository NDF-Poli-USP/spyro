[![CircleCI](https://img.shields.io/circleci/project/github/krober10nd/spyro/main.svg?style=flat-square)](https://circleci.com/gh/krober10nd/spyro/tree/main)
[![CodeCov](https://codecov.io/gh/krober10nd/spyro/branch/main/graph/badge.svg)](https://codecov.io/gh/krober10nd/spyro)

spyro: Acoustic wave modeling in Firedrake
============================================

spyro is a Python library for modeling acoustic waves. The main
functionality is a set of forward and discrete adjoint wave propagators for solving the acoustic wave equation in the time domain.
These wave propagators can be used to form complete Full Waveform Inversion or Reverse Time Migration applications. See the [demos](https://github.com/krober10nd/spyro/tree/main/demos).
To implement these solvers, spyro uses the finite element package [Firedrake](https://www.firedrakeproject.org/index.html).

To use Spyro, you'll need to have some knowledge of Python and some basic concepts in inverse modeling relevant to active-sourcce seismology.

Functionality
=============

* Finite Element discretizations for scalar wave equation in 2D and 3D using triangular and tetrahedral meshes.
    * Continuous Galerkin with arbitrary spatial order and stable and accurate higher-order mass lumping up to p = 5.
* Spatial and ensemble (*shot*) parallelism for source simulations.
* Leapfrog and Strong Stability Preserving Runga-Kutta (SSPRK) time-stepping schemes (up to 4th order accurate in time).
* Perfectly Matched Layer to absorb reflected waves in both 2D and 3D.
* Mesh-independent functional gradient for Leapfrog and SSPRK time-stepping methods using the discrete adjoint method.
* Sparse interpolation and injection.

Using this functionality, short Python scripts can written that perform Full Waveform Inversion (FWI) type algorithms using well-developed numerical optimization algorithms such as L-BFGS from the SciPy package. See the notebooks folder for an FWI example.

Performance
===========

Strong scaling on both Intel Xeon processors and AMD processors is quite good. This test was performed with an 11 M DoF 3D tetrahedral mesh adapted to the Overthrust3D model. A 1 second wave simulation was executed with a 750-m PML on all sides but the free surface: 

![ScalingAmdIntel](https://user-images.githubusercontent.com/18619644/111385935-41a6ee80-868a-11eb-8da3-256274bf1c0f.png)


A worked example
=================

A simple example of a forward simulation in 2D on a rectangle with a uniform triangular mesh and using the Perfectly Matched Layer is like the following below. Note here we first specify the input file and build a uniform mesh using the meshing capabilities provided by Firedrake. However, more complex meshes for realistic problems can be generated via [SeismicMesh](https://github.com/krober10nd/SeismicMesh).


See the demos folder for an FWI example (this requires some other dependencies pyrol and ROLtrilinos).



![Above shows the simulation at two timesteps in ParaView that results from running the code below](https://user-images.githubusercontent.com/18619644/94087976-7e81df00-fde5-11ea-96c0-474348286091.png)

```python
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
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
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
model["PML"] = {
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
sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

# And now we simulate the shot using a Leapfrog time-stepping scheme
# Other time-stepping options are available (see the documentation).
# Note: simulation results are stored in the folder `results/`
p_field, p_at_recv = spyro.solvers.Leapfrog(
    model, mesh, comm, vp, sources, receivers
)

# Visualize the shot record
spyro.plots.plot_shotrecords(model, p_at_recv, "example_shot", vmin=-1e-5, vmax=1e-5)

# Save the shot (a Numpy array) as a pickle for other use.
spyro.io.save_shots("example_shot.dat", p_at_recv)
```

### Testing

To run the spyro unit tests (and turn off plots), check out this repository and type
```
MPLBACKEND=Agg pytest --maxfail=1
```


### License

This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html)
