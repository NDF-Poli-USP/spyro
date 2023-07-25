[![DOI](https://zenodo.org/badge/318542339.svg)](https://zenodo.org/badge/latestdoi/318542339)
[![Python tests](https://github.com/NDF-Poli-USP/spyro/actions/workflows/python-tests.yml/badge.svg)](https://github.com/NDF-Poli-USP/spyro/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/NDF-Poli-USP/spyro/branch/main/graph/badge.svg?token=8NM4N4N7YW)](https://codecov.io/gh/NDF-Poli-USP/spyro)

spyro: Acoustic wave modeling in Firedrake
============================================

spyro is a Python library for modeling acoustic waves. The main
functionality is a set of forward and adjoint wave propagators for solving the acoustic wave equation in the time domain.
These wave propagators can be used to form complete full waveform inversion (FWI) applications. See the [demos](https://github.com/krober10nd/spyro/tree/main/demos).
To implement these solvers, spyro uses the finite element package [Firedrake](https://www.firedrakeproject.org/index.html).

To use Spyro, you'll need to have some knowledge of Python and some basic concepts in inverse modeling relevant to active-sourcce seismology.

Discussions about development take place on our Slack channel. Everyone is invited to join using the link: https://join.slack.com/t/spyroworkspace/shared_invite/zt-u87ih28m-2h9JobfkdArs4ku3a1wLLQ

If you want to know more or cite our code please see our open access publication: https://gmd.copernicus.org/articles/15/8639/2022/gmd-15-8639-2022.html

Functionality
=============

* Finite element discretizations for scalar wave equation in 2D and 3D using triangular and tetrahedral meshes.
    * Continuous Galerkin with arbitrary spatial order and stable and accurate higher-order mass lumping up to p = 5.
* Spatial and ensemble (*shot*) parallelism for source simulations.
* Central explicit scheme (2nd order accurate) in time.
* Perfectly Matched Layer (PML) to absorb reflected waves in both 2D and 3D.
* Mesh-independent functional gradient using the optimize-then-discretize approach.
* Sparse interpolation and injection with point sources or force sources. 


Performance
===========

The performance of the `forward.py` wave propagator was assessed in the following benchmark 2D triangular (a) and 3D tetrahedral meshes (b), where the ideal strong scaling line for each KMV element is represented as dashed and the number of degrees of freedom per core is annotated. For the 2D benchmark, the domain spans a physical space of 110 km by 85 km. A domain of 8 km by 8 km by 8 km was used in the 3D case. Both had a 0.287 km wide PML included on all sides of the domain except the free surface and a uniform velocity of 1.43 km/s (see the folder benchmarks).

![scaling2dand3d](https://user-images.githubusercontent.com/45005909/127859352-f9fac860-c9db-4585-8416-45b7fa002eed.png)

As one can see, higher-order mass lumping yields excellent strong scaling on Intel Xeon processors for a moderate sized 3D problem. The usage of higher-order elements benefits both the adjoint and gradient calculation in addition to the forward calculation, which makes it possible to perform FWI with simplex elements.


A worked example
=================

A first example of a forward simulation in 2D on a rectangle with a uniform triangular mesh and using the Perfectly Matched Layer is shown in the following below. Note here we first specify the input file and build a uniform mesh using the meshing capabilities provided by Firedrake. However, more complex (i.e., non-structured) triangular meshes for realistic problems can be generated via [SeismicMesh](https://github.com/krober10nd/SeismicMesh).


See the demos folder for an FWI example (this requires some other dependencies pyrol and ROLtrilinos).



![Above shows the simulation at two timesteps in ParaView that results from running the code below](https://user-images.githubusercontent.com/18619644/94087976-7e81df00-fde5-11ea-96c0-474348286091.png)

```python
import spyro

dictionary = {}

# Choose spatial discretization method and parameters
dictionary["options"] = {
    # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "cell_type": "T",  
    # lumped, equispaced or DG, default is lumped "method":"MLT",
    # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
    # You can either specify a cell_type+variant or a method.
    "variant": 'lumped',  
    # Polynomial order of the spatial discretion's basis functions.
    # For MLT we recomend 4th order in 2D, 3rd order in 3D, for SEM 4th or 8th.
    "degree": 4,  
    # Dimension (2 or 3)
    "dimension": 2,  
}

# Number of cores for the shot. For simplicity, we keep things automatic.
# spyro supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    # options: automatic (same number of cores for every shot) or spatial
    "type": "automatic",
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
dictionary["mesh"] = {
    # depth in km - always positive
    "Lz": 0.75,
    # width in km - always positive
    "Lx": 1.50,
    # thickness in km - always positive
    "Ly": 0.0,
    # If we are loading and external .msh mesh file
    "mesh_file": None,
    # options: None (default), firedrake_mesh, user_mesh, or SeismicMesh
    # use this opion if your are not loading an external file
    # 'firedrake_mesh' will create an automatic mesh using firedrake's built-in meshing tools
    # 'user_mesh' gives the option to load other user generated meshes from unsuported formats
    # 'SeismicMesh' automatically creates a waveform adapted unstructured mesh to reduce total
    # DoFs using the SeismicMesh tool.
    "mesh_type": "firedrake_mesh",
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.3, 0.75)],
    "frequency": 8.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect(
        (-0.5, 0.1), (-0.5, 1.4), 100
    ),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    # Initial time for event
    "initial_time": 0.0,
    # Final time for event
    "final_time": 0.50,
    # timestep size
    "dt": 0.0001,
    # the Ricker has an amplitude of 1.
    "amplitude": 1,
    # how frequently to output solution to pvds
    "output_frequency": 100,
    # how frequently to save solution to RAM
    "gradient_sampling_frequency": 100,
}

dictionary["visualization"] = {
    "forward_output" : True,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}


# Create an AcousticWave object with the above dictionary.
Wave_obj = spyro.AcousticWave(dictionary=dictionary)

# Defines the element size in the automatically generated firedrake mesh.
Wave_obj.set_mesh(dx=0.01)


# Manually create a simple two layer seismic velocity model.
# Note: the user can specify their own velocity model in a HDF5 or SEG-Y file format.
# The HDF5 file has to contain an array with
# the velocity data and it is linearly interpolated onto the mesh nodes at run-time.
z = Wave_obj.mesh_z
import firedrake as fire
velocity_conditional = fire.conditional(z > -0.35, 1.5, 3.0)
Wave_obj.set_initial_velocity_model(conditional=velocity_conditional, output=True)

# And now we simulate the shot using a 2nd order central time-stepping scheme
# Note: simulation results are stored in the folder `~/results/` by default
Wave_obj.forward_solve()

# Visualize the shot record
spyro.plots.plot_shots(Wave_obj, show=True)

# Save the shot (a Numpy array) as a pickle for other use.
spyro.io.save_shots(Wave_obj)

# can be loaded back via
my_shot = spyro.io.load_shots(Wave_obj)
```

### Testing

To run the spyro unit tests (and turn off plots), check out this repository and type
```
MPLBACKEND=Agg pytest --maxfail=1
```


### License

This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html)
