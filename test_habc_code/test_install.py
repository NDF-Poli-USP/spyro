import firedrake as fire
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
# SPIRO supports both spatial parallelism and "shot" parallelism.
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
    "forward_output": True,
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
velocity_conditional = fire.conditional(z > -0.35, 1.5, 3.0)
Wave_obj.set_initial_velocity_model(
    conditional=velocity_conditional, output=True)

# And now we simulate the shot using a 2nd order central time-stepping scheme
# Note: simulation results are stored in the folder `~/results/` by default
Wave_obj.forward_solve()

# Visualize the shot record
spyro.plots.plot_shots(Wave_obj, show=True)

# Save the shot (a Numpy array) as a pickle for other use.
spyro.io.save_shots(Wave_obj)

# can be loaded back via
my_shot = spyro.io.load_shots(Wave_obj)
