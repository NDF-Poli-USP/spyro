import spyro

cell_type = "T"  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
mesh_file = "YOUR_MESH_HERE.msh"
velocity_model_file = "YOUR_VELOCITY_MODEL_HERE"
length_z = None  # depth in km
length_x = None  # width in km

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    "length_z": length_z,  # depth in km - always positive
    "length_x": length_x,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": mesh_file,
}
dictionary["synthetic_data"] = {
    "real_velocity_file": velocity_model_file,
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.5, length_x/2.)],
    "frequency": 5.0,
    "delay": 0.3,
    "receiver_locations": [(-0.5, length_x/2. + 0.5)],
    "delay_type": "time",
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.0,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

wave_obj = spyro.AcousticWave(dictionary=dictionary)
wave_obj.forward_solve()
