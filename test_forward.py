import spyro

dt = 0.0001

final_time = 1.4

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_type": "SeismicMesh",  # options: firedrake_mesh or user_mesh
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.2, 1.75)],
    "frequency": 5.0,
    "receiver_locations": spyro.create_transect(
        (-0.15, 1.0), (-0.15, 16.0), 50
    ),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": dt,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 200,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 200,  # how frequently to save solution to RAM
}

dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "PML",
    "exponent": 2,
    "cmax": 4.5,
    "R": 1e-6,
    "pad_length": 0.25,
}

dictionary["synthetic_data"] = {
    "real_velocity_file": "/media/alexandre/Extreme SSD/common_files/velocity_models/vp_marmousi-ii.segy"
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/new_extended_pml_propagation.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
    "debug_output": True,
}

Wave_obj = spyro.AcousticWave(dictionary)
Wave_obj.set_mesh()
Wave_obj.forward_solve()

print("END")
