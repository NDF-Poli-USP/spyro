import spyro
import firedrake as fire


def make_minas_cheese_conditional(mesh_z, mesh_x):
    outside_vp = 1.5
    circle_vp = 2.0
    square_vp = 3.0
    r_c = 0.5
    center_z = -1.0
    center_x = 1.0
    square_top_z = -0.9
    square_bot_z = -1.1
    square_left_x = 0.9
    square_right_x = 1.1
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < r_c**2, circle_vp, outside_vp)
    cond = fire.conditional(
        fire.And(
            fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
            fire.And(mesh_x > square_left_x, mesh_x < square_right_x)
        ),
        square_vp,
        cond,
    )
    return cond

frequency = 5.0
dictionary = {}

# Finite element options - using triangular elements with lumped mass matrix
dictionary["options"] = {
    "cell_type": "T",  # Triangular elements (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # Polynomial order (higher degree = better accuracy)
    "dimension": 2,  # 2D problem
}
dictionary["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
    "shot_ids_per_propagation": [[0], [1], [2], [3], [4]],
}
dictionary["mesh"] = {
    "length_z": 2.0,  # Depth in km (always positive)
    "length_x": 2.0,  # Width in km (always positive)
    "length_y": 0.0,  # Thickness in km (0 for 2D)
    "mesh_type": "user_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",  # Ricker wavelet source
    "source_locations": spyro.create_transect((-0.35, 0.5), (-0.35, 1.5), 4),  # Eigth sources
    "frequency": frequency,  # Dominant frequency in Hz
    "delay": 1.0/frequency,  # Source delay (1 period)
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.65, 0.5), (-1.65, 1.5), 200),  # 200 receivers
}
dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "local",  # Damping in the boundaries
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Start time
    "final_time": 1.3,  # End time (must be long enough for waves to propagate)
    "dt": 0.0001,  # Time step (small for stability), but can be a lot larger. WHy don't you try increasing it?
    "amplitude": 1,  # Source amplitude
    "output_frequency": 100,  # Save solution every 100 time steps for visualization
    "gradient_sampling_frequency": 1,  # Save every time step for gradient computation (How high can we increase this? Look at nyquist frequency)
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}

# Let us first create a velocity grid based on the minas cheese model
wave_obj = spyro.AcousticWave(dictionary=dictionary)
# First, create a simple mesh to generate a velocity grid
grid_spacing = 0.02
wave_obj.set_mesh(input_mesh_parameters={
    "mesh_type": "firedrake_mesh", 
    "edge_length": grid_spacing,
})
mesh_z = wave_obj.mesh_z  # Depth coordinate
mesh_x = wave_obj.mesh_x  # Horizontal coordinate
wave_obj.set_initial_velocity_model(conditional=make_minas_cheese_conditional(mesh_z, mesh_x))
# Creading a grid velocity data dictionary, this can be used to7
# create a wave-adapted mesh later on and is similar to the data
# read from a segy file
z = spyro.io.write_function_to_grid(wave_obj.initial_velocity_model, wave_obj.function_space, grid_spacing, buffer=False)
grid_velocity_data = {
    "vp_values": z,
    "grid_spacing": grid_spacing,
}

print("END")
