import spyro

length_z = 7760.0
length_x = 32040.0
length_y = 0.0
quadrilateral = False
mesh_type = "gmsh_mesh"
velocity_model = "tests/inputfiles/velocity_models/avenir.segy"
cells_per_wavelength = 2.6
dt = 0.0001  # if none, will use 70% of stable (which takes time to calculate)

if not quadrilateral:
    cell_type = "T"
else:
    cell_type = "Q"

dictionary = {}
dictionary["options"] = {
    "cell_type": cell_type,  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["synthetic_data"] = {
    "real_velocity_file": velocity_model,
}
# Define the domain size without the layer.
dictionary["mesh"] = {
    "length_z": length_z,  # depth in km - always positive
    "length_x": length_x,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_type": mesh_type,
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.2, length_x/2.0)],
    "frequency": 5.0,
    "delay": 0.3,
    "receiver_locations": [(-0.5, length_x/2.0)],
    "delay_type": "time",
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.5,  # Final time for event
    "dt": dt,  # timestep size
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
wave_obj.set_mesh(input_mesh_parameters={
    "cells_per_wavelength": cells_per_wavelength,
    "velocity_model": velocity_model,
})
wave_obj.forward_solve()

print("END")
