import spyro
import math
from generate_velocity_model_from_paper import get_paper_velocity

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
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
    "Lz": 2.4,  # depth in km - always positive
    "Lx": 4.8,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "user_mesh": None,
    "mesh_type": "SeismicMesh",
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.6, 4.8-1.68)],
    "frequency": 5.0,
    "delay": 1.5,
    "receiver_locations": [(-0.6, 4.8-1.68)],
}

# Simulate for 1.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.00,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/figeigteen_forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
    "debug_output": True,
}

Wave_obj = spyro.HABC(dictionary=dictionary)

cpw = 6.0
lba = 1.5 / 5.0
edge_length = lba / cpw
Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})
V = Wave_obj.function_space
mesh = Wave_obj.mesh
c = get_paper_velocity(mesh, V)

Wave_obj.set_initial_velocity_model(velocity_model_function=c)
Wave_obj._get_initial_velocity_model()

Wave_obj.c = Wave_obj.initial_velocity_model
# Wave_obj.get_and_set_maximum_dt(fraction=0.5)
Wave_obj.no_boundary_forward_solve()
Wave_obj.set_damping_field()
