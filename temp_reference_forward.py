import numpy as np
import spyro

final_time = 6.0
dt = 0.0001

# Source and receiver calculations
source_z = -0.02
source_x = 1.0
source_locations = [(source_z, source_x)]

# Receiver calculations
receiver_bin_center1 = 2000.0/1000
receiver_bin_center2 = 10000.0/1000
receiver_quantity = 500

bin1_startZ = source_z
bin1_endZ = source_z
bin1_startX = source_x + receiver_bin_center1
bin1_endX = source_x + receiver_bin_center2

receiver_locations = spyro.create_transect(
    (bin1_startZ, bin1_startX),
    (bin1_endZ, bin1_endX),
    receiver_quantity
)

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
    "mesh_file": None,
    "mesh_type": "SeismicMesh_mesh",  # options: firedrake_mesh or user_mesh
    "cells_per_wavelength": 5.0,
}
dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "PML",
    "exponent": 2,
    "cmax": 4.5,
    "R": 1e-6,
    "pad_length": 0.3,
}
# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": source_locations,
    "frequency": 5.0,
    "receiver_locations": receiver_locations,
}
# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": dt,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 1000,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/temp_propagation.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}
dictionary["synthetic_data"] = {
    "real_velocity_file": "/media/alexandre/Extreme SSD/common_files/velocity_models/vp_marmousi-ii.segy",
}

Wave_obj = spyro.AcousticWave(dictionary=dictionary)
Wave_obj.forward_solve()
