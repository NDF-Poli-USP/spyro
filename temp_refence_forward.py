import numpy as np
import matplotlib.pyplot as plt

import spyro


def error_calc(receivers, analytical, dt):
    rec_len, num_rec = np.shape(receivers)

    # Interpolate analytical solution into numerical dts
    final_time = dt * (rec_len - 1)
    time_vector_rec = np.linspace(0.0, final_time, rec_len)
    time_vector_ana = np.linspace(0.0, final_time, len(analytical[:, 0]))
    ana = np.zeros(np.shape(receivers))
    for i in range(num_rec):
        ana[:, i] = np.interp(
            time_vector_rec, time_vector_ana, analytical[:, i]
        )

    total_numerator = 0.0
    total_denumenator = 0.0
    for i in range(num_rec):
        diff = receivers[:, i] - ana[:, i]
        diff_squared = np.power(diff, 2)
        numerator = np.trapz(diff_squared, dx=dt)
        ref_squared = np.power(ana[:, i], 2)
        denominator = np.trapz(ref_squared, dx=dt)
        total_numerator += numerator
        total_denumenator += denominator

    squared_error = total_numerator / total_denumenator

    error = np.sqrt(squared_error)
    return error


final_time = 7.5
dt = 0.0005
degree = 4
cpw = 2.67
# Source and receiver calculations
source_z = -0.1
source_x = 3.0
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
    "method": "mass_lumped_triangle",
    "degree": degree,
    "dimension": 2,
    "automatic_adjoint": False,
}
dictionary["parallelism"] = {
    "type": "automatic",
}
dictionary["mesh"] = {
    "Lz": 3.5,
    "Lx": 17.0,
    "Ly": 0.0,
    # "cells_per_wavelength": cpw,
    # "mesh_type": "SeismicMesh",
    "mesh_file": "test2p5.msh",
}
dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "PML",
    "exponent": 2,
    "cmax": 4.5,
    "R": 1e-6,
    "pad_length": 0.3,
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": source_locations,
    "frequency": 5.0,
    "receiver_locations": receiver_locations,
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": dt,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 500,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/2p5temp_forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
    "debug_output": True,
}
dictionary["synthetic_data"] = {
    "real_velocity_file": "/home/olender/common_files/velocity_models/vp_marmousi-ii.segy",
}
spyro.io.saving_source_and_receiver_location_in_csv(dictionary)
Wave_obj = spyro.AcousticWave(dictionary)
# Wave_obj.set_mesh(mesh_parameters={"cells_per_wavelength": cpw})
Wave_obj.forward_solve()
p_receivers = Wave_obj.forward_solution_receivers
np.save("test2p67.npy", p_receivers)
print("END")
