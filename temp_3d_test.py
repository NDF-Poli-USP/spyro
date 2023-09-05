import numpy as np
import matplotlib.pyplot as plt
import time
import spyro


dt = 0.0005
final_time = 0.5
offset = 0.1

dictionary = {}
dictionary["options"] = {
    # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "cell_type": "Q",
    # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "variant": 'lumped',
    # p order
    "degree": 4,
    # dimension
    "dimension": 3,
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    # options: automatic (same number of cores for evey processor) or spatial
    "type": "automatic",
}

# Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    # depth in km - always positive
    "Lz": 0.8,
    # width in km - always positive
    "Lx": 0.8,
    # thickness in km - always positive
    "Ly": 0.8,
    "mesh_file": None,
    # options: firedrake_mesh or user_mesh
    "mesh_type": "firedrake_mesh",
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.4, 0.4, 0.4)],
    "frequency": 5.0,
    "delay": 1.5,
    "receiver_locations": [(-0.4 - offset, 0.4, 0.4)],
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": dt,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_3d_output3by1by1.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

Wave_obj = spyro.AcousticWave(dictionary=dictionary)
Wave_obj.set_mesh(dx=0.02, periodic=True)

Wave_obj.set_initial_velocity_model(constant=1.5)
t0 = time.time()
Wave_obj.forward_solve()

time_vector = np.linspace(0.0, final_time, int(final_time/dt)+1)

rec_out = Wave_obj.receivers_output
t1 = time.time()
print(f"Total runtime :{t1-t0}", flush=True)

plt.plot(time_vector, Wave_obj.receivers_output)
plt.show()

print("END")
