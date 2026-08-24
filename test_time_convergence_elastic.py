import spyro
import numpy as np

source_z = -1.1
receiver_z = -1.9
edge_length = 0.02
receiver_locations = spyro.create_transect((receiver_z, 1.2), (receiver_z, 1.8), 300)
final_time = 1.5
dt = 1e-3
time_delay = 0.2
frequency = 5.0
amplitude = np.array([0.0, 1.0])

dictionary = {
    "options": {
        "cell_type": "Q",
        "variant": "lumped",
        "degree": 4,
        "dimension": 2,
    },
    "parallelism": {
        "type": "automatic",
    },
    "mesh": {
        "length_z": 3.0,
        "length_x": 3.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    },
    "acquisition": {
        "source_type": "ricker",
        "source_locations": [(source_z, 1.5)],
        "frequency": frequency,
        "delay": time_delay,
        "delay_type": "time",
        "receiver_locations": receiver_locations,
        "amplitude": amplitude,
    },
    "time_axis": {
        "initial_time": 0.0,
        "final_time": final_time,
        "dt": dt,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    },
    "visualization": {
        "forward_output": False,
        "fwi_velocity_model_output": False,
        "gradient_output": False,
        "adjoint_output": False,
        "debug_output": False,
    },
}

dictionary["synthetic_data"] = {
    "type": "object",
    "density": 0.1,
    "p_wave_velocity": 1.5,
    "s_wave_velocity": 1.0,
    "real_velocity_file": None,
}
dictionary["acquisition"]["amplitude"] = np.array([0.0, 1.0])
wave = spyro.IsotropicWave(dictionary)
wave.set_mesh(input_mesh_parameters={"edge_length": edge_length, "periodic": False})

spyro.utils.analytical_solution_elastic(
    "force_source",
    wave.source_locations[0] - wave.receiver_locations[0],
    p_wave_velocity=1.5,
    s_wave_velocity=1.0,
    density=0.1,
    amplitude=1.0,
    force_direction=1,
    frequency=frequency,
    time_delay=time_delay,
    final_time=final_time,
    dt=dt,
    dimension=2,
)

# wave.forward_solve()

# nt = int(final_time/0.001) + 1
# time_vector = np.linspace(0.0, final_time, nt)

# fig = spyro.plots.plot_displacement_components(
#     time_vector, wave.forward_solution_receivers[:, 0], show=False, hold=True,
# )
print("END")
