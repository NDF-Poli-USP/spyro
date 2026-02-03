import spyro
import numpy as np

dictionary = {}

dictionary["options"] = {
    "cell_type": "Q",
    "variant": "lumped",
    "degree": 2,
    "dimension": 2,
    }

dictionary["parallelism"] = {
    "type": "automatic",
    }

dictionary["mesh"] = {
    "Lx": 4.0,
    "Lz": 4.0,
    "Ly": 0.0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
    }

dictionary["acquisition"] = {
    "source_type" = "ricker",
    "source_locations": [(2.0, 2.0)],
    "frequency": 6.0,
    "delay": 1.0/6.0,
    "delay_type": "time",
    "receiver_locations": [(2.15, 2.0)],
    }

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 1.0,
    "dt": 0.001,
    "output_frequency": 10,
    "gradient_sampling_frequency": 1,
    }

Wave_obj = spyro.AcousticWaveJessica(dictionary=dictionary)
Wave_obj.set_mesh(input_mesh_parameters={"edge_lengh": 4.0/150.0, "periodic": True})
Wave_obj.set_initial_velocity_model(constant=1.5)
Wave_obj.plot.plot_model(Wave_obj, filename="model.png", flip_axis=False, show=True)
Wave_obj.forward_solve()

last_pressure = Wave_obj.u_n
last_pressure_data = last_pressure.dat.data[:]
spyro.plots.plot_function(last_pressure)

spyro.plots.plot_shots(Wave_obj, contour_lines=100,
                       vmin=-np.max(last_pressure_data),
                       vmax=np.max(last_pressure_data),
                       show=True)
