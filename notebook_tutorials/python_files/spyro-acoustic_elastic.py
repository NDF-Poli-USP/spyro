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
    "source_type": "ricker",
    "source_locations": [(-2.0, 2.0)],
    "frequency": 6.0,
    "delay": 1.0/6.0,
    "delay_type": "time",
    "receiver_locations": [(-2.15, 2.0)],
    }

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 1.0,
    "dt": 0.001,
    "output_frequency": 10,
    "gradient_sampling_frequency": 1,
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

dictionary["synthetic_data"] = {
    "type": "object",
    "density": 0.1,
    "p_wave_velocity": 1.5,
    "s_wave_velocity": 1.0,
    "real_velocity_file": None,
    }

Wave_obj = spyro.AcousticElasticWave(dictionary=dictionary)

Wave_obj.set_mesh(input_mesh_parameters={"edge_length": 4.0/150.0, "periodic": True})

Wave_obj.set_initial_velocity_model(constant=1.5)

Wave_obj.forward_solve()






