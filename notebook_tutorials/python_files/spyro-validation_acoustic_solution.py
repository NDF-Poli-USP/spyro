import spyro
import numpy as np

def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time

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

Wave_obj = spyro.AcousticWave(dictionary=dictionary)

Wave_obj.set_mesh(input_mesh_parameters={"edge_length": 4.0/150.0, "periodic": True})

Wave_obj.set_initial_velocity_model(constant=1.5)

Wave_obj.forward_solve()

t0 = dictionary["time_axis"]["initial_time"]
tf = dictionary["time_axis"]["final_time"]
dt = dictionary["time_axis"]["dt"]
time_points = np.arange(t0, tf + dt, dt)

numerical_solution = Wave_obj.receivers_output.flatten()

analytical_solution = spyro.utils.nodal_homogeneous_analytical(
        Wave_obj, 0.15, 1.5, n_extra=100
        )

error = error_calc(numerical_solution, analytical_solution, len(analytical_solution))
print(f"Error = {error * 100:.2f}%")

spyro.plots.plot_validation_acoustic(time_points, analytical_solution, numerical_solution)
