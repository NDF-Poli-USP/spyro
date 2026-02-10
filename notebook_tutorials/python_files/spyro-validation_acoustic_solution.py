import spyro
import numpy as np

def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time

def scale_to_reference(p_numerical, p_analytical):
    min_len = min(len(p_numerical), len(p_analytical))
    p_num = np.asarray(p_numerical[:min_len]).flatten()
    p_analy = np.asarray(p_analytical[:min_len]).flatten()
    alpha = np.dot(p_analy, p_num) / np.dot(p_num, p_num)
    p_numerical_scaled = alpha * p_num
    return p_numerical_scaled

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

numerical_solution = Wave_obj.receivers_output

analytical_solution = spyro.utils.nodal_homogeneous_analytical(
        Wave_obj, 0.15, 1.5, n_extra=100
        )

numerical_solution_scaled = scale_to_reference(numerical_solution, analytical_solution)

error = error_calc(numerical_solution_scaled, analytical_solution, len(analytical_solution))
print(f"Error spyro/analitycal = {error * 100:.2f}%")

# Gar6more2D:
try:
    gar6_path = "P.dat"
    gar6_data = np.loadtxt(gar6_path)
    t_gar6 = gar6_data[:, 0]
    p_gar6 = gar6_data[:, 1]
    gar6_solution_scaled = scale_to_reference(p_gar6, analytical_solution)
except Exception as e:
    print(f"Error loaging Gar6: {e}")
    p_gar6_final = None
error_gar6 = error_calc(gar6_solution_scaled, analytical_solution, len(analytical_solution))
print(f"Error Gar6more2D/analytical = {error_gar6 * 100:.2f}%")

spyro.plots.plot_validation_acoustic(time_points, analytical_solution, numerical_solution_scaled, gar6_solution_scaled)




