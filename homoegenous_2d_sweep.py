import matplotlib.pyplot as plt
from matplotlib import use
import numpy as np
import spyro

from spyro.tools.error_measure import MeasureError
use("agg")
# PAramters from Lyu's comparison of sem vs D FDM: 10.1029/2023JB027576

vp = 5.6
vs = 2.75
rho = 2.0
dimension = 2

frequency = 10.0

source_z = -2.0
source_x = -source_z

receiver_z = -4.0

# Source-receiver offset bin
horizontal_offset_min = 0.0  # km
horizontal_offset_max = 0.2  # km
n_receivers = 11
receiver_xs = np.linspace(
    source_x + horizontal_offset_min, source_x + horizontal_offset_max, n_receivers,
)

source_locations = [(source_z, source_x)]

receiver_locations = [
    (receiver_z, float(receiver_x))
    for receiver_x in receiver_xs
]

final_time = 1.1
dt = 1.0e-4
periodic_mesh = True

# calculating lambda s and lambda p
lp = vp/frequency
ls = vs/frequency

dictionary = {
    "options": {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 2,
        "dimension": dimension,
    },
    "parallelism": {
        "type": "automatic",
    },
    "mesh": {
        "length_z": 6.0,
        "length_x": 6.0,
        "mesh_type": "firedrake_mesh",
        "periodic": periodic_mesh,
    },
    "acquisition": {
        "source_type": "ricker",
        "source_locations": source_locations,
        "frequency": frequency,
        "delay": 1/frequency,
        "delay_type": "time",
        "receiver_locations": receiver_locations,
        "amplitude": np.array([0.0, 1.0]),
    },
    "time_axis": {
        "initial_time": 0.0,
        "final_time": final_time,
        "dt": dt,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    },
    "visualization": {
        "forward_output": True,
        "fwi_velocity_model_output": False,
        "gradient_output": False,
        "adjoint_output": False,
        "debug_output": False,
    },
}
dictionary["synthetic_data"] = {
    "type": "object",
    "density": rho,
    "p_wave_velocity": vp,
    "s_wave_velocity": vs,
    "real_velocity_file": None,
}
wave = spyro.IsotropicWave(dictionary)

offsets = np.array(source_locations[0]) - np.array(receiver_locations[0])

def get_analytical_solution(dt, load_analytical=False):
    if load_analytical is False:
        nt = len(spyro.utils.get_time_vector(wave))
        n_rec = len(receiver_locations)
        solution_receivers = np.zeros((nt, n_rec, dimension))
        rec_id = 0
        for receiver in receiver_locations:
            offsets = np.array(source_locations[0]) - np.array(receiver)
            solution_receiver0, solution_receiver1 = spyro.utils.analytical_solution_elastic(
                source_type="force_source",
                offsets=offsets,
                p_wave_velocity=vp,
                s_wave_velocity=vs,
                density=rho,
                amplitude=1.0,
                frequency=frequency,
                time_delay=1/frequency,
                final_time=final_time,
                dt=dt,
                force_direction=1,
                dimension=2,
            )
            print(f"Calculating analytical for REC_ID{rec_id}.", flush=True)
            solution_receivers[:, rec_id, 0] = solution_receiver0
            solution_receivers[:, rec_id, 1] = solution_receiver1
            rec_id += 1

        np.save("analyticalsolution.npy", solution_receivers)
    else:
        solution_receivers = np.load("analyticalsolution.npy")

    return solution_receivers

def get_numerical_result(wave, h):
    wave.set_mesh(
        input_mesh_parameters={
            "edge_length": h,
            "periodic": periodic_mesh,
        }
    )
    wave.forward_solve()
    return wave.forward_solution_receivers

mesh_sizes = np.array([
    # 0.03928571428571429,  # ls/7.0
    # 0.04230769230769231,  # ls/6.5
    # 0.04583333333333334,  # ls/6.0
    # 0.05,  # ls/5.5
    # 0.055, # ls/5.0
    # 0.06875,  # ls/4.0
    # 0.061111, # ls/4.5
    # 0.07857142857142858,  # ls/3.5
    # 0.09166666666666667,  # ls/3.0
    # 0.09821428571428573,  # ls/2.8
    # 0.10576923076923077,  # ls/2.6
    # 0.11458333333333334,  # ls/2.4
    # 0.125,  # ls/2.2
    # 0.1375,  # ls/2.0
    # 0.1527777777777778,  # 1.8
    # 0.171875,  # 1.6
    # 0.19642857142857145, # 1.4
    # 0.22916666666666669,  # 1.2
    # 0.275, # 1.0
])

cpws = []

errors = []

load_analytical = False
for h in mesh_sizes:
    numerical_result = get_numerical_result(wave, h)
    analytical_result = get_analytical_solution(dt, load_analytical=load_analytical)
    
    rec_error = MeasureError.calculate_receiver_error(wave.forward_solution_receivers, analytical_result, dt)*100
    cpw = ls/wave.mesh_parameters.actual_h
    print(f"Receiver error of {rec_error} percent for cpw of {cpw}.")

    plt.close()
    rec_id = 10
    time_vector = spyro.utils.get_time_vector(wave)
    plt.plot(time_vector, wave.forward_solution_receivers[:, rec_id, 1], label='numerical')
    plt.plot(time_vector, analytical_result[:, rec_id, 1], "--", label='analytical')
    plt.legend()
    plt.savefig("debug.png")

    errors.append(rec_error)
    cpws.append(cpw)
    load_analytical = True

print(f"Errors of {errors}", flush=True)
errors = np.array(errors)
for cpw, error, h in zip(cpws, errors, mesh_sizes):
    print(f"cpw = {cpw:.3e}  |  error = {error:.6e}  |  h = {h:.3e}")

log_h = np.log(mesh_sizes)
log_errors = np.log(errors)

p, intercept = np.polyfit(log_h, log_errors, 1)

print(f"Observed spatial convergence order: {p:.3f}")


plt.show()