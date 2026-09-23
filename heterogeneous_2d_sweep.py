import matplotlib.pyplot as plt
from matplotlib import use
import numpy as np
import spyro

from spyro.tools.error_measure import MeasureError
from generate_velocity_models import get_velocities
use("agg")


frequency = 10.0

source_locations = [
    (-0.5, 5.0)
]

# receiver_locations = [
#     (-0.5, x)
#     for x in np.arange(0.5, 9.51, 0.1)
# ]

receiver_locations = [
    (-3.0, x)
    for x in np.arange(0.5, 9.51, 0.1)
]

final_time = 1.0
dt = 1.0e-4

dictionary = {
    "options": {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 4,
        "dimension": 2,
    },
    "parallelism": {
        "type": "automatic",
    },
    "mesh": {
        "length_z": 5.0,
        "length_x": 10.0,
        "mesh_type": "file",
        "mesh_file": "example_mesh_2D.msh",
    },
    "acquisition": {
        "source_type": "ricker",
        "source_locations": source_locations,
        "frequency": frequency,
        "delay": 1/frequency,
        "delay_type": "time",
        "receiver_locations": receiver_locations,
        "amplitude": np.array([0.0, 1.0]),
        "use_vertex_only_mesh": True,
    },
    # "absorving_boundary_conditions": {
    #     "status": True,
    #     "abc_type": "nrbc",
    #     "nrbc": {"type": "Stacey", "dt_scheme": "backward"},
    # },
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
    "density": 1.0,
    "p_wave_velocity": 10000000.0,
    "s_wave_velocity": 10000000.0,
    "real_velocity_file": None,
}
wave = spyro.IsotropicWave(dictionary)
wave.scalar_function_space = spyro.domains.space.create_function_space(
        wave.mesh, wave.method, wave.degree, dim=1,
    )
vp, vs, rho = get_velocities(wave)


def get_numerical_result(wave):
    # wave.set_mesh(
    #     input_mesh_parameters={
    #         "mesh_file": "example_mesh_2D.msh",
    #     }
    # )
    wave._initialize_model_parameters()
    wave.rho = rho
    wave.c_s = vs
    wave.c = vp
    wave.mu = wave.rho*wave.c_s**2
    wave.lmbda = wave.rho*wave.c**2 - 2*wave.mu
    wave.forward_solve()
    return wave.forward_solution_receivers


numerical_result = get_numerical_result(wave)

plt.close()
rec_id = 10
time_vector = spyro.utils.get_time_vector(wave)
plt.plot(time_vector, wave.forward_solution_receivers[:, rec_id, 1], label='numerical')
plt.legend()
plt.savefig("debug.png")

print("END")
