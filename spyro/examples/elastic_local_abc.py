import numpy as np
import spyro


output_dir = "results"

L = 3000  # Edge size [m]
n = 60    # Number of elements in each direction
h = L/n   # Element size [m]

rho = 2700     # Density [kg/m3]
Vp = 3000      # P wave velocity [m/s]
Vs = 1732      # S wave velocity [m/s]

smag = 1e6
freq = 10  # Central frequency of Ricker wavelet [Hz]
source_locations = [[-L/2, L/2]]

time_step = 2e-4  # [s]
final_time = 2.0  # [s]
out_freq = int(0.01/time_step)


def build_solver(local_abc, dt_scheme):
    d = {}

    d["options"] = {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 4,
        "dimension": 2,
    }

    d["parallelism"] = {
        "type": "automatic",
    }

    d["mesh"] = {
        "Lz": L,
        "Lx": L,
        "Ly": 0,
        "h": h,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }

    d["acquisition"] = {
        "source_type": "ricker",
        "source_locations": source_locations,
        "frequency": freq,
        "delay": 1.5,
        "delay_type": "multiples_of_minimun",
        "amplitude": smag * np.array([0, 1]),
        "receiver_locations": [],
    }

    d["synthetic_data"] = {
        "type": "object",
        "density": rho,
        "p_wave_velocity": Vp,
        "s_wave_velocity": Vs,
        "real_velocity_file": None,
    }

    d["time_axis"] = {
        "initial_time": 0,
        "final_time": final_time,
        "dt": time_step,
        "output_frequency": out_freq,
        "gradient_sampling_frequency": 1,
    }

    d["visualization"] = {
        "forward_output": False,
        "time": True,
        "time_filename": f"{output_dir}/time.npy",
        "mechanical_energy": True,
        "mechanical_energy_filename": f"{output_dir}/mechanical_energy_{local_abc}_{dt_scheme}.npy",
    }

    if local_abc is not None:
        d["absorving_boundary_conditions"] = {
            "status": True,
            "damping_type": "local",
            "local": {
                "type": local_abc,
                "dt_scheme": dt_scheme,
            },
        }

    wave = spyro.IsotropicWave(d)
    wave.set_mesh(mesh_parameters={'dx': h})

    return wave
