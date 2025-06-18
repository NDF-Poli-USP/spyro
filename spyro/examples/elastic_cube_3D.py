'''
From Turkel et al. (2023):
https://doi.org/10.1016/j.wavemoti.2022.103109
'''
import numpy as np
import spyro

L = 2000   # Size of the edges of the cube [m]
N = 5      # Number of elements in each direction
h = L/N    # Element size [m]

c_p = 5000  # P-wave velocity [m/s]
c_s = 2500  # S-wave velocity [m/s]
rho = 1000  # Density [kg/m3]

smag = 1e9  # Source magnitude
freq = 1   # Source frequency [Hz]

final_time = 2
time_step = 5e-4
out_freq = (0.01/time_step)

print(f'Element size: {h} m')
print(f'Cross time  : {h/c_p} s')
print(f'Time step   : {time_step} s')

d = {}

d["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 3,
    "dimension": 3,
}

d["parallelism"] = {
    "type": "automatic",
}

d["mesh"] = {
    "Lz": L,
    "Lx": L,
    "Ly": L,
    "h": h,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

d["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [[-L/2, L/2, L/2]],
    "frequency": freq,
    "delay": 0,
    "delay_type": "time",
    "amplitude": np.array([smag, 0, 0]),
    "receiver_locations": [],
}

d["synthetic_data"] = {
    "type": "object",
    "density": rho,
    "p_wave_velocity": c_p,
    "s_wave_velocity": c_s,
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
    "forward_output": True,
    "forward_output_filename": "results/elastic_cube_3D.pvd",
    "fwi_velocity_model_output": False,
    "gradient_output": False,
    "adjoint_output": False,
    "debug_output": False,
}

d["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "local",
}

wave = spyro.IsotropicWave(d)
wave.set_mesh(input_mesh_parameters={'edge_length': h})
