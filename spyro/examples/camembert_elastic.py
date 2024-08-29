'''
From Mohammad Mehdi Ghorbani's dissertation:
https://www.teses.usp.br/teses/disponiveis/3/3152/tde-16032023-085210/pt-br.php
'''
import firedrake as fire
import numpy as np
import spyro

Lz = 500 # [m]
Lx = 500 # [m]

rho = 7850          # [kg/m3]
lambda_in = 6.86e9  # [Pa]
lambda_out = 9.88e9 # [Pa]
mu_in = 3.86e9      # [Pa]
mu_out = 5.86e9     # [Pa]

smag = 1e6
freq = 2 # Central frequency of Ricker wavelet [Hz]
hf =  90 # [m]
hs = 100 # [m]
source_locations = spyro.create_transect((-hf, 0.2*Lx), (-hf, 0.8*Lx), 3)
receiver_locations = spyro.create_transect((-hs, 0), (-hs, Lx), 40)
source_locations = [[-hf, 0.5*Lx]]

time_step = 2e-4 # [s]
final_time = 1.5 # [s]
out_freq = int(0.01/time_step)

nz = 20
nx = 20
mesh = fire.RectangleMesh(nz, nx, 0, Lx, originX=-Lz, diagonal='crossed')
z, x = fire.SpatialCoordinate(mesh)

zc = 250 # [m]
xc = 250 # [m]
ri =  50 # [m]
camembert = lambda v_inside, v_outside: fire.conditional(
    (z - zc) ** 2 + (x - xc) ** 2 < ri**2, v_inside, v_outside)

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
    "user_mesh": mesh,
}

d["acquisition"] = {
    "source_type": "ricker",
    "source_locations": source_locations,
    "frequency": freq,
    "delay": 0,
    "delay_type": "time",
    "amplitude": np.array([0, smag]),
    "receiver_locations": receiver_locations,
}

d["synthetic_data"] = {
    "type": "object",
    "density": fire.Constant(rho),
    "lambda": camembert(lambda_in, lambda_out),
    "mu": camembert(mu_in, mu_out),
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
    "forward_output_filename": "results/forward_output.pvd",
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
wave.set_mesh(user_mesh=mesh, mesh_parameters={})

print(f'Number of degrees of freedom: {wave.function_space.dim()}')

wave.forward_solve()
