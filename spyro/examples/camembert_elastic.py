'''
From Mohammad Mehdi Ghorbani's dissertation:
https://www.teses.usp.br/teses/disponiveis/3/3152/tde-16032023-085210/pt-br.php
'''
import firedrake as fire
import numpy as np
import spyro

Lz = 0.500 # [km]
Lx = 0.500 # [km]

rho = 7850          # [kg/m3]
lambda_in = 6.86e9  # [Pa]
lambda_out = 9.88e9 # [Pa]
mu_in = 3.86e9      # [Pa]
mu_out = 5.86e9     # [Pa]

freq = 2 # Central frequency of Ricker wavelet [Hz]
hf = 0.090 # [km]
hs = 0.100 # [km]
source_locations = spyro.create_transect((-hf, 0.2*Lx), (-hf, 0.8*Lx), 3)
receiver_locations = spyro.create_transect((-hs, 0), (-hs, Lx), 40)
source_locations = [[-hf, 0.5*Lx]]

time_step = 5e-4 # [s]
final_time = 5   # [s]
out_freq = 100

nz = 80
nx = 80
mesh = fire.RectangleMesh(nz, nx, 0, Lx, originX=-Lz, diagonal='crossed')
z, x = fire.SpatialCoordinate(mesh)

zc = 0.250 # [km]
xc = 0.250 # [km]
ri = 0.050 # [km]
camembert = lambda v_inside, v_outside: fire.conditional(
    (z - zc) ** 2 + (x - xc) ** 2 < ri**2, v_inside, v_outside)

d = {}

d["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 2,
    "dimension": 2,
}

d["parallelism"] = {
    "type": "automatic",
}

d["mesh"] = {
    "Lz": Lz,
    "Lx": Lx,
    "Ly": 0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

d["acquisition"] = {
    "source_type": "ricker",
    "source_locations": source_locations,
    "frequency": freq,
    "delay": 0,
    "delay_type": "time",
    "amplitude": np.array([0, 1e-9]),
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

wave = spyro.IsotropicWave(d)
wave.set_mesh(user_mesh=mesh, mesh_parameters={})
#wave.set_initial_velocity_model(constant=[1.5, 1.5])
#spyro.plots.plot_model(wave, filename="model.png", flip_axis=False, show=True)
wave.forward_solve()