import firedrake as fire
import numpy as np

import spyro

L = 10000  # [m]

rho = 7850          # [kg/m3]
lambda_in = 6.86e9  # [Pa]
lambda_out = 9.88e9  # [Pa]
mu_in = 3.86e9      # [Pa]
mu_out = 5.86e9     # [Pa]

smag = 1e6
freq = 5  # Central frequency of Ricker wavelet [Hz]
hf = 90  # [m]
hs = 100  # [m]

source_z = -L*0.5
source_x = L*0.5
source_locations = [(source_z, source_x)]
receiver_bin_center_offset = 3000
receiver_bin_width = 1500
receiver_quantity = 500

bin1_startZ = source_z + receiver_bin_center_offset - receiver_bin_width / 2.0
bin1_endZ = source_z + receiver_bin_center_offset + receiver_bin_width / 2.0
bin1_startX = source_x - receiver_bin_width / 2.0
bin1_endX = source_x + receiver_bin_width / 2.0

receiver_locations = spyro.create_2d_grid(
    bin1_startZ,
    bin1_endZ,
    bin1_startX,
    bin1_endX,
    int(np.sqrt(receiver_quantity)),
)

time_step = 2e-4  # [s]
final_time = 1.0  # [s]
out_freq = int(0.01/time_step)
cpw = 5

c = ((lambda_out + 2*mu_out)/7850)**0.5

edge_length = c/(freq*cpw)
n = int(L/edge_length)+1
mesh = fire.RectangleMesh(n, n, 0, L, originX=-L)
z, x = fire.SpatialCoordinate(mesh)

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
    "delay": 1.5,
    "amplitude": np.array([0, smag]),
    # "amplitude": smag * np.eye(2),
    # "amplitude": smag * np.array([[0, 1], [-1, 0]]),
    "receiver_locations": receiver_locations,
}

d["synthetic_data"] = {
    "type": "object",
    "density": fire.Constant(rho),
    "lambda": fire.Constant(lambda_out),
    "mu": fire.Constant(mu_out),
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
wave.set_mesh(user_mesh=mesh, input_mesh_parameters={})
wave.forward_solve()
