import firedrake as fire
import numpy as np
import spyro

Lz = 3000  # [m]
Lx = 4000  # [m]
nz = 60
nx = 80

rho1 = 2700     # [kg/m3]
Vp1 = 3000      # [Pa]
Vs1 = 1732.051  # [Pa]
rho2 = 2200     # [kg/m3]
Vp2 = 2500      # [Pa]
Vs2 = 1443.375  # [Pa]

smag = 1e10
freq = 10  # Central frequency of Ricker wavelet [Hz]
source_locations = [[-400, 2000]]
receiver_locations = spyro.create_transect((-800, 300), (-800, 3700), 11)

time_step = 1e-4  # [s]
final_time = 2.0  # [s]
out_freq = int(0.01/time_step)

mesh = fire.RectangleMesh(nz, nx, 0, Lx, originX=-Lz, diagonal='crossed')
z, x = fire.SpatialCoordinate(mesh)

hole = lambda v_inside, v_outside: fire.conditional(
    fire.And(fire.And(1000 < z, z < 2000),
             fire.And(1500 < x, x < 2500)),
    v_inside,
    v_outside)

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
    "delay_type": "multiples_of_minimun",
    "amplitude": smag * np.eye(2),
    "receiver_locations": receiver_locations,
}

d["synthetic_data"] = {
    "type": "object",
    "density": hole(rho2, rho1),
    "p_wave_velocity": hole(Vp2, Vp1),
    "s_wave_velocity": hole(Vs2, Vs1),
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


if __name__ == "__main__":
    #wave.set_initial_velocity_model(constant=fire.Constant((1.5, 1.0)))
    #spyro.plots.plot_model(wave, filename="model.png", flip_axis=False, show=True)
    wave.forward_solve()