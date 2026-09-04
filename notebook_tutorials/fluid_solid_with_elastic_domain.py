import time
import resource
import numpy as np
import firedrake as fire

from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave

# ===============================================================================
# Mesmos parâmetros físicos do caso acoplado
V_f, rho_f = 1.5, 1.0
V_p, V_s, rho_s = 2.0, 1.2, 2.0

mu_solid  = rho_s * V_s**2
lam_solid = rho_s * V_p**2 - 2.0 * mu_solid
mu_fluid_equiv  = 0.0
lam_fluid_equiv = rho_f * V_f**2

interface_x = 0.5 

# ===============================================================================
dictionary = {}
dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 2,
    "dimension": 2,
}

dictionary["mesh"] = {
    "length_z": 1.0,
    "length_x": 1.0,
    "length_y": 0.0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
    "edge_length": 0.0025,
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.5, 0.6)],
    "frequency": 25.0,
    "delay": 1.0/25.0,
    "delay_type": "time",
    "receiver_locations": [(-0.51, 0.5025)],
    "user_vertex_only_mesh": True,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 0.22,
    "dt": 0.0001,
    "output_frequency": 10,
    "gradient_sampling_frequency": 1,
}

dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/elastic_only_displacement.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
    "debug_output": False,
}

# ===============================================================================
Wave_obj = IsotropicWave(dictionary=dictionary)

DG0 = fire.FunctionSpace(Wave_obj.mesh, "DG", 0)
x_coord = Wave_obj.mesh_x 

lam_field = fire.Function(DG0).interpolate(
    fire.conditional(x_coord < interface_x, lam_solid, lam_fluid_equiv)
)
mu_field = fire.Function(DG0).interpolate(
    fire.conditional(x_coord < interface_x, mu_solid, mu_fluid_equiv)
)
rho_field = fire.Function(DG0).interpolate(
    fire.conditional(x_coord < interface_x, rho_s, rho_f)
)

dictionary["synthetic_data"] = {
    "type": "object",
    "density": rho_field,
    "lambda": lam_field,
    "mu": mu_field,
}

# ===============================================================================
t_start = time.perf_counter()
Wave_obj.forward_solve()
t_end = time.perf_counter()

elapsed = t_end - t_start
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print(f"\n{'='*50}")
print("Computational cost: Elastic-only (whole domain)")
print(f"  Elapsed time (s): {elapsed:.2f}")
print(f"  Memory (MB):      {mem_mb:.2f}")
print(f"{'='*50}")

np.savez("results/cost_elastic_only.npz", elapsed=elapsed, memory_mb=mem_mb)