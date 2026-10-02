import time
import resource
import numpy as np

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.plots.receiver_plots import (
    plot_receiver_response,
    plot_interface_displacement_continuity,
)

dictionary = {}

dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 2,
    "dimension": 3,
}

dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": 2.0,
    "length_x": 2.0,
    "length_y": 2.0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
    "edge_length": 0.05,
    "interface_x": 1.0,
    "absorb_left": False,
    "absorb_right": False,
    "absorb_top": False,
    "absorb_bottom": False,
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.0, 1.1, 1.0)],
    "frequency": 25.0,
    "delay": 1.0 / 25.0,
    "delay_type": "time",
    "receiver_locations": [(-1.0, 1.025, 1.0)],
    "solid_receiver_locations": [(-1.0, 0.975, 1.0)],
    "user_vertex_only_mesh": True,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 0.05,   # TESTE CURTO primeiro — só pra ver se roda sem travar
    "dt": 0.001,
    "output_frequency": 10,
    "gradient_sampling_frequency": 1,
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/pressure.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "graadient_output": False,
    "gradient_filename": None,
    "debug_output": False,
    "displacement_output": False,
    "displacement_output_filename": "results/displacement.pvd",
    "snapshot_frequency": 20,
    "snapshot_output_dir": "results/snapshots",
    "p_equivalent_output": True,
    "p_equivalent_output_filename": "results/p_equivalent.pvd",
    "interface_error_frequency": 20,
    "sigma_xx_output": False,
    "sigma_xx_output_filename": "results/sigma_xx.pvd",
}

dictionary["synthetic_data"] = {
    "type": "object",
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.0,
    "p_wave_velocity": 2.0,
    "s_wave_velocity": 1.2,
    "real_velocity_file": None,
}

Wave_obj = AcousticElasticWave(dictionary=dictionary)
Wave_obj.use_monolithic = True

t_start = time.perf_counter()
Wave_obj.forward_solve()
elapsed = time.perf_counter() - t_start
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print("Computational cost: Fluid-Solid Coupled (3D test)")
print(f"  Elapsed time (s): {elapsed:.2f}")
print(f"  Memory (MB):      {mem_mb:.2f}")
np.savez("results/cost_3d.npz", elapsed=elapsed, memory_mb=mem_mb)

plot_interface_displacement_continuity(Wave_obj, receiver_index=0)

receiver_data = Wave_obj.forward_solution_receivers[:, 0]
plot_receiver_response(
    receiver_data,
    final_time=dictionary["time_axis"]["final_time"],
    filename="results/receiver_fluid_3d.png",
    receiver_id_for_title=0,
)

# NOTA: plot_model_jessica e os plots de triplot(mesh) foram removidos
# neste teste 3D — tripcolor/triplot do Firedrake são específicos pra 2D.