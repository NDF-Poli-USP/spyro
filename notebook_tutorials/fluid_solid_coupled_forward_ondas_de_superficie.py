import time
import resource
import numpy as np

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.plots.receiver_plots import (
    plot_receiver_response,
    plot_interface_displacement_continuity,
)
from spyro.plots.general_plots import (
    plot_model_jessica,
    plot_shots,
)

dictionary = {}

dictionary["options"] = {
    "cell_type": "Q",
    "variant": "lumped",
    "degree": 2,
    "dimension": 2,
}

dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": 31.2,
    "length_x": 12.0,
    "length_y": 0.0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
    "edge_length": 0.06,
    "interface_x": 6.0,
    "absorb_left": False,
    "absorb_right": False,
    "absorb_top": False,
    "absorb_bottom": False,
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-15.6, 6.5)],
    "frequency": 10.0,
    "delay": 1.0 / 10.0,
    "delay_type": "time",
    "receiver_locations": [(-15.6, 6.03)],
    "solid_receiver_locations": [(-15.6, 5.97)],
    "user_vertex_only_mesh": True,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 1.5,
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
    "p_equivalent_output": False,
    "p_equivalent_output_filename": "results/p_equivalent.pvd",
    "interface_error_frequency": 20,
    "sigma_xx_output": True,
    "sigma_xx_output_filename": "results/sigma_xx.pvd",
}

dictionary["synthetic_data"] = {
    "type": "object",
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
    "real_velocity_file": None,
}

Wave_obj = AcousticElasticWave(dictionary=dictionary)
Wave_obj.use_monolithic = False

t_start = time.perf_counter()
Wave_obj.forward_solve()
elapsed = time.perf_counter() - t_start
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

plot_model_jessica(Wave_obj, filename="results/model.png", flip_axis=False, show=True)
last_pressure_data = Wave_obj.get_function().dat.data_ro_with_halos[:]
# plot_shots(
#     Wave_obj, contour_lines=100,
#     vmin=-np.max(last_pressure_data), vmax=np.max(last_pressure_data),
#     show=True,
# )

print("Computational cost: Fluid-Solid Coupled")
print(f"  Elapsed time (s): {elapsed:.2f}")
print(f"  Memory (MB):      {mem_mb:.2f}")
np.savez("results/cost.npz", elapsed=elapsed, memory_mb=mem_mb)

plot_interface_displacement_continuity(Wave_obj, receiver_index=0)

receiver_data = Wave_obj.forward_solution_receivers[:, 0]
plot_receiver_response(
    receiver_data,
    final_time=dictionary["time_axis"]["final_time"],
    filename="results/receiver_fluid.png",
    receiver_id_for_title=0,
)

# solid_data = np.array(Wave_obj.solid_receiver_history)[:, 0, :]
# plot_displacement_components(
#     time_vector=np.linspace(0, dictionary["time_axis"]["final_time"], len(solid_data)),
#     receiver_results=solid_data,
#     source_type="Ricker",
#     filename="results/receiver_solid.png",
# )

import matplotlib.pyplot as plt
from firedrake import triplot

fig, ax = plt.subplots(figsize=(8, 8))
triplot(Wave_obj.mesh, axes=ax)
ax.set_aspect("equal")
ax.set_xlabel("Z (km)", fontsize=18)
ax.set_ylabel("X (km)", fontsize=18)
ax.set_title("Parent mesh", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
plt.savefig("results/mesh_only.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 8))
triplot(Wave_obj.submesh_fluid, axes=ax, interior_kw={"edgecolors": "blue"})
triplot(Wave_obj.submesh_solid, axes=ax, interior_kw={"edgecolors": "orange"})
ax.set_aspect("equal")
ax.set_xlabel("Z (km)", fontsize=18)
ax.set_ylabel("X (km)", fontsize=18)
ax.set_title("Child submeshes", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
plt.savefig("results/mesh_domains.png", dpi=150)
plt.close()