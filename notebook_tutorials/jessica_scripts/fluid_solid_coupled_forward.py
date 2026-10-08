import os
import time
import resource
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from firedrake import triplot

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.plots.general_plots import plot_model_jessica
from sparsity_plots import save_sparsity_matrices
from spyro.utils.utils import communicate

import sparsity_plots; print("sparsity_plots from:", sparsity_plots.__file__)

# ===========================================================================
# Settings
# ===========================================================================
USE_MONOLITHIC = False
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ===========================================================================
# Cases
# ===========================================================================
case_small_2D = {
    "cell_type": "Q",
    "degree": 4,
    "length_z": 1.0,
    "length_x": 1.0,
    "length_y": 0.0,
    "edge_length": 0.010,
    "interface_x": 0.5,
    "source_locations": [(-0.5, 0.55)],
    "frequency": 25,
    "receiver_location": [(-0.51, 0.51)],
    "solid_receiver_locations": [(-0.49, 0.49)],
    "final_time": 0.22,
    "dt": 0.0001,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.0,
    "p_wave_velocity": 2.0,
    "s_wave_velocity": 1.2,
}

case_large_2D = {
    "cell_type": "Q",
    "degree": 4,
    "length_z": 30.0,
    "length_x": 12.0,
    "length_y": 0.0,
    "edge_length": 0.040,
    "interface_x": 6.0,
    "source_locations": [(-15.0, 6.6)],
    "frequency": 10,
    "receiver_location": [(-15.01, 6.01)],
    "solid_receiver_locations": [(-14.99, 5.99)],
    "final_time": 2.0,
    "dt": 0.0005,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
}

case_3D = {
    "cell_type": "T",
    "degree": 3,
    "length_z": 6.0,
    "length_x": 4.0,
    "length_y": 6.0,
    "edge_length": 0.050,
    "interface_x": 2.5,
    "source_locations": [(-3.0, 2.65, 3.0)],
    "frequency": 10,
    "receiver_location": [(-3.01, 2.51, 3.0)],
    "solid_receiver_locations": [(-2.99, 2.49, 3.0)],
    "final_time": 1.0,
    "dt": 4e-5,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
}

cases = {1: ("case_small_2d", case_small_2D),
         2: ("case_large_2d", case_large_2D),
         3: ("case_3d", case_3D)}
CASE_NUM = 3
CASE_NAME, selected_case = cases[CASE_NUM]

DIMENSION = 3 if CASE_NAME == "case_3d" else 2
SCHEME = "monolithic" if USE_MONOLITHIC else "sequential"
TAG = f"{CASE_NAME}_{SCHEME}_{DIMENSION}d"

# ===========================================================================
# Dictionary
# ===========================================================================
dictionary = {}

dictionary["options"] = {
    "cell_type": selected_case["cell_type"],
    "variant": "lumped",
    "degree": selected_case["degree"],
    "dimension": DIMENSION,
}

dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": selected_case["length_z"],
    "length_x": selected_case["length_x"],
    "length_y": selected_case["length_y"],
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
    "edge_length": selected_case["edge_length"],
    "interface_x": selected_case["interface_x"],
    "absorb_left": False,
    "absorb_right": False,
    "absorb_top": False,
    "absorb_bottom": False,
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": selected_case["source_locations"],
    "frequency": selected_case["frequency"],
    "delay": 1.0 / selected_case["frequency"],
    "delay_type": "time",
    "amplitude": 1.0,
    "receiver_locations": selected_case["receiver_location"],
    "solid_receiver_locations": selected_case["solid_receiver_locations"],
    "use_vertex_only_mesh": False,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": selected_case["final_time"],
    "dt": selected_case["dt"],
    "output_frequency": 2500,
    "gradient_sampling_frequency": 1,
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": f"{OUT}/pressure_{TAG}.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "graadient_output": False,
    "gradient_filename": None,
    "debug_output": False,
    "displacement_output": False,
    "displacement_output_filename": f"{OUT}/displacement_{TAG}.pvd",
    "snapshot_frequency": 20,
    "snapshot_output_dir": f"{OUT}/snapshots_{TAG}",
    "p_equivalent_output": False,
    "p_equivalent_output_filename": f"{OUT}/p_equivalent_{TAG}.pvd",
    "interface_error_frequency": 20,
    "sigma_xx_output": True,
    "sigma_xx_output_filename": f"{OUT}/sigma_xx_{TAG}.pvd",
}

dictionary["synthetic_data"] = {
    "type": "object",
    "velocity_fluid": selected_case["velocity_fluid"],
    "bulk_modulus": selected_case["bulk_modulus"],
    "density_fluid": selected_case["density_fluid"],
    "density_solid": selected_case["density_solid"],
    "p_wave_velocity": selected_case["p_wave_velocity"],
    "s_wave_velocity": selected_case["s_wave_velocity"],
    "real_velocity_file": None,
    "s_wave_velocity_fluid": 0.0,
    "welded_interface_test": False,
    "rotation_penalty": 1.0,
}

# ===========================================================================
# Run
# ===========================================================================
Wave_obj = AcousticElasticWave(dictionary=dictionary)
Wave_obj.use_monolithic = USE_MONOLITHIC

t_start = time.perf_counter()
Wave_obj.forward_solve()
elapsed = time.perf_counter() - t_start
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print(f"Case: {TAG}")
print(f"Elapsed time: {elapsed:.2f}s")
print(f"Memory: {mem_mb:.2f} MB")

# ===========================================================================
# Plots (2D only)
# ===========================================================================
if DIMENSION == 2 and Wave_obj.comm.comm.size == 1:
    plot_model_jessica(Wave_obj, filename=f"{OUT}/model_{TAG}.png", flip_axis=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    triplot(Wave_obj.mesh, axes=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("Z (km)", fontsize=18)
    ax.set_ylabel("X (km)", fontsize=18)
    ax.set_title("Parent mesh", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    plt.savefig(f"{OUT}/mesh_parent.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    triplot(Wave_obj.submesh_fluid, axes=ax, interior_kw={"edgecolors": "blue"})
    triplot(Wave_obj.submesh_solid, axes=ax, interior_kw={"edgecolors": "orange"})
    ax.set_aspect("equal")
    ax.set_xlabel("Z (km)", fontsize=18)
    ax.set_ylabel("X (km)", fontsize=18)
    ax.set_title("Child submeshes", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    plt.savefig(f"{OUT}/mesh_child.png", dpi=150)
    plt.close()

from spyro.utils.utils import communicate

# ===========================================================================
# Save data
# ===========================================================================
p_spyro = np.asarray(Wave_obj.forward_solution_receivers)[:, 0]
u_all = communicate(np.asarray(Wave_obj.solid_receiver_history), Wave_obj.comm)
u_solid = u_all[:, 0, :]

n_procs = Wave_obj.comm.comm.size
rank = Wave_obj.comm.comm.rank

if n_procs == 1:
    # assembled with local rows only in parallel: run in serial
    save_sparsity_matrices(Wave_obj, f"{OUT}/sparsity_{TAG}.npz")

if rank == 0:
    np.savez(
        f"{OUT}/spyro_receiver_data_{TAG}.npz",
        dt=dictionary["time_axis"]["dt"],
        final_time=dictionary["time_axis"]["final_time"],
        p_spyro=p_spyro,
        u_solid=u_solid,
    )
    print(f"[OK] Saved: {OUT}/spyro_receiver_data_{TAG}.npz")