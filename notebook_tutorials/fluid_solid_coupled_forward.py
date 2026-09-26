import os
import time
import resource
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from firedrake import triplot

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.plots.general_plots import (
    plot_model_jessica,
    plot_shots,
)
from sparsity_plots import save_sparsity_matrices

from check_spyro_moment_source import check_spyro_moment_source

import sparsity_plots; print("sparsity_plots from:", sparsity_plots.__file__)

# ===========================================================================
# CASE SELECTION: formulation and scheme. Every saved file carries this tag.
# ===========================================================================
FLUID_FORMULATION = "displacement"   # "pressure" (P-us) or "displacement" (u_f-u_s)
USE_MONOLITHIC = False           # True = monolithic, False = sequential
COST_RUN = False                 # True: measure cost (no outputs, no recordings)
                                 # False: ParaView outputs + receiver .npz for plots

SCHEME = "monolithic" if USE_MONOLITHIC else "sequential"
TAG = f"{FLUID_FORMULATION}_{SCHEME}"   # e.g. pressure_sequential
OUT = "results"
os.makedirs(OUT, exist_ok=True)
print(f"=== Case: {TAG} ({'cost run' if COST_RUN else 'validation run'}) ===")

dictionary = {}

dictionary["options"] = {
    "cell_type": "Q",
    "variant": "lumped",
    "degree": 2,
    "dimension": 2,
    "fluid_formulation": FLUID_FORMULATION,
}

dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": 1.0,
    "length_x": 1.0,
    "length_y": 0.0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
    "edge_length": 0.005,
    "interface_x": 0.5,
    "absorb_left": False,
    "absorb_right": False,
    "absorb_top": False,
    "absorb_bottom": False,
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.5, 0.6)],
    "frequency": 25.0,
    "delay": 1.0 / 25.0,
    "delay_type": "time",
    # isotropic moment source for u_f-u_s; scalar source for P-us
    "amplitude": np.eye(2) if FLUID_FORMULATION == "displacement" else 1.0,
    "receiver_locations": [(-0.51, 0.5025)],
    "solid_receiver_locations": [(-0.49, 0.4975)],
    "use_vertex_only_mesh": False,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 0.22,
    "dt": 0.0001,
    "output_frequency": 10,
    "gradient_sampling_frequency": 1,
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": f"{OUT}/fluid_{TAG}.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "graadient_output": False,
    "gradient_filename": None,
    "debug_output": False,
    "displacement_output": True,
    "displacement_output_filename": f"{OUT}/displacement_{TAG}.pvd",
    "snapshot_frequency": 20,
    "snapshot_output_dir": f"{OUT}/snapshots_{TAG}",
    "p_equivalent_output": True,
    "p_equivalent_output_filename": f"{OUT}/p_equivalent_{TAG}.pvd",
    "interface_error_frequency": 20,
    "sigma_xx_output": True,
    "sigma_xx_output_filename": f"{OUT}/sigma_xx_{TAG}.pvd",
    "fluid_pressure_output": True,
    "fluid_pressure_output_filename": f"{OUT}/fluid_pressure_{TAG}.pvd",
}
if COST_RUN:  # no ParaView output in cost runs
    for key in dictionary["visualization"]:
        if key.endswith("_output"):
            dictionary["visualization"][key] = False

dictionary["synthetic_data"] = {
    "type": "object",
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.0,
    "p_wave_velocity": 2.0,
    "s_wave_velocity": 1.2,
    "real_velocity_file": None,
    "s_wave_velocity_fluid": 0.0,
    "welded_interface_test": False,
    "rotation_penalty": 1.0,
}

Wave_obj = AcousticElasticWave(dictionary=dictionary)
Wave_obj.use_monolithic = USE_MONOLITHIC
# Extra per-step recordings (off in cost runs):
#   u_f-u_s: fluid pressure -kappa*div(u_f) at the fluid receiver
#   P-us:    fluid normal displacement from -grad(P)/rho_f (interface continuity)
Wave_obj.record_fluid_pressure = not COST_RUN
Wave_obj.record_fluid_displacement = not COST_RUN

if Wave_obj.fluid_is_vector:
    print("amplitude:", Wave_obj.sources.amplitude)
    print("tabulation shape:", np.shape(Wave_obj.sources.cell_tabulations))
    # check_spyro_moment_source(Wave_obj)

t_start = time.perf_counter()
Wave_obj.forward_solve()
elapsed = time.perf_counter() - t_start
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

plot_model_jessica(Wave_obj, filename=f"{OUT}/model_{TAG}.png",
                   flip_axis=False, show=True)

print(f"Computational cost: Fluid-Solid Coupled ({TAG})")
print(f"  Elapsed time (s): {elapsed:.2f}")
print(f"  Memory (MB):      {mem_mb:.2f}")
if COST_RUN:
    np.savez(f"{OUT}/cost_{TAG}.npz", elapsed=elapsed, memory_mb=mem_mb)

# ---- Meshes (independent of the case) ----
fig, ax = plt.subplots(figsize=(8, 8))
triplot(Wave_obj.mesh, axes=ax)
ax.set_aspect("equal")
ax.set_xlabel("Z (km)", fontsize=18)
ax.set_ylabel("X (km)", fontsize=18)
ax.set_title("Parent mesh", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
plt.savefig(f"{OUT}/mesh_only.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 8))
triplot(Wave_obj.submesh_fluid, axes=ax, interior_kw={"edgecolors": "blue"})
triplot(Wave_obj.submesh_solid, axes=ax, interior_kw={"edgecolors": "orange"})
ax.set_aspect("equal")
ax.set_xlabel("Z (km)", fontsize=18)
ax.set_ylabel("X (km)", fontsize=18)
ax.set_title("Child submeshes", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
plt.savefig(f"{OUT}/mesh_domains.png", dpi=150)
plt.close()

# ---- Sparsity data (plotted later in plot_all_results.py) ----
save_sparsity_matrices(Wave_obj, f"{OUT}/sparsity_{TAG}.npz")

# ---- Receiver data for plot_all_results.py ----
if COST_RUN:
    print("[INFO] Cost run: receiver .npz not saved.")
else:
    # Fluid receiver: pressure
    if Wave_obj.fluid_is_vector:
        p_spyro = np.asarray(Wave_obj.fluid_pressure_history)[:, 0]
    else:
        p_spyro = np.asarray(Wave_obj.forward_solution_receivers)[:, 0]

    # Solid receiver: (u_z, u_x)
    u_solid = np.asarray(Wave_obj.solid_receiver_history)[:, 0, :]

    # Fluid-side normal displacement u_x (interface continuity plot)
    if Wave_obj.fluid_is_vector:
        # u_f is the unknown itself: x component at the fluid receiver
        u_fluid_x = np.asarray(Wave_obj.forward_solution_receivers)[:, 0, 1]
    elif Wave_obj.fluid_displacement_history:
        # integrated from -grad(P)/rho_f in the class
        u_fluid_x = np.asarray(Wave_obj.fluid_displacement_history)[:, 0]
    else:
        u_fluid_x = np.array([])

    np.savez(
        f"{OUT}/spyro_receiver_data_{TAG}.npz",
        dt=dictionary["time_axis"]["dt"],
        final_time=dictionary["time_axis"]["final_time"],
        p_spyro=p_spyro,
        u_solid=u_solid,
        u_fluid_x=u_fluid_x,
    )
    print(f"[OK] Saved: {OUT}/spyro_receiver_data_{TAG}.npz")