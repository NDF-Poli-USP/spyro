import os
import time
import resource
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from firedrake import Function, FunctionSpace, triplot, tripcolor

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.solvers.acoustic_solver_construction_no_pml import velocity_fluid
from spyro.utils.utils import communicate
from spyro.io.segy_io import create_segy_from_grid
from sparsity_plots import save_sparsity_matrices

SCRIPT_DIR = Path(__file__).resolve().parent


def build_segy(filename, length_z, length_x, dz, interface, v_fluid, v_solid, dx=None):
    dx = dz if dx is None else dx
    z = np.linspace(0.0, -length_z, int(round(length_z / dz)) + 1)
    x = np.linspace(0.0, length_x, int(round(length_x / dx)) + 1)
    X, Z = np.meshgrid(x, z, indexing="ij")
    if callable(interface):
        z_int = np.broadcast_to(interface(x), x.shape)
    elif isinstance(interface, (str, Path)):
        d = np.loadtxt(interface)
        z_int = np.interp(x, d[:, 0], d[:, 1])
    else:
        z_int = np.full(x.shape, float(interface))
    field = lambda v: np.broadcast_to(v(Z, X), Z.shape) if callable(v) else np.full(Z.shape, float(v))
    v = np.where(Z > z_int[:, None], field(v_fluid), field(v_solid))
    create_segy_from_grid(v[:, ::-1].astype(np.float32), str(filename))


def specfem_interface(filename, index, length_z):
    with open(filename) as f:
        rows = [line.split() for line in f
                if line.strip() and not line.lstrip().startswith("#")]
    it = iter(rows[1:])
    interfaces = []
    for _ in range(int(rows[0][0])):
        n = int(next(it)[0])
        interfaces.append(np.array([[float(a), float(b)] for a, b in (next(it) for _ in range(n))]))
    xs = interfaces[index][:, 0] / 1000.0
    zs = interfaces[index][:, 1] / 1000.0 - length_z
    return lambda x: np.interp(x, xs, zs)


def acquisition_on_interface(z_int, x_source, height_source, offset, distance):
    z_at = lambda x: float(z_int(np.array([x]))[0])
    return {
        "source_locations": [(z_at(x_source) + height_source, x_source)],
        "receiver_location": [(z_at(x_source - offset) + distance, x_source - offset)],
        "solid_receiver_locations": [(z_at(x_source + offset) - distance, x_source + offset)],
    }


def plot_mesh_xz(meshes, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    swap = Affine2D().from_values(0, 1, 1, 0, 0, 0) + ax.transData
    for mesh, color in meshes:
        kw = {"edgecolors": color} if color else {}
        for coll in triplot(mesh, axes=ax, interior_kw=kw):
            coll.set_transform(swap)
    ax.set_xlim(0.0, selected_case["length_x"])
    ax.set_ylim(-selected_case["length_z"], 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (km)", fontsize=18)
    ax.set_ylabel("Z (km)", fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_model_xz(wave, filename):
    vf = Function(FunctionSpace(wave.submesh_fluid, "CG", 1)).interpolate(velocity_fluid(wave))
    vs = Function(FunctionSpace(wave.submesh_solid, "CG", 1)).interpolate(
        ((wave.lmbda + 2 * wave.mu) / wave.rho) ** 0.5)
    vmin = min(vf.dat.data_ro.min(), vs.dat.data_ro.min())
    vmax = max(vf.dat.data_ro.max(), vs.dat.data_ro.max())

    fig, ax = plt.subplots(figsize=(9, 9))
    swap = Affine2D().from_values(0, 1, 1, 0, 0, 0) + ax.transData
    for f in (vf, vs):
        img = tripcolor(f, axes=ax, vmin=vmin, vmax=vmax)
        img.set_transform(swap)

    acq = dictionary["acquisition"]
    for key, marker, color, label in (
        ("source_locations", "*", "green", "Source"),
        ("receiver_locations", "v", "red", "Receiver (fluid)"),
        ("solid_receiver_locations", "^", "orange", "Receiver (solid)"),
    ):
        pts = np.asarray(acq[key])
        ax.scatter(pts[:, 1], pts[:, 0], marker=marker, c=color, s=120, label=label, zorder=5)

    ax.set_xlim(0.0, selected_case["length_x"])
    ax.set_ylim(-selected_case["length_z"], 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (km)", fontsize=18)
    ax.set_ylabel("Z (km)", fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="best", fontsize=14)
    cbar = fig.colorbar(img, ax=ax, orientation="horizontal")
    cbar.set_label("Velocity (km/s)", fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ===========================================================================
# Settings
# ===========================================================================
USE_MONOLITHIC = False
MESH_TYPE = "firedrake_mesh" # "gmsh_mesh" or "firedrake_mesh"
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
    "interface_z": -0.5,
    "source_locations": [(-0.45, 0.5)],
    "frequency": 25,
    "receiver_location": [(-0.49, 0.49)],
    "solid_receiver_locations": [(-0.51, 0.51)],
    "final_time": 0.22,
    "dt": 0.0001,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.0,
    "p_wave_velocity": 2.0,
    "s_wave_velocity": 1.2,
    "segy_dz": 0.002,
    "segy_interface": -0.5,
    "cells_per_wavelength": 6.0,
    "epsilon": 0.3,
    "delta": 0.15,
    "gamma": 0.1,
    "anisotropy_type": "exact",
}

case_large_2D = {
    "cell_type": "Q",
    "degree": 4,
    "length_z": 12.0,
    "length_x": 30.0,
    "length_y": 0.0,
    "edge_length": 0.040,
    "interface_z": -6.0,
    "source_locations": [(-5.4, 15.0)],
    "frequency": 10,
    "receiver_location": [(-5.99, 14.99)],
    "solid_receiver_locations": [(-6.01, 15.01)],
    "final_time": 2.0,
    "dt": 0.0005,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
    "segy_dz": 0.005,
    "segy_interface": -6.0,
    "cells_per_wavelength": 6.0,
}

case_3D = {
    "cell_type": "T",
    "degree": 3,
    "length_z": 3.0,
    "length_x": 3.0,
    "length_y": 3.0,
    "edge_length": 0.035,
    "interface_z": -1.5,
    "source_locations": [(-1.35, 1.5, 1.5)],
    "frequency": 10,
    "receiver_location": [(-1.49, 1.49, 1.49)],
    "solid_receiver_locations": [(-1.51, 1.51, 1.51)],
    "final_time": 0.8,
    "dt": 2.8e-5,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
}

small_irregular_interface = lambda x: -0.5 + 0.02 * np.sin(2 * np.pi * x / 0.25)
# case_small_2D_irregular = dict(
#     case_small_2D,
#     segy_interface=small_irregular_interface,
#     segy_dz=0.0005,
#     segy_dx=0.010,
# )
case_small_2D_irregular = dict(
    case_small_2D,
    cell_type="Q",
    segy_interface=small_irregular_interface,
    segy_dz=0.0005,
    segy_dx=0.010,
)

def case_large_2D_irregular():
    interface = specfem_interface(SCRIPT_DIR / "interfaces_specfem.dat", 1, 9.6)
    return dict(
        case_large_2D,
        cell_type="T",
        length_z=9.6,
        length_x=20.0,
        interface_z=None,
        segy_interface=interface,
        segy_dz=0.0005,
        segy_dx=0.040,
        dt=0.00025,
        **acquisition_on_interface(interface, 10.0, 0.6, 0.05, 0.01),
    )


cases = {1: ("case_small_2d", case_small_2D),
         2: ("case_large_2d", case_large_2D),
         3: ("case_3d", case_3D),
         4: ("case_small_2d_irregular_topography", case_small_2D_irregular),
         5: ("case_large_2d_irregular_topography", case_large_2D_irregular)}
CASE_NUM = 1
CASE_NAME, selected_case = cases[CASE_NUM]
if callable(selected_case):
    selected_case = selected_case()

if "irregular" in CASE_NAME:
    MESH_TYPE = "gmsh_mesh"
elif CASE_NAME.startswith("case_3d"):
    MESH_TYPE = "firedrake_mesh"

DIMENSION = 3 if CASE_NAME.startswith("case_3d") else 2
SCHEME = "monolithic" if USE_MONOLITHIC else "sequential"
TAG = f"{CASE_NAME}_{SCHEME}_{DIMENSION}d"

if selected_case["velocity_fluid"] is not None:
    VP_WATER = selected_case["velocity_fluid"]
else:
    VP_WATER = np.sqrt(selected_case["bulk_modulus"] / selected_case["density_fluid"])

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
    "mesh_type": MESH_TYPE,
    "edge_length": selected_case["edge_length"],
    "interface_z": selected_case["interface_z"],
}

if MESH_TYPE == "gmsh_mesh":
    segy_file = f"{OUT}/model_{TAG}.segy"
    build_segy(
        segy_file,
        selected_case["length_z"],
        selected_case["length_x"],
        selected_case["segy_dz"],
        selected_case["segy_interface"],
        VP_WATER, selected_case["s_wave_velocity"],
        dx=selected_case.get("segy_dx")
    )
    dictionary["mesh"].pop("edge_length")
    dictionary["mesh"].update({
        "output_filename": f"{OUT}/mesh_{TAG}.msh",
        "segy_velocity_model": segy_file,
        "cells_per_wavelength": selected_case["cells_per_wavelength"],
        "frequency": selected_case["frequency"],
        "min_element_size": selected_case["edge_length"],
        "grade": 0.05,
        "structured_mesh": selected_case["cell_type"] == "Q",
        "apply_winslow": False,
        "winslow_iterations": 2000,
        "vp_water": VP_WATER,
        "water_interface": True,
        "water_search_value": VP_WATER,
    })

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
    "output_frequency": int(round(0.005 / selected_case["dt"])),
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
    "sigma_zz_output": True,
    "sigma_zz_output_filename": f"{OUT}/sigma_zz_{TAG}.pvd",
}

dictionary["synthetic_data"] = {
    "type": "object",
    "bulk_modulus": selected_case["bulk_modulus"],
    "density_fluid": selected_case["density_fluid"],
    "density_solid": selected_case["density_solid"],
    "p_wave_velocity": selected_case["p_wave_velocity"],
    "s_wave_velocity": selected_case["s_wave_velocity"],
    "real_velocity_file": None,
    "epsilon": selected_case.get("epsilon", None),
    "delta": selected_case.get("delta", None),
    "gamma": selected_case.get("gamma", None),
    "anisotropy_type": selected_case.get("anisotropy_type", None),
}

dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "abc_type": "nrbc",
    "absorb_top": True,
    "absorb_bottom": True,
    "absorb_right": True,
    "absorb_left": True,
}

# ===========================================================================
# Run
# ===========================================================================
Wave_obj = AcousticElasticWave(dictionary=dictionary)

# Checagem dos markers das submalhas
print("Fluid exterior facet markers:",
      Wave_obj.scalar_function_space.mesh().exterior_facets.unique_markers)
print("Solid exterior facet markers:",
      Wave_obj.vector_function_space.mesh().exterior_facets.unique_markers)

import firedrake as fire
for name, sub in (("fluido", Wave_obj.submesh_fluid), ("solido", Wave_obj.submesh_solid)):
    V = fire.VectorFunctionSpace(sub, "DG", 0)
    c = fire.Function(V).interpolate(fire.SpatialCoordinate(sub)).dat.data_ro
    zi = -0.5 + 0.02 * np.sin(2 * np.pi * c[:, 1] / 0.25)
    wrong = (c[:, 0] < zi) if name == "fluido" else (c[:, 0] > zi)
    print(name, "- elementos do lado errado:", wrong.sum(), "de", len(c))
L = fire.assemble(fire.Constant(1.0) * Wave_obj.ds_int(Wave_obj.interface_id))
print("comprimento da interface:", L, "(senoide exata: ~1.06 km)")

Wave_obj.use_monolithic = USE_MONOLITHIC

t_start = time.perf_counter()
Wave_obj.forward_solve()
elapsed = time.perf_counter() - t_start
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print(f"Case: {TAG}")
print(f"Elapsed time: {elapsed:.2f}s")
print(f"Memory: {mem_mb:.2f} MB")

# ===========================================================================
# Save data
# ===========================================================================
p_spyro = np.asarray(Wave_obj.forward_solution_receivers)[:, 0]
u_all = communicate(np.asarray(Wave_obj.solid_receiver_history), Wave_obj.comm)
u_solid = u_all[:, 0, :]

n_procs = Wave_obj.comm.comm.size
rank = Wave_obj.comm.comm.rank

if n_procs == 1:
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

# ===========================================================================
# Plots (2D only)
# ===========================================================================
if DIMENSION == 2 and n_procs == 1:
    plot_model_xz(Wave_obj, f"{OUT}/model_{TAG}.png")
    plot_mesh_xz([(Wave_obj.mesh, None)], "Parent mesh", f"{OUT}/mesh_parent_{TAG}.png")
    plot_mesh_xz([(Wave_obj.submesh_fluid, "blue"), (Wave_obj.submesh_solid, "orange")],
                 "Child submeshes", f"{OUT}/mesh_child_{TAG}.png")