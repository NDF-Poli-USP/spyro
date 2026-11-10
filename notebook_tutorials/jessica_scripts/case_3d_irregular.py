"""
3D irregular interface case for AcousticElasticWave.

Spyro coordinate convention: (z, x, y) where z is negative downward.

Interface geometry (sinusoidal, 3D extension of case_small_2D_irregular):
    z_int(x, y) = -z0 + a * sin(2*pi*x / l) * sin(2*pi*y / l)
    z0 = 1.5 km (mean interface depth), a = 0.05 km, l = 1.5 km

Domain: 3 x 3 x 3 km  (same as case_3D)
Material parameters: same as case_3D (non-homogeneous)
Source/receiver placement: same relative distances as case_3D
  - source  : 0.15 km above interface  (fluid side, z more positive)
  - fluid R : 0.01 km above interface
  - solid R : 0.01 km below interface
  - lateral offset of 0.01 km in x and y to avoid source axis alignment
"""

import os
import time
import resource
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.utils.utils import communicate
from spyro.io.segy_io import create_segy_from_grid

OUT = "results"
os.makedirs(OUT, exist_ok=True)

USE_MONOLITHIC = False
SCHEME = "monolithic" if USE_MONOLITHIC else "sequential"
TAG = f"case_3d_irregular_{SCHEME}_3d"

# ===========================================================================
# Domain and interface parameters
# ===========================================================================
LENGTH_Z = 3.0   # km
LENGTH_X = 3.0   # km
LENGTH_Y = 3.0   # km

Z0 = 1.5    # km — mean interface depth (positive, will be negated for z coord)
A  = 0.05   # km — amplitude
L  = 1.5    # km — spatial period

# Interface: z_int(x, y) = -Z0 + A*sin(2pi*x/L)*sin(2pi*y/L)
# In spyro z is negative downward, so "above interface" means z > z_int.
def interface_z(x, y):
    return -Z0 + A * np.sin(2 * np.pi * x / L) * np.sin(2 * np.pi * y / L)

# Evaluate interface at domain centre for source/receiver placement
x_centre = LENGTH_X / 2   # 1.5 km
y_centre = LENGTH_Y / 2   # 1.5 km
z_int_centre = interface_z(x_centre, y_centre)   # = -1.5 at centre (sin=0)

# Lateral offset to avoid source-receiver alignment (same as case_3D: 0.01 km)
LATERAL = 0.01

z_source      = z_int_centre + 0.15          # fluid side
z_fluid_recv  = z_int_centre + LATERAL       # just above interface
z_solid_recv  = z_int_centre - LATERAL       # just below interface

x_recv = x_centre - LATERAL   # slight lateral offset in x
y_recv = y_centre - LATERAL   # slight lateral offset in y

print(f"Interface z at centre (x={x_centre}, y={y_centre}): {z_int_centre:.4f} km")
print(f"Source:        z={z_source:.4f}  x={x_centre}  y={y_centre}")
print(f"Fluid recv:    z={z_fluid_recv:.4f}  x={x_recv}  y={y_recv}")
print(f"Solid recv:    z={z_solid_recv:.4f}  x={x_recv}  y={y_recv}")

# ===========================================================================
# Build 3D SEG-Y velocity model
# ===========================================================================
VP_WATER = np.sqrt(2.25 / 1.0)   # 1.5 km/s
VP_SOLID = 3.4                    # km/s  (VS not encoded in SEG-Y, set via dictionary)

def build_segy_3d(filename, length_z, length_x, length_y, dz, dx, dy,
                  interface_func, v_fluid, v_solid):
    """
    Build a 3D SEG-Y on a regular (z, x, y) grid.
    Spyro convention: z negative downward.
    Above interface_func(x, y) -> v_fluid; below -> v_solid.
    Stored as inline=ix, crossline=iy, samples along z (depth axis).
    """
    import segyio

    nz = int(round(length_z / dz)) + 1
    nx = int(round(length_x / dx)) + 1
    ny = int(round(length_y / dy)) + 1

    z_arr = np.linspace(-length_z, 0.0, nz)   # negative downward
    x_arr = np.linspace(0.0, length_x, nx)
    y_arr = np.linspace(0.0, length_y, ny)

    # cube[ix, iy, iz]: velocity at that grid point
    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")   # (nx, ny)
    Z_int = interface_func(X, Y)                        # (nx, ny)

    # Assign velocity: above interface (z > z_int) -> fluid, below -> solid
    cube = np.where(
        z_arr[np.newaxis, np.newaxis, :] > Z_int[:, :, np.newaxis],
        v_fluid, v_solid
    ).astype(np.float32)   # shape (nx, ny, nz)

    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format  = segyio.SegySampleFormat.IBM_FLOAT_4_BYTE
    spec.samples = np.abs(z_arr).astype(np.float32)
    spec.ilines  = np.arange(1, nx + 1)
    spec.xlines  = np.arange(1, ny + 1)

    with segyio.create(str(filename), spec) as f:
        f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)
        for ix in range(nx):
            for iy in range(ny):
                tr_idx = ix * ny + iy
                f.header[tr_idx] = {
                    segyio.TraceField.INLINE_3D:    ix + 1,
                    segyio.TraceField.CROSSLINE_3D: iy + 1,
                }
                f.trace[tr_idx] = cube[ix, iy, :]

    print(f"[OK] 3D SEG-Y written: {filename}  ({nx}x x {ny}y x {nz}z samples)")

DZ = 0.025
DX = 0.025
DY = 0.025
NZ = int(round(LENGTH_Z / DZ)) + 1
NX = int(round(LENGTH_X / DX)) + 1
NY = int(round(LENGTH_Y / DY)) + 1

bin_file = f"{OUT}/model_{TAG}.bin"

def build_binary_3d(filename, nz, nx, ny, dz, dx, dy,
                    length_z, length_x, length_y,
                    interface_func, v_fluid, v_solid):
    """
    Write a raw binary velocity model in canonical (nz, nx, ny) order,
    big-endian float32, as expected by read_velocity_binary3d.
    z is negative downward; above interface_func(x,y) -> v_fluid, below -> v_solid.
    """
    z_arr = np.linspace(-length_z, 0.0, nz)   # negative downward
    x_arr = np.linspace(0.0, length_x, nx)
    y_arr = np.linspace(0.0, length_y, ny)

    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")   # (nx, ny)
    Z_int = interface_func(X, Y)                        # (nx, ny)

    # cube[iz, ix, iy] = velocity
    cube = np.where(
        z_arr[:, np.newaxis, np.newaxis] > Z_int[np.newaxis, :, :],
        v_fluid, v_solid
    ).astype(">f4")   # big-endian float32, shape (nz, nx, ny)

    cube.tofile(filename)
    print(f"[OK] Binary model written: {filename}  ({nz}z x {nx}x x {ny}y)")

build_binary_3d(
    bin_file, NZ, NX, NY, DZ, DX, DY,
    LENGTH_Z, LENGTH_X, LENGTH_Y,
    interface_z, VP_WATER, VP_SOLID,
)

# ===========================================================================
# Dictionary
# ===========================================================================
dictionary = {}

dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 3,
    "dimension": 3,
}

dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": LENGTH_Z,
    "length_x": LENGTH_X,
    "length_y": LENGTH_Y,
    "mesh_file": None,
    "mesh_type": "gmsh_mesh",
    "output_filename": f"{OUT}/mesh_{TAG}.msh",
    "segy_velocity_model": bin_file,
    "cells_per_wavelength": 4.0,   # >= 3.72 required for degree-3 tetrahedra (KMV)
    "frequency": 10,
    "min_element_size": 0.025,
    "grade": 0.05,
    "structured_mesh": False,
    "apply_winslow": False,
    "winslow_iterations": 2000,
    "vp_water": VP_WATER,
    "water_interface": False,
    "water_search_value": VP_WATER,
    "segy_nz": NZ,
    "segy_nx": NX,
    "segy_ny": NY,
    "segy_dz": DZ,
    "segy_dx": DX,
    "segy_dy": DY,
    "padding_x": 0.3,
    "padding_y": 0.3,
    "padding_z": 0.3,
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations":         [(z_source,     x_centre, y_centre)],
    "frequency": 10,
    "delay": 1.0 / 10,
    "delay_type": "time",
    "amplitude": 1.0,
    "receiver_locations":       [(z_fluid_recv, x_recv,   y_recv)],
    "solid_receiver_locations": [(z_solid_recv, x_recv,   y_recv)],
    "use_vertex_only_mesh": False,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 0.01,
    "dt": 2.8e-5,
    "output_frequency": 100,
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
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
    "real_velocity_file": None,
    "epsilon": None,
    "delta": None,
    "gamma": None,
    "anisotropy_type": None,
}

dictionary["absorving_boundary_conditions"] = {
    "status": False,
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
Wave_obj.use_monolithic = USE_MONOLITHIC

t_start = time.perf_counter()
Wave_obj.forward_solve()
elapsed = time.perf_counter() - t_start
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print(f"Case: {TAG}")
print(f"Elapsed time: {elapsed:.2f}s")
print(f"Memory:       {mem_mb:.2f} MB")

# ===========================================================================
# Save data
# ===========================================================================
p_spyro = np.asarray(Wave_obj.forward_solution_receivers)[:, 0]
u_all   = communicate(np.asarray(Wave_obj.solid_receiver_history), Wave_obj.comm)
u_solid = u_all[:, 0, :]

n_procs = Wave_obj.comm.comm.size
rank    = Wave_obj.comm.comm.rank

if rank == 0:
    np.savez(
        f"{OUT}/spyro_receiver_data_{TAG}.npz",
        dt=dictionary["time_axis"]["dt"],
        final_time=dictionary["time_axis"]["final_time"],
        p_spyro=p_spyro,
        u_solid=u_solid,
    )
    print(f"[OK] Saved: {OUT}/spyro_receiver_data_{TAG}.npz")