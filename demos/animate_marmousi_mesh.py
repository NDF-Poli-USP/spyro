"""Prototype: Marmousi mesh adapted to the velocity model with Animate.

Builds a triangular mesh whose edge length follows the cells-per-wavelength
rule ``h = vp / (frequency * cpw)`` by remeshing a uniform rectangle with
Animate/Mmg, and compares it against uniform meshes:

* ``uniform_hmin``: edge length set by the minimum velocity (what
  ``mesh_type="firedrake_mesh"`` produces for the same cpw);
* ``reference``: ``uniform_hmin`` refined by two, used as the reference
  solution for the receiver comparison.

For every mesh it reports the cell count, the minimum angle, the achieved
cells per wavelength and, with ``--forward``, the maximum stable time step
and the relative L2 receiver error against the reference.

Animate needs PETSc built with Mmg (``--download-mmg``). Run, e.g.::

    python demos/animate_marmousi_mesh.py vp_marmousi.segy \\
        --length-z 3.0 --length-x 12.0 --forward
"""

import argparse
import os
import time

import firedrake as fire
import matplotlib
import numpy as np
from firedrake.pyplot import tripcolor, triplot

import spyro
from spyro.io.segy_io import read_segy_velocity_model
from spyro.meshing.animate_based_wrappers import (
    create_animate_velocity_mesh,
    sample_grid_velocity,
)
from spyro.meshing.meshing_parameters import cells_per_wavelength

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def mesh_statistics(
    mesh: fire.MeshGeometry,
    vp_grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    frequency: float,
) -> dict:
    """Compute size and quality statistics of a triangular mesh.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Triangular mesh to evaluate.
    vp_grid : numpy.ndarray
        Velocity grid in spyro's ``(z, x)`` convention.
    bbox : tuple of float
        Grid extent ``(z_min, z_max, x_min, x_max)``.
    frequency : float
        Source frequency in Hz.

    Returns
    -------
    dict
        Number of cells and vertices, minimum angle in degrees, and the 1st
        percentile, median and maximum of the achieved cells per wavelength
        (local wavelength over the longest edge of each cell).
    """
    P0 = fire.FunctionSpace(mesh, "DG", 0)
    vp_cell = sample_grid_velocity(P0, vp_grid, bbox).dat.data_ro

    coords = mesh.coordinates.dat.data_ro
    cell_nodes = mesh.coordinates.function_space().cell_node_list
    xy = coords[cell_nodes]
    edges = np.stack([
        np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1),
        np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1),
        np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1),
    ], axis=1)
    achieved_cpw = vp_cell / frequency / edges.max(axis=1)

    # Law of cosines: the smallest angle is opposite the shortest edge.
    a = edges.min(axis=1)
    b_plus_c = edges.sum(axis=1) - a
    b_times_c = edges.prod(axis=1) / a
    cos_min = (b_plus_c**2 - 2 * b_times_c - a**2) / (2 * b_times_c)
    min_angle = np.arccos(np.clip(cos_min, -1.0, 1.0))
    return {
        "cells": mesh.num_cells(),
        "vertices": mesh.num_vertices(),
        "min_angle_deg": float(np.degrees(min_angle.min())),
        "cpw_p01": float(np.percentile(achieved_cpw, 1)),
        "cpw_median": float(np.median(achieved_cpw)),
        "cpw_max": float(achieved_cpw.max()),
    }


def forward_receivers(
    mesh: fire.MeshGeometry,
    vp_grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    dictionary: dict,
    dt: float | None = None,
) -> tuple[float, np.ndarray | None]:
    """Estimate the stable time step and optionally run a forward shot.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh used by the acoustic solver.
    vp_grid : numpy.ndarray
        Velocity grid in spyro's ``(z, x)`` convention.
    bbox : tuple of float
        Grid extent ``(z_min, z_max, x_min, x_max)``.
    dictionary : dict
        spyro input dictionary of the acoustic wave.
    dt : float, optional
        Time step of the forward run. If None, only the maximum stable time
        step is estimated and no shot is propagated.

    Returns
    -------
    max_dt : float
        Maximum stable time step estimated by spyro on this mesh.
    receivers : numpy.ndarray or None
        Receiver traces of shape ``(n_steps, n_receivers)``, or None when
        ``dt`` is None.
    """
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(user_mesh=mesh)
    vp = sample_grid_velocity(wave.function_space, vp_grid, bbox)
    wave.set_initial_velocity_model(velocity_model_function=vp)
    max_dt = wave.get_and_set_maximum_dt(fraction=0.7)
    if dt is None:
        return max_dt, None
    wave.dt = dt
    wave.forward_solve()
    return max_dt, np.asarray(wave.forward_solution_receivers)


def plot_meshes(
    meshes: dict,
    vp_grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    zoom: tuple[float, float, float, float],
    filename: str,
) -> None:
    """Plot the velocity on each mesh with a zoomed view of its edges.

    Parameters
    ----------
    meshes : dict
        Mapping from label to mesh.
    vp_grid : numpy.ndarray
        Velocity grid in spyro's ``(z, x)`` convention.
    bbox : tuple of float
        Grid extent ``(z_min, z_max, x_min, x_max)``.
    zoom : tuple of float
        Zoom window ``(z_min, z_max, x_min, x_max)``.
    filename : str
        Output image path.
    """
    fig, axes = plt.subplots(
        len(meshes), 2, figsize=(16, 3.4 * len(meshes)),
        gridspec_kw={"width_ratios": [2.6, 1]}, squeeze=False,
    )
    for row, (label, mesh) in zip(axes, meshes.items()):
        # Plot with (x, -z) so the figure reads like a seismic section.
        # The swapped mesh shares the topology, so DG0 data can be copied.
        z, x = fire.SpatialCoordinate(mesh)
        swapped = fire.Mesh(fire.Function(
            fire.VectorFunctionSpace(mesh, "CG", 1)
        ).interpolate(fire.as_vector([x, z])))
        vp = fire.Function(fire.FunctionSpace(swapped, "DG", 0))
        vp.dat.data[:] = sample_grid_velocity(
            fire.FunctionSpace(mesh, "DG", 0), vp_grid, bbox
        ).dat.data_ro
        collection = tripcolor(vp, axes=row[0], cmap="jet")
        triplot(swapped, axes=row[0], interior_kw={"linewidths": 0.15,
                                                   "edgecolors": "k"},
                boundary_kw={"colors": "k"})
        row[0].set_title(f"{label}: {mesh.num_cells()} cells")
        row[0].set_aspect("equal")
        row[0].set_xlabel("x (km)")
        row[0].set_ylabel("z (km)")
        fig.colorbar(collection, ax=row[0], label="vp (km/s)", fraction=0.02)

        tripcolor(vp, axes=row[1], cmap="jet")
        triplot(swapped, axes=row[1], interior_kw={"linewidths": 0.4,
                                                   "edgecolors": "k"},
                boundary_kw={"colors": "k"})
        row[1].set_xlim(zoom[2], zoom[3])
        row[1].set_ylim(zoom[0], zoom[1])
        row[1].set_aspect("equal")
        row[1].set_title("zoom")
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def main() -> None:
    """Run the Animate Marmousi meshing prototype from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("segy", help="velocity model SEG-Y file (km/s)")
    parser.add_argument("--length-z", type=float, required=True)
    parser.add_argument("--length-x", type=float, required=True)
    parser.add_argument("--frequency", type=float, default=5.0)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gradation", type=float, default=1.3)
    parser.add_argument("--cpw-factor", type=float, default=1.0,
                        help="safety factor applied to cpw for the Animate mesh only")
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--output-dir", default="results/animate_marmousi")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    vp_grid, _, _ = read_segy_velocity_model(args.segy)
    bbox = (-args.length_z, 0.0, 0.0, args.length_x)
    cpw = cells_per_wavelength("mass_lumped_triangle", args.degree, 2)
    h_min = float(vp_grid.min()) / (args.frequency * cpw)
    print(f"vp in [{vp_grid.min():.3f}, {vp_grid.max():.3f}] km/s, "
          f"cpw = {cpw}, h_min = {h_min * 1e3:.1f} m, "
          f"Animate cpw = {cpw * args.cpw_factor:.3f}", flush=True)

    start = time.perf_counter()
    adapted = create_animate_velocity_mesh(
        vp_grid, bbox, args.frequency, cpw * args.cpw_factor,
        num_iterations=args.iterations, gradation_factor=args.gradation,
    )
    adapt_time = time.perf_counter() - start

    def uniform(h: float) -> fire.MeshGeometry:
        return spyro.meshing.rectangle_mesh(
            int(np.ceil(args.length_z / h)), int(np.ceil(args.length_x / h)),
            args.length_z, args.length_x,
        )

    meshes = {
        "animate": adapted,
        "uniform_hmin": uniform(h_min),
        "reference": uniform(h_min / 2),
    }
    stats = {label: mesh_statistics(mesh, vp_grid, bbox, args.frequency)
             for label, mesh in meshes.items()}
    stats["animate"]["adapt_time_s"] = adapt_time

    zoom = (-args.length_z, -args.length_z + 1.0,
            0.45 * args.length_x, 0.45 * args.length_x + 2.0)
    figure = os.path.join(args.output_dir, "animate_marmousi_mesh.png")
    plot_meshes({k: meshes[k] for k in ("animate", "uniform_hmin")},
                vp_grid, bbox, zoom, figure)
    print(f"Figure written to {figure}", flush=True)

    if args.forward:
        dictionary = {
            "options": {"cell_type": "T", "variant": "lumped",
                        "degree": args.degree, "dimension": 2},
            "parallelism": {"type": "automatic"},
            "mesh": {"length_z": args.length_z, "length_x": args.length_x,
                     "length_y": 0.0},
            "acquisition": {
                "source_type": "ricker",
                "source_locations": [(-0.01, args.length_x / 2)],
                "frequency": args.frequency,
                "delay": 1.0 / args.frequency,
                "delay_type": "time",
                "receiver_locations": spyro.create_transect(
                    (-0.1, 0.1 * args.length_x), (-0.1, 0.9 * args.length_x), 100
                ),
            },
            "time_axis": {"initial_time": 0.0, "final_time": args.final_time,
                          "dt": 0.001, "amplitude": 1, "output_frequency": 10000,
                          "gradient_sampling_frequency": 10000},
            "visualization": {"forward_output": False},
        }
        max_dt = {label: forward_receivers(mesh, vp_grid, bbox, dictionary)[0]
                  for label, mesh in meshes.items()}
        dt = min(max_dt.values())
        receivers = {}
        for label, mesh in meshes.items():
            start = time.perf_counter()
            _, receivers[label] = forward_receivers(mesh, vp_grid, bbox,
                                                    dictionary, dt=dt)
            stats[label]["max_dt_ms"] = 1e3 * max_dt[label]
            stats[label]["forward_time_s"] = time.perf_counter() - start
        reference = receivers["reference"]
        for label in meshes:
            stats[label]["receiver_rel_l2"] = float(
                np.linalg.norm(receivers[label] - reference)
                / np.linalg.norm(reference)
            )
        np.savez(os.path.join(args.output_dir, "receivers.npz"), **receivers)

    columns = sorted({key for row in stats.values() for key in row})
    print("\n" + " | ".join(["mesh".ljust(13)] + [c.rjust(15) for c in columns]))
    for label, row in stats.items():
        print(" | ".join([label.ljust(13)] + [
            f"{row[c]:15.4g}" if c in row else " " * 15 for c in columns
        ]))


if __name__ == "__main__":
    main()
