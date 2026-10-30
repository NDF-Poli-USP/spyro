"""Automatic mesh loop on a Marmousi2 window: CPW, velocity gradient or adjoint.

The user gives only the tolerance on the relative trace error; the loop
builds a sequence of meshes until the two-level pilot estimate of the worst
shot falls below ``theta * tol``. The three strategies share the pilot, the
estimate and the stopping rule; they differ in how the next mesh is chosen
(see :mod:`spyro.adaptivity.automatic_loop`). The adjoint strategy obtains
its indicator from spyro's automated adjoint.

Default case (same window and acquisition as the KMV4 calibration study):
KMV4 triangles, 7 Hz Ricker, 2 s, a 2 km x 1.6 km window of Marmousi2 with
0.3 km of absorbing pad at the bottom and sides, three sources at 50 m depth
and 101 receivers at 100 m depth. The trace error is measured after the sea
floor reflection (window of the calibration study).

The velocity is read from an ``.npz`` file with ``vp`` (km/s, indexed
``[ix, idepth]``), ``x_origin`` and ``h`` (km), as written by the calibration
study. Values outside the file are extended from its edges.

Usage::

    mpiexec -n 4 python marmousi_automatic_loop.py --model marmousi_window.npz \\
        --strategy adjoint --tol 0.05 --output results/adjoint_tol0p05

Outputs ``report_<strategy>.json`` and the meshes in ``--output``. Positions
in the meshes and reports are local: ``x`` is measured from the left end of
the pad (8.7 km of Marmousi2 by default), because spyro requires
``0 <= x <= length_x``.

Notes
-----
Boundary conditions are spyro's: first-order non-reflecting conditions on the
bottom and sides (``abc_type: "nrbc"``) and a natural (Neumann) top boundary.
The calibration study used a damping layer and a pressure-free top instead,
so absolute errors are not comparable between the two codes; the comparison
between strategies is.

AI assistance: implementation by Claude (Anthropic), reviewed by the author.
"""

from __future__ import annotations

import argparse

import numpy as np

from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.adaptivity import (RectangularDomain, TwoLevelPilot, VelocityGrid)
from spyro.adaptivity.automatic_loop import (AutomaticMeshAdaptation,
                                             LoopParameters)


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="velocity window (.npz)")
    parser.add_argument("--strategy", choices=["cpw", "gradient", "adjoint"],
                        required=True)
    parser.add_argument("--tol", type=float, required=True,
                        help="relative trace-error tolerance, e.g. 0.05")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--initial-cpw", default="auto")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--growth-min", type=float, default=1.15)
    parser.add_argument("--growth-safety", type=float, default=1.0)
    parser.add_argument("--frequency", type=float, default=7.0, help="Hz")
    parser.add_argument("--final-time", type=float, default=2.0, help="s")
    parser.add_argument("--degree", type=int, default=4, help="KMV degree")
    parser.add_argument("--pad", type=float, default=0.3, help="km")
    parser.add_argument("--x-range", type=float, nargs=2, default=(9.0, 11.0))
    parser.add_argument("--depth", type=float, default=1.6, help="km, below z = 0")
    parser.add_argument("--conforming-depths", type=float, nargs="*",
                        default=[-0.45125], help="horizontal interfaces (z, km)")
    parser.add_argument("--water-depth", type=float, default=0.46,
                        help="km, for the post-sea-floor window")
    parser.add_argument("--window-delay", type=float, default=0.12,
                        help="s after the sea-floor reflection; negative for full traces")
    parser.add_argument("--dt-fraction", type=float, default=0.5)
    parser.add_argument("--accumulate-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def velocity_from_npz(path: str, offset: float) -> VelocityGrid:
    """Build the velocity grid from the calibration-study file.

    Parameters
    ----------
    path : str
        ``.npz`` with ``vp[ix, idepth]`` (km/s), ``x_origin`` and ``h``.
    offset : float
        Position subtracted from ``x`` (km), so that the domain starts at 0.

    Returns
    -------
    VelocityGrid
        The velocity in spyro's ``(z, x)`` coordinates.
    """
    data = np.load(path)
    vp = data["vp"].astype(float)
    if vp.max() > 100.0:
        vp = vp / 1000.0
    h = float(data["h"])
    n_depth = vp.shape[1]
    # Rows of the grid go upwards: the deepest sample first.
    return VelocityGrid(vp[:, ::-1].T, z0=-(n_depth - 1) * h,
                        x0=float(data["x_origin"]) - offset, dz=h, dx=h)


def main() -> None:
    """Run the loop with the command-line settings.

    Returns
    -------
    None
        The report and meshes are written to the output directory.
    """
    args = parse_arguments()
    # spyro expects 0 <= x <= length_x, so the window is shifted to start at
    # the left end of the pad; meshes and reports use these local positions.
    offset = args.x_range[0] - args.pad
    x0, x1 = args.pad, args.x_range[1] - offset
    domain = RectangularDomain(z_min=-args.depth - args.pad, x_min=0.0,
                               x_max=x1 + args.pad, pad=args.pad)
    velocity = velocity_from_npz(args.model, offset)
    sources = [(-0.05, x) for x in (x0 + 0.125 * (x1 - x0), 0.5 * (x0 + x1),
                                    x0 + 0.875 * (x1 - x0))]
    receivers = [(-0.10, x) for x in np.linspace(x0, x1, 101)]
    delay = 1.5 * np.sqrt(6.0) / (np.pi * args.frequency)

    def window(times: np.ndarray, shot: int) -> np.ndarray:
        """Traces after the sea-floor reflection plus a delay.

        Parameters
        ----------
        times : numpy.ndarray
            Sample times (s).
        shot : int
            Source index.

        Returns
        -------
        numpy.ndarray
            Boolean ``(n_times, n_receivers)``.
        """
        if args.window_delay < 0:
            return np.ones((len(times), len(receivers)), dtype=bool)
        zs, xs = sources[shot]
        zr = receivers[0][0]
        rx = np.array([r[1] for r in receivers])
        path = np.sqrt((rx - xs) ** 2 + (2 * args.water_depth + zs + zr) ** 2)
        start = delay + path / 1.5 + args.window_delay
        return times[:, None] > start[None, :]

    dictionary = {
        "options": {"cell_type": "T", "variant": "lumped",
                    "degree": args.degree, "dimension": 2},
        "parallelism": {"type": "spatial"},
        "mesh": {"length_z": args.depth + args.pad,
                 "length_x": x1 - x0 + 2 * args.pad, "length_y": 0.0,
                 "mesh_type": "user_mesh"},
        "acquisition": {
            "source_type": "ricker", "source_locations": sources,
            "frequency": args.frequency, "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": receivers,
        },
        "absorving_boundary_conditions": {"status": True, "abc_type": "nrbc"},
        "time_axis": {"initial_time": 0.0, "final_time": args.final_time,
                      "dt": 1e-4, "amplitude": 1, "output_frequency": 10**9,
                      "gradient_sampling_frequency": 1},
        "visualization": {"forward_output": False, "forward_output_filename": None,
                          "fwi_velocity_model_output": False,
                          "velocity_model_filename": None, "gradient_output": False,
                          "gradient_filename": None, "adjoint_output": False,
                          "adjoint_filename": None, "debug_output": False},
    }
    pilot = TwoLevelPilot(dictionary, velocity, window=window,
                          dt_fraction=args.dt_fraction,
                          accumulate_every=args.accumulate_every)
    initial_cpw = args.initial_cpw if args.initial_cpw == "auto" else float(args.initial_cpw)
    parameters = LoopParameters(tolerance=args.tol, theta=args.theta,
                                initial_cpw=initial_cpw,
                                max_iterations=args.max_iterations,
                                growth_min=args.growth_min,
                                growth_safety=args.growth_safety, seed=args.seed)
    loop = AutomaticMeshAdaptation(args.strategy, pilot, velocity, domain,
                                   args.frequency, args.degree, args.output,
                                   parameters, args.conforming_depths)
    report = loop.run()
    SpyroEnsemble.print(
        f"accepted: {report['accepted']}  mesh: {report['accepted_mesh']}  "
        f"search work: {report['search_cost']['work']:.3e} DOF-steps")


if __name__ == "__main__":
    main()
