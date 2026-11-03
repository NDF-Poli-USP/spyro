"""Real trace error of meshes against a reference record.

The loop accepts a mesh when the two-level estimate (the mesh against its
uniform refinement, both with the time step of the refinement) is below
``theta * tol``. This script measures what the estimate stands for: the
traces of the mesh run as it would be used, with its own time step, against
an independent fine reference, in the same window as the loop.

Cases, for each mesh:

``production``
    the mesh with ``dt_fraction`` of its own critical step;
``pilot_dt``
    the mesh with the time step of its refinement, as in the estimate;
    against ``production`` it isolates the time-step error;
``refined``
    the uniform refinement with its own time step: the level the estimate
    uses as reference, and a check of the agreement between the two codes.

The time step is rounded down so that ``--sample-dt`` is a whole number of
steps, and the traces are taken every ``--sample-dt`` (sample 0 at t = 0).
The reference is a ``.npz`` with ``traces[shot, sample, receiver]`` and its
``dt``, as written by the calibration study (e.g. U5m, 5 m uniform mesh).
The problem options are those of ``marmousi_automatic_loop.py``; use the
same boundaries as the reference (``--damping-reflection 1e-3``).

Usage::

    mpiexec -x PYTHONPATH -n 4 python validate_against_reference.py \\
        --model marmousi_window_s250.npz --damping-reflection 1e-3 \\
        --reference U5m_s0/traces.npz --meshes results/a.msh results/b.msh \\
        --report results/validation.json

AI assistance: implementation by Claude (Anthropic), reviewed by the author.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import firedrake as fire
import numpy as np

from spyro.mpi.spyro_mpi import SpyroEnsemble
from marmousi_automatic_loop import MarmousiProblem, add_problem_arguments, build_problem

CASES = ("production", "pilot_dt", "refined")


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_problem_arguments(parser)
    parser.add_argument("--meshes", nargs="+", required=True,
                        help="gmsh files in spyro's (z, x) convention")
    parser.add_argument("--reference", required=True,
                        help=".npz with traces[shot, sample, receiver] and dt")
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=["production", "pilot_dt"])
    parser.add_argument("--sample-dt", type=float, default=1e-3, help="s")
    parser.add_argument("--report", required=True, help="output .json")
    parser.add_argument("--save-traces", action="store_true",
                        help="also write <report stem>_<mesh>_<case>.npz")
    return parser.parse_args()


def sampled_step(critical: float, fraction: float, sample_dt: float
                 ) -> tuple[float, int]:
    """Largest step not above ``fraction * critical`` that divides the sampling.

    Parameters
    ----------
    critical : float
        Critical time step (s).
    fraction : float
        Fraction of the critical step.
    sample_dt : float
        Sampling interval of the traces (s).

    Returns
    -------
    tuple of (float, int)
        The time step and the number of steps per sample.
    """
    steps_per_sample = int(np.ceil(sample_dt / (fraction * critical)))
    return sample_dt / steps_per_sample, steps_per_sample


def windowed_comparison(traces: np.ndarray, reference: np.ndarray,
                        mask: np.ndarray) -> dict[str, float]:
    """Relative error and diagnostics of one shot in the window.

    Parameters
    ----------
    traces : numpy.ndarray
        ``(samples, receivers)`` of the mesh.
    reference : numpy.ndarray
        ``(samples, receivers)`` of the reference.
    mask : numpy.ndarray
        Boolean window, same shape.

    Returns
    -------
    dict
        ``error`` (:math:`\\|d - d^{ref}\\|_W / \\|d^{ref}\\|_W`),
        ``amplitude_ratio`` (:math:`\\|d\\|_W / \\|d^{ref}\\|_W`) and
        ``correlation`` (cosine between ``d`` and ``d^{ref}`` in the window),
        and the squared norms used by the aggregate error.
    """
    d = np.where(mask, traces, 0.0)
    r = np.where(mask, reference, 0.0)
    difference2 = float(np.sum((d - r) ** 2))
    reference2 = float(np.sum(r ** 2))
    traces2 = float(np.sum(d ** 2))
    return {"error": np.sqrt(difference2 / reference2),
            "amplitude_ratio": np.sqrt(traces2 / reference2),
            "correlation": float(np.sum(d * r)) / np.sqrt(traces2 * reference2),
            "difference2": difference2, "reference2": reference2}


def run_case(problem: MarmousiProblem, mesh: fire.MeshGeometry, dt: float,
             steps_per_sample: int, reference: np.ndarray, sample_dt: float
             ) -> tuple[dict, np.ndarray]:
    """Propagate every shot on a mesh and compare with the reference.

    Parameters
    ----------
    problem : MarmousiProblem
        Pilot and trace window.
    mesh : firedrake.MeshGeometry
        Mesh loaded by the pilot.
    dt : float
        Time step (s).
    steps_per_sample : int
        Steps between two samples.
    reference : numpy.ndarray
        ``(shots, samples, receivers)``.
    sample_dt : float
        Sampling interval (s).

    Returns
    -------
    tuple of (dict, numpy.ndarray)
        Summary of the case and the sampled traces ``(shots, samples,
        receivers)``.
    """
    n_shots, n_samples, _ = reference.shape
    times = np.arange(n_samples) * sample_dt
    traces = np.zeros_like(reference)
    per_shot = []
    seconds = []
    for shot in range(n_shots):
        start = time.perf_counter()
        record = problem.pilot.shot_record(shot, mesh, dt)
        seconds.append(time.perf_counter() - start)
        # Record row n holds t = (n + 1) dt; sample m is t = m sample_dt.
        rows = np.arange(1, n_samples) * steps_per_sample - 1
        if rows[-1] >= record.shape[0]:
            raise RuntimeError(
                f"the record has {record.shape[0]} steps; sample {n_samples - 1} "
                f"needs {rows[-1] + 1}")
        traces[shot, 1:] = record[rows]
        mask = np.asarray(problem.window(times, shot), dtype=bool)
        per_shot.append(windowed_comparison(traces[shot], reference[shot], mask))
    total = np.sqrt(sum(s["difference2"] for s in per_shot)
                    / sum(s["reference2"] for s in per_shot))
    summary = {
        "dofs": fire.FunctionSpace(mesh, "KMV", problem.pilot.dictionary[
            "options"]["degree"]).dim(),
        "dt": dt, "steps_per_sample": steps_per_sample,
        "error_per_shot": [s["error"] for s in per_shot],
        "error_worst": max(s["error"] for s in per_shot),
        "error_total": float(total),
        "amplitude_ratio_per_shot": [s["amplitude_ratio"] for s in per_shot],
        "correlation_per_shot": [s["correlation"] for s in per_shot],
        "wall_seconds_per_shot": seconds,
    }
    return summary, traces


def main() -> None:
    """Validate every mesh and write the report.

    Returns
    -------
    None
        The report (and, if asked, the traces) are written on rank 0.
    """
    args = parse_arguments()
    problem = build_problem(args)
    data = np.load(args.reference)
    reference = np.asarray(data["traces"], dtype=float)
    if abs(float(data["dt"]) - args.sample_dt) > 1e-12:
        raise ValueError(f"reference dt {float(data['dt'])} != --sample-dt")
    if reference.shape[0] != problem.pilot.number_of_shots:
        raise ValueError("the reference and the acquisition differ in shots")
    rank = fire.COMM_WORLD.rank
    report_path = Path(args.report)
    report = {"reference": str(args.reference), "sample_dt": args.sample_dt,
              "dt_fraction": args.dt_fraction,
              "damping_reflection": args.damping_reflection,
              "free_surface": not args.rigid_top, "meshes": {}}

    for mesh_file in args.meshes:
        need_fine = any(case in ("pilot_dt", "refined") for case in args.cases)
        meshes = problem.pilot.load_meshes(mesh_file, levels=1 if need_fine else 0)
        critical = {"mesh": problem.pilot.critical_time_step(meshes[0])}
        if need_fine:
            critical["refined"] = problem.pilot.critical_time_step(meshes[1])
        cases = {"production": (meshes[0], critical["mesh"]),
                 "pilot_dt": (meshes[0], critical.get("refined")),
                 "refined": (meshes[-1], critical.get("refined"))}
        entry = {"critical_dt": critical, "cases": {}}
        for case in args.cases:
            mesh, dt_critical = cases[case]
            dt, steps_per_sample = sampled_step(dt_critical, args.dt_fraction,
                                                args.sample_dt)
            SpyroEnsemble.print(f"{Path(mesh_file).name} [{case}]: dt = "
                                f"{1e3 * dt:.4f} ms")
            summary, traces = run_case(problem, mesh, dt, steps_per_sample,
                                       reference, args.sample_dt)
            entry["cases"][case] = summary
            SpyroEnsemble.print(
                f"  {summary['dofs']} DOFs, worst error "
                f"{100 * summary['error_worst']:.2f}% "
                f"(shots: {', '.join(f'{100 * e:.2f}' for e in summary['error_per_shot'])})")
            if args.save_traces and rank == 0:
                np.savez_compressed(
                    report_path.with_name(
                        f"{report_path.stem}_{Path(mesh_file).stem}_{case}.npz"),
                    traces=traces, dt=args.sample_dt)
        report["meshes"][str(mesh_file)] = entry
        if rank == 0:
            # Written after every mesh, so a long run keeps what is done.
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
