"""Compare production and reference meshes for the 3D elastic benchmark."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from driver_elastic_fwi_3d import _observed, settings


def _edge_config(base: dict, edge: float) -> dict:
    config = dict(base)
    config.update({
        "sources": 1,
        "source_grid": [1, 1],
        "edge": edge,
        "minimum_cells_per_wavelength": (
            0.72 / (config["frequency"] * edge)
        ),
    })
    config["minimum_gll_intervals_per_wavelength"] = (
        config["degree"] * config["minimum_cells_per_wavelength"]
    )
    return config


def main():
    if MPI.COMM_WORLD.size != 4:
        raise RuntimeError("The mesh gate must run with four spatial MPI ranks.")

    started = time.time()
    base = settings()
    candidate_edge = float(os.environ.get("FWI3D_MESH_GATE_EDGE", "0.05"))
    reference_edge = float(os.environ.get("FWI3D_MESH_GATE_REFERENCE_EDGE", "0.04"))
    tolerance = float(os.environ.get("FWI3D_MESH_GATE_RTOL", "0.05"))

    candidate, candidate_energy = _observed(
        _edge_config(base, candidate_edge)
    )
    reference, reference_energy = _observed(
        _edge_config(base, reference_edge)
    )
    difference = np.linalg.norm(candidate - reference)
    relative_trace_error = float(
        difference / max(np.linalg.norm(reference), 1.0e-30)
    )

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gib = peak_rss / (1024**3 if sys.platform == "darwin" else 1024**2)
    peak_rss_gib = MPI.COMM_WORLD.allreduce(peak_rss_gib, op=MPI.MAX)
    wall_time = MPI.COMM_WORLD.allreduce(time.time() - started, op=MPI.MAX)
    report = {
        "element": "structured_extruded_spectral",
        "degree": base["degree"],
        "candidate_edge_km": candidate_edge,
        "reference_edge_km": reference_edge,
        "sources": 1,
        "receivers": base["receivers_per_axis"] ** 2,
        "dt_s": base["dt"],
        "final_time_s": base["final_time"],
        "candidate_energy": candidate_energy,
        "reference_energy": reference_energy,
        "relative_trace_error": relative_trace_error,
        "tolerance": tolerance,
        "passed": relative_trace_error <= tolerance,
        "wall_time_s": wall_time,
        "peak_rss_gib": peak_rss_gib,
    }

    output = Path(os.environ.get(
        "FWI3D_MESH_GATE_OUTPUT", "spectral_mesh_gate.json"
    ))
    if MPI.COMM_WORLD.rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="ascii")
        print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError(
            "Production mesh failed the trace comparison: "
            f"{relative_trace_error:.3e} > {tolerance:.3e}."
        )


if __name__ == "__main__":
    main()
