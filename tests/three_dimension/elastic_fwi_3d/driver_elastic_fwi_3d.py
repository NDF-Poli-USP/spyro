"""Checkpointed 3D elastic FWI comparison.

The four arms intentionally match the optimizer/geometry ablations used by
the 2D elastic campaign: latent LMVM with physical H1 regularization, physical
BLMVM with and without H1, and persistent external latent L-BFGS without H1.
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyadjoint import Control, MinimizationProblem, continue_annotation
from pyadjoint import pause_annotation
from pyadjoint.optimization.tao_solver import TAOConvergenceError, TAOSolver

# Direct execution puts this file's directory, rather than the repository
# root, first on sys.path. Prefer the checked-out Spyro over another editable
# installation in the same Firedrake environment.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

import spyro
from spyro.utils.typing import ElasticMaterialParameter


PARAMETERS = (
    ElasticMaterialParameter.P_WAVE_VELOCITY,
    ElasticMaterialParameter.S_WAVE_VELOCITY,
)
METHODS = {"lmvm_h1", "blmvm_h1", "blmvm", "qn_latent"}
LATENT_METHODS = {"lmvm_h1", "qn_latent"}
H1_METHODS = {"lmvm_h1", "blmvm_h1"}

VP_MIN, VP_MAX = 1.60, 2.10
VS_MIN, VS_MAX = 0.72, 1.02
VP_INITIAL, VS_INITIAL = 1.78, 0.86
RHO = 0.10


def _env(name: str, default, cast):
    return cast(os.environ.get(name, default))


def settings() -> dict:
    smoke = os.environ.get("FWI3D_SMOKE", "0") == "1"
    source_count = _env("FWI3D_SOURCES", 1 if smoke else 8, int)
    grid = int(round(source_count**0.5))
    if grid * grid != source_count:
        raise ValueError("FWI3D_SOURCES must be a perfect square (1, 4, 9, ...).")
    method = os.environ.get("FWI3D_METHOD", "qn_latent").strip().lower()
    if method not in METHODS:
        raise ValueError(f"FWI3D_METHOD must be one of {sorted(METHODS)}")
    return {
        "method": method,
        "smoke": smoke,
        "sources": source_count,
        "source_grid": grid,
        "receivers_per_axis": _env("FWI3D_RECEIVERS_PER_AXIS", 3 if smoke else 9, int),
        "edge": _env("FWI3D_EDGE", 0.25 if smoke else 0.10, float),
        "degree": _env("FWI3D_DEGREE", 2 if smoke else 3, int),
        "dt": _env("FWI3D_DT", 0.002 if smoke else 0.001, float),
        "final_time": _env("FWI3D_FINAL_TIME", 0.30 if smoke else 1.0, float),
        "frequency": _env("FWI3D_FREQUENCY", 4.0 if smoke else 5.0, float),
        "snapshots": _env("FWI3D_CHECKPOINT_SNAPSHOTS", 8 if smoke else 32, int),
        "gc_frequency": _env("FWI3D_GC_TIMESTEP_FREQUENCY", 25 if smoke else 50, int),
        "max_iterations": _env("FWI3D_MAX_ITERATIONS", 2 if smoke else 50, int),
        "gradient_rtol": _env("FWI3D_GRADIENT_RTOL", 1.0e-4, float),
        "h1_weight": _env("FWI3D_H1_WEIGHT", 4.0e-6, float),
        "qn_memory": _env("FWI3D_QN_MEMORY", 10, int),
        "qn_shift": _env("FWI3D_QN_SHIFT", 1.0e-2, float),
        "qn_cap": _env("FWI3D_QN_LATENT_CAP", 12.0, float),
        "results_root": Path(os.environ.get(
            "FWI3D_RESULTS_ROOT", "fwi_3d_results/elastic_inclusion"
        )).expanduser(),
    }


def _surface_grid(count: int, z: float, inset: float) -> list[tuple[float, ...]]:
    axis = np.linspace(inset, 1.0 - inset, count)
    return [(z, float(x), float(y)) for x in axis for y in axis]


def model_dictionary(config: dict, *, source_count: int | None = None) -> dict:
    count = config["sources"] if source_count is None else source_count
    source_grid = int(round(count**0.5))
    sources = _surface_grid(source_grid, -0.05, 0.18)
    receivers = _surface_grid(config["receivers_per_axis"], -0.05, 0.08)
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": config["degree"],
            "dimension": 3,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 1.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": sources,
            "frequency": config["frequency"],
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "amplitude": np.array([1.0, 0.0, 0.0]),
            "receiver_locations": receivers,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": config["final_time"],
            "dt": config["dt"],
            "output_frequency": 100000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
        "absorving_boundary_conditions": {
            "status": True,
            "abc_type": "nrbc",
        },
        "synthetic_data": {
            "type": "object",
            "density": RHO,
            "p_wave_velocity": VP_INITIAL,
            "s_wave_velocity": VS_INITIAL,
            "real_velocity_file": None,
        },
    }


def truth_expressions(mesh):
    """Co-located volumetric Vp/Vs anomaly in a weak depth gradient."""
    z, x, y = fire.SpatialCoordinate(mesh)
    depth = -z
    radius_sq = (x - 0.55) ** 2 + (y - 0.50) ** 2 + (depth - 0.55) ** 2
    anomaly = fire.exp(-radius_sq / (2.0 * 0.12**2))
    vp = 1.72 + 0.12 * depth + 0.20 * anomaly
    vs = 0.82 + 0.06 * depth + 0.10 * anomaly
    return vp, vs


def _make_wave(config: dict, *, truth: bool):
    wave = spyro.IsotropicWave(model_dictionary(config))
    wave.set_mesh(input_mesh_parameters={"edge_length": config["edge"]})
    wave.initialize_physical_parameters()
    if truth:
        vp, vs = truth_expressions(wave.mesh)
        wave.c.interpolate(vp)
        wave.c_s.interpolate(vs)
    else:
        wave.c.assign(VP_INITIAL)
        wave.c_s.assign(VS_INITIAL)
    return wave


def _observed(config: dict):
    wave = _make_wave(config, truth=True)
    wave.forward_solve()
    data = np.asarray(wave.forward_solution_receivers, dtype=float).copy()
    local_energy = config["dt"] * float(np.sum(data**2))
    energy = wave.comm.ensemble_comm.allreduce(local_energy, op=MPI.SUM)
    return data, float(energy)


def _sigmoid(psi, lower: float, upper: float):
    return lower + (upper - lower) / (1.0 + fire.exp(-psi))


def _logit_field(field, lower: float, upper: float, name: str):
    ratio = (field - lower) / (upper - lower)
    ratio = fire.max_value(1.0e-9, fire.min_value(1.0 - 1.0e-9, ratio))
    return fire.Function(field.function_space(), name=name).interpolate(
        fire.ln(ratio / (1.0 - ratio))
    )


def _controls(wave, method: str):
    if method not in LATENT_METHODS:
        return [wave.c, wave.c_s]
    controls = [
        _logit_field(wave.c, VP_MIN, VP_MAX, "psi_vp"),
        _logit_field(wave.c_s, VS_MIN, VS_MAX, "psi_vs"),
    ]
    wave.c = _sigmoid(controls[0], VP_MIN, VP_MAX)
    wave.c_s = _sigmoid(controls[1], VS_MIN, VS_MAX)
    wave.mu = wave.rho * wave.c_s**2
    wave.lmbda = wave.rho * wave.c**2 - 2.0 * wave.mu
    return controls


def _physical(controls, method: str):
    if method in LATENT_METHODS:
        return (
            _sigmoid(controls[0], VP_MIN, VP_MAX),
            _sigmoid(controls[1], VS_MIN, VS_MAX),
        )
    return tuple(controls)


def _field_metrics(controls, method: str) -> dict:
    vp, vs = _physical(controls, method)
    vp_true, vs_true = truth_expressions(controls[0].function_space().mesh())

    def relative_error(field, truth):
        numerator = float(fire.assemble((field - truth) ** 2 * fire.dx))
        denominator = float(fire.assemble(truth**2 * fire.dx))
        return math.sqrt(numerator / denominator)

    return {
        "vp_error": relative_error(vp, vp_true),
        "vs_error": relative_error(vs, vs_true),
    }


class Tracker:
    def __init__(self, directory: Path, config: dict):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.rows = []
        self.objective_evaluations = 0
        self.gradient_evaluations = 0
        self.started = time.time()

    @property
    def root(self):
        return MPI.COMM_WORLD.rank == 0

    def objective_done(self, *_args):
        self.objective_evaluations += 1

    def gradient_done(self, _functional, derivatives, _controls):
        self.gradient_evaluations += 1
        return derivatives

    def record(self, iteration: int, objective: float, gradient_norm: float, controls):
        row = {
            "iteration": int(iteration),
            "objective": float(objective),
            "gradient_norm": float(gradient_norm),
            "objective_evaluations": self.objective_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "wall_time_s": time.time() - self.started,
            **_field_metrics(controls, self.config["method"]),
        }
        self.rows.append(row)
        self.flush()

    def flush(self):
        if not self.root or not self.rows:
            return
        path = self.directory / "convergence.csv"
        with path.open("w", newline="", encoding="ascii") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def _h1_term(controls, method: str, config: dict):
    if method not in H1_METHODS:
        return 0.0
    vp, vs = _physical(controls, method)
    density = (
        fire.inner(fire.grad(vp), fire.grad(vp)) / (VP_MAX - VP_MIN) ** 2
        + fire.inner(fire.grad(vs), fire.grad(vs)) / (VS_MAX - VS_MIN) ** 2
    )
    return fire.assemble(
        0.5 * config["h1_weight"] * density
        * fire.dx(degree=2 * config["degree"] + 4)
    )


def _build_reduced(config: dict, observed, data_energy: float, tracker: Tracker):
    wave = _make_wave(config, truth=False)
    wave.real_shot_record = observed
    # Initialize the public adjoint machinery before replacing physical fields
    # by latent UFL expressions. The tape controls are replaced immediately
    # below for the latent arms.
    wave.enable_automated_adjoint(
        control_parameters=PARAMETERS,
        checkpointing=True,
        snapshots=config["snapshots"],
        gc_timestep_frequency=config["gc_frequency"],
    )
    controls = _controls(wave, config["method"])
    wave.automated_adjoint.controls = controls
    wave.automated_adjoint.control_parameter_names = list(PARAMETERS)
    wave.forward_solve()

    fire_ad.set_working_tape(wave.automated_adjoint._tape)
    continue_annotation()
    functional = wave.functional_value / max(data_energy, 1.0e-30)
    # Every ensemble member records the same regularizer; divide before the
    # EnsembleReducedFunctional sums local functionals over sources.
    functional += _h1_term(controls, config["method"], config) / max(
        1, wave.comm.ensemble_comm.size
    )
    pause_annotation()
    reduced = fire_ad.EnsembleReducedFunctional(
        functional,
        [Control(control) for control in controls],
        wave.comm,
        scatter_control=True,
        tape=wave.automated_adjoint._tape,
        eval_cb_post=tracker.objective_done,
        derivative_cb_post=tracker.gradient_done,
    )
    wave.automated_adjoint.reduced_functional = reduced
    return wave, controls, reduced


def _tao_bounds(controls, method: str):
    if method in LATENT_METHODS:
        return None
    limits = ((VP_MIN, VP_MAX), (VS_MIN, VS_MAX))
    return [
        (
            fire.Function(control.function_space()).assign(lower),
            fire.Function(control.function_space()).assign(upper),
        )
        for control, (lower, upper) in zip(controls, limits)
    ]


def _solve_tao(reduced, controls, config: dict, tracker: Tracker):
    problem = MinimizationProblem(
        reduced, bounds=_tao_bounds(controls, config["method"])
    )
    solver = TAOSolver(problem, parameters={
        "tao_type": "lmvm" if config["method"] in LATENT_METHODS else "blmvm",
        "tao_max_it": config["max_iterations"],
        "tao_gatol": 0.0,
        "tao_grtol": config["gradient_rtol"],
        "tao_gttol": 0.0,
        "tao_monitor": None,
    })

    work = tuple(control.copy(deepcopy=True) for control in controls)

    def monitor(tao):
        iteration, objective, gradient_norm, *_rest = tao.getSolutionStatus()
        solver._vec_interface.from_petsc(tao.getSolution(), work)
        tracker.record(
            iteration,
            objective,
            gradient_norm,
            work,
        )

    solver.tao.setMonitor(monitor)
    termination = "converged"
    try:
        result = solver.solve()
        result = list(result) if isinstance(result, (list, tuple)) else [result]
    except (TAOConvergenceError, PETSc.Error) as exc:
        termination = f"{type(exc).__name__}: {exc}"
        solver._vec_interface.from_petsc(solver.x, work)
        result = list(work)
    for control, value in zip(controls, result):
        control.assign(value)
    return controls, termination


class LBFGS:
    def __init__(self, controls, memory: int):
        self.controls = controls
        self.memory = memory
        self.pairs = []
        self.scale = 1.0

    def copy(self, values):
        return [value.copy(deepcopy=True) for value in values]

    def combine(self, *terms):
        result = [fire.Function(c.function_space()).assign(0.0) for c in self.controls]
        for coefficient, values in terms:
            for component, value in zip(result, values):
                component += float(coefficient) * value
        return result

    def inner(self, left, right):
        return sum(float(fire.assemble(a * b * fire.dx)) for a, b in zip(left, right))

    def apply(self, vector):
        """Apply the current limited-memory inverse Hessian."""
        q = self.copy(vector)
        alpha = []
        for step, delta, rho in reversed(self.pairs):
            value = rho * self.inner(step, q)
            alpha.append(value)
            q = self.combine((1.0, q), (-value, delta))
        result = self.combine((self.scale, q))
        for (step, delta, rho), value in zip(self.pairs, reversed(alpha)):
            beta = rho * self.inner(delta, result)
            result = self.combine((1.0, result), (value - beta, step))
        return result

    def direction(self, gradient, shift: float):
        """Solve ``(I + shift H) s = -H g`` by matrix-free CG."""
        h_gradient = self.apply(gradient)
        rhs = self.combine((-1.0, h_gradient))
        solution = self.combine((0.0, gradient))
        residual = self.copy(rhs)
        direction = self.copy(residual)
        initial_sq = self.inner(residual, residual)
        residual_sq = initial_sq
        if initial_sq == 0.0:
            return solution
        for _iteration in range(30):
            operator_direction = self.combine(
                (1.0, direction), (shift, self.apply(direction))
            )
            curvature = self.inner(direction, operator_direction)
            if not math.isfinite(curvature) or curvature <= 0.0:
                break
            coefficient = residual_sq / curvature
            solution = self.combine(
                (1.0, solution), (coefficient, direction)
            )
            residual = self.combine(
                (1.0, residual), (-coefficient, operator_direction)
            )
            next_sq = self.inner(residual, residual)
            if math.sqrt(next_sq / initial_sq) <= 1.0e-8:
                break
            direction = self.combine(
                (1.0, residual), (next_sq / residual_sq, direction)
            )
            residual_sq = next_sq
        return solution

    def update(self, step, delta):
        sy = self.inner(step, delta)
        yy = self.inner(delta, delta)
        if sy <= 1.0e-10 * math.sqrt(max(self.inner(step, step) * yy, 0.0)):
            return False
        self.scale = min(max(sy / yy, 1.0e-8), 1.0e8)
        self.pairs.append((self.copy(step), self.copy(delta), 1.0 / sy))
        self.pairs[:] = self.pairs[-self.memory:]
        return True


def _solve_external_qn(reduced, controls, config: dict, tracker: Tracker):
    controls = list(controls)
    history = LBFGS(controls, config["qn_memory"])

    def gradient():
        value = reduced.derivative(apply_riesz=True)
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def capped(values):
        result = history.copy(values)
        for field in result:
            with field.dat.vec as vector:
                vector.array[:] = np.clip(vector.array, -config["qn_cap"], config["qn_cap"])
        return result

    objective = float(reduced(controls))
    current_gradient = gradient()
    initial_norm = math.sqrt(history.inner(current_gradient, current_gradient))
    tracker.record(0, objective, initial_norm, controls)
    termination = "iteration_cap"
    for iteration in range(1, config["max_iterations"] + 1):
        direction = history.direction(current_gradient, config["qn_shift"])
        step_length = 1.0
        accepted = False
        while step_length >= 1.0e-8:
            trial = capped(history.combine((1.0, controls), (step_length, direction)))
            step = history.combine((1.0, trial), (-1.0, controls))
            trial_objective = float(reduced(trial))
            if trial_objective <= objective + 1.0e-4 * history.inner(current_gradient, step):
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            termination = "line_search_failed"
            break
        trial_gradient = gradient()
        delta = history.combine((1.0, trial_gradient), (-1.0, current_gradient))
        history.update(step, delta)
        for control, value in zip(controls, trial):
            control.assign(value)
        objective = trial_objective
        current_gradient = trial_gradient
        norm = math.sqrt(history.inner(current_gradient, current_gradient))
        tracker.record(iteration, objective, norm, controls)
        if norm <= config["gradient_rtol"] * initial_norm:
            termination = "relative_gradient_tolerance"
            break
    return controls, termination


def _save_models(directory: Path, controls, method: str):
    vp, vs = _physical(controls, method)
    mesh = controls[0].function_space().mesh()
    vp_true_expr, vs_true_expr = truth_expressions(mesh)
    fields = [
        fire.Function(controls[0].function_space(), name="vp_true").interpolate(vp_true_expr),
        fire.Function(controls[0].function_space(), name="vs_true").interpolate(vs_true_expr),
        fire.Function(controls[0].function_space(), name="vp_initial").assign(VP_INITIAL),
        fire.Function(controls[0].function_space(), name="vs_initial").assign(VS_INITIAL),
        fire.Function(controls[0].function_space(), name="vp_recovered").interpolate(vp),
        fire.Function(controls[0].function_space(), name="vs_recovered").interpolate(vs),
    ]
    # One source group writes collectively; other ensemble groups hold the
    # same distributed model and only wait at the world barrier.
    if MPI.COMM_WORLD.rank < mesh.comm.size:
        with fire.CheckpointFile(str(directory / "models.h5"), "w", comm=mesh.comm) as chk:
            chk.save_mesh(mesh)
            for field in fields:
                chk.save_function(field, name=field.name())
    MPI.COMM_WORLD.Barrier()


def main():
    config = settings()
    configured_run_id = os.environ.get("FWI3D_RUN_ID")
    run_id = MPI.COMM_WORLD.bcast(
        configured_run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            if MPI.COMM_WORLD.rank == 0 else None
        ),
        root=0,
    )
    directory = config["results_root"] / config["method"] / run_id
    tracker = Tracker(directory, config)
    if tracker.root:
        directory.mkdir(parents=True, exist_ok=True)
        configuration = {
            **config,
            "results_root": str(config["results_root"]),
            "dimension": 3,
            "controls": ["p_wave_velocity", "s_wave_velocity"],
            "checkpointing": True,
            "checkpoint_schedule": "MixedCheckpointSchedule",
            "checkpoint_storage": "RAM",
            "vp_bounds": [VP_MIN, VP_MAX],
            "vs_bounds": [VS_MIN, VS_MAX],
            "vp_initial": VP_INITIAL,
            "vs_initial": VS_INITIAL,
            "density": RHO,
            "truth": "co_located_vp_vs_gaussian_volume",
        }
        (directory / "configuration.json").write_text(
            json.dumps(configuration, indent=2),
            encoding="ascii",
        )
    MPI.COMM_WORLD.Barrier()
    observed, energy = _observed(config)
    wave, controls, reduced = _build_reduced(config, observed, energy, tracker)
    if config["method"] == "qn_latent":
        controls, termination = _solve_external_qn(reduced, controls, config, tracker)
    else:
        controls, termination = _solve_tao(reduced, controls, config, tracker)
    metrics = _field_metrics(controls, config["method"])
    _save_models(directory, controls, config["method"])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    peak_rss_gib = peak_rss / (1024**3 if sys.platform == "darwin" else 1024**2)
    summary = {
        "method": config["method"],
        "termination": termination,
        "iterations": tracker.rows[-1]["iteration"] if tracker.rows else 0,
        "objective_evaluations": tracker.objective_evaluations,
        "gradient_evaluations": tracker.gradient_evaluations,
        "wall_time_s": time.time() - tracker.started,
        "peak_rss_gib_per_rank": peak_rss_gib,
        **metrics,
    }
    if tracker.root:
        (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
        print(json.dumps(summary, indent=2), flush=True)
    wave.automated_adjoint.clear_tape()


if __name__ == "__main__":
    main()
