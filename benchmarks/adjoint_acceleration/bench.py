"""Measure the automated adjoint of spyro's acoustic solver.

Run once per configuration in a fresh process (peak RSS is per process):

    PYTHONPATH=<spyro tree> python bench.py --schedule mixed --snapshots 10 \
        --label new --out results/new_mixed10.npz

Records receivers, functional, gradient, timings, peak RSS, checkpoint bytes
held during the reverse sweep, recomputed steps and the content of the
``checkpointable_state`` / ``adjoint_dependencies`` sets of the tape.
"""
import argparse
import gc
import json
import resource
import sys
import time
from collections import Counter

import numpy as np

import firedrake as fire
from firedrake.adjoint_utils.checkpointing import DelegatedFunctionCheckpoint
import pyadjoint
from pyadjoint.checkpointing import CheckpointManager
from checkpoint_schedules import (
    Forward, Reverse, Copy, Move, MixedCheckpointSchedule, Revolve,
    SingleMemoryStorageSchedule, StorageType,
)

import spyro
from spyro.utils.typing import AdjointType, AbsorbingBCsType
from spyro.solvers import automatic_differentiation_solver as ads


def build_dictionary(final_time, dt, abc, edge_length):
    d = {
        "options": {"cell_type": "T", "variant": "lumped", "degree": 2,
                    "dimension": 2},
        "parallelism": {"type": "automatic"},
        "mesh": {"length_z": 1.0, "length_x": 1.0, "length_y": 0.0,
                 "mesh_file": None, "mesh_type": "firedrake_mesh"},
        "acquisition": {
            "source_type": "ricker", "source_locations": [(-0.1, 0.5)],
            "frequency": 5.0, "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": spyro.create_transect(
                (-0.8, 0.1), (-0.8, 0.9), 5),
        },
        "time_axis": {"initial_time": 0.0, "final_time": final_time,
                      "dt": dt, "amplitude": 1, "output_frequency": 100000,
                      "gradient_sampling_frequency": 1},
        "visualization": {"forward_output": False,
                          "forward_output_filename": None,
                          "fwi_velocity_model_output": False,
                          "velocity_model_filename": None,
                          "gradient_output": False, "gradient_filename": None,
                          "adjoint_output": False, "adjoint_filename": None,
                          "debug_output": False},
    }
    if abc == "nrbc":
        d["absorving_boundary_conditions"] = {
            "status": True, "abc_type": "nrbc",
            "absorb_top": True, "absorb_bottom": True,
            "absorb_right": True, "absorb_left": True,
        }
    return d


def make_wave(args):
    wave = spyro.AcousticWave(dictionary=build_dictionary(
        args.final_time, args.dt, args.abc, args.edge_length))
    wave.set_mesh(input_mesh_parameters={"edge_length": args.edge_length})
    return wave


def true_record(args):
    wave = make_wave(args)
    wave.set_initial_velocity_model(
        conditional=fire.conditional(wave.mesh_z > -0.5, 1.5, 3.5),
        dg_velocity_model=False)
    wave.forward_solve()
    return wave.forward_solution_receivers


def guess_velocity(wave, shift=0.0):
    # Heterogeneous guess: smooth vertical gradient plus a lateral bump.
    z, x = wave.mesh_z, wave.mesh_x
    return 1.8 + shift - 0.6 * z + 0.2 * fire.exp(-((x - 0.5) ** 2) / 0.02)


# ---------------------------------------------------------------- accounting
def _category(obj):
    try:
        mesh = obj.function_space().mesh()
    except AttributeError:
        return "scalar"
    if "vom" in mesh.name:
        return "receiver"
    return "field"


def checkpoint_buffers(obj):
    """(address, nbytes, category) of the storage a checkpoint object owns."""
    if obj is None or isinstance(obj, DelegatedFunctionCheckpoint):
        return []
    if isinstance(obj, (float, int, pyadjoint.AdjFloat)):
        return [(id(obj), 8, "scalar")]
    dat = getattr(obj, "dat", None)
    if dat is None:
        return []
    arr = dat.data_ro_with_halos
    return [(arr.__array_interface__["data"][0], arr.nbytes, _category(obj))]


_BV_CACHE = {}


def _tape_block_variables(tape):
    # The blocks do not change once recording has finished.
    key = (id(tape), len(tape.get_blocks()))
    if key not in _BV_CACHE:
        bvs = {}
        for block in tape.get_blocks():
            for bv in list(block.get_dependencies()) + list(block.get_outputs()):
                bvs[id(bv)] = bv
        _BV_CACHE.clear()
        _BV_CACHE[key] = list(bvs.values())
    return _BV_CACHE[key]


def live_checkpoint_bytes(tape, by_category=False):
    """Bytes held by checkpoints on the tape, counting each buffer once."""
    seen = {}
    for step in tape.timesteps:
        for chk in step._checkpoint.values():
            for addr, nb, cat in checkpoint_buffers(chk):
                seen[addr] = (nb, cat)
    for bv in _tape_block_variables(tape):
        for addr, nb, cat in checkpoint_buffers(bv._checkpoint):
            seen[addr] = (nb, cat)
    if by_category:
        out = Counter()
        for nb, cat in seen.values():
            out[cat] += nb
        return dict(out)
    return sum(nb for nb, _ in seen.values())


def describe_bv(bv, creators):
    out = bv.output
    name = getattr(out, "name", lambda: type(out).__name__)
    try:
        name = name()
    except TypeError:
        name = str(name)
    kind = type(out).__name__
    chk = bv._checkpoint
    if isinstance(chk, DelegatedFunctionCheckpoint):
        storage = "delegated"
    elif chk is None:
        storage = "none"
    else:
        storage = "own copy"
    creator = creators.get(id(bv), "input")
    cat = _category(out)
    if cat != "field":
        name = "*"
    return f"{cat} {kind}:{name} [{creator}; bv chk={storage}]"


def inspect_tape(tape, label):
    creators = {}
    for block in tape.get_blocks():
        for out in block.get_outputs():
            creators[id(out)] = type(block).__name__
    steps = tape.timesteps
    report = {"label": label, "n_timesteps": len(steps),
              "n_blocks": len(tape.get_blocks())}
    if len(steps) < 3:
        return report
    k = len(steps) // 2
    step = steps[k]
    report["mid_step"] = k
    report["mid_blocks"] = [str(type(b).__name__) for b in step]
    report["checkpointable_state"] = dict(sorted(Counter(
        describe_bv(bv, creators) for bv in step.checkpointable_state).items()))
    report["adjoint_dependencies"] = dict(sorted(Counter(
        describe_bv(bv, creators) for bv in step.adjoint_dependencies).items()))
    # What the step checkpoint dictionary holds, and whether the entries are
    # separate copies of the same values.
    own = [checkpoint_buffers(c) for c in step._checkpoint.values()]
    report["step_checkpoint_entries"] = len(step._checkpoint)
    report["step_checkpoint_bytes"] = int(sum(nb for b in own for _, nb, _ in b))
    report["live_bytes_by_category"] = live_checkpoint_bytes(tape, True)
    # Census of the per-step checkpoint dictionaries: what one stored step
    # costs, by category, counting each buffer it owns.
    census = Counter()
    for st in steps:
        if not st._checkpoint:
            continue
        cats = Counter()
        for c in st._checkpoint.values():
            for _, nb, cat in checkpoint_buffers(c):
                cats[cat] += nb
        census[json.dumps(dict(sorted(cats.items())))] += 1
    report["step_checkpoint_census"] = dict(census)
    return report


# ------------------------------------------------------------- instrumented
class Probe:
    def __init__(self):
        self.actions = []
        self.peak_bytes = 0
        self.peak_field = 0
        self.measure_bytes = False
        self.adj_unit = None
        self.ics_unit = None
        self.tape = None

    def install(self):
        orig = CheckpointManager.__dict__["process_operation"]
        probe = self

        def wrapped(manager, cp_action, progress_bar, **kwargs):
            res = orig.__get__(manager, CheckpointManager)(
                cp_action, progress_bar, **kwargs)
            if (isinstance(cp_action, Forward) and cp_action.write_adj_deps
                    and manager.mode.name in ("EVALUATED", "EVALUATE_ADJOINT")
                    and probe.adj_unit is None):
                step = manager.tape.timesteps[cp_action.n1 - 1]
                creators = {}
                for block in manager.tape.get_blocks():
                    for out in block.get_outputs():
                        creators[id(out)] = type(block).__name__
                rows = Counter()
                for var, chk in step._checkpoint.items():
                    nb = sum(b for _, b, _ in checkpoint_buffers(chk))
                    rows[describe_bv(var, creators) + f" {nb}B"] += 1
                probe.adj_unit = dict(sorted(rows.items()))
            if (isinstance(cp_action, Forward) and not cp_action.write_adj_deps
                    and cp_action.write_ics
                    and manager.mode.name in ("EVALUATED", "EVALUATE_ADJOINT")
                    and probe.ics_unit is None):
                step = manager.tape.timesteps[cp_action.n0]
                creators = {}
                rows = Counter()
                for var, chk in step._checkpoint.items():
                    nb = sum(b for _, b, _ in checkpoint_buffers(chk))
                    rows[describe_bv(var, creators) + f" {nb}B"] += 1
                probe.ics_unit = dict(sorted(rows.items()))
            if not probe.measure_bytes:
                probe.actions.append((type(cp_action).__name__,
                                      getattr(cp_action, "n0", None),
                                      getattr(cp_action, "n1", None),
                                      manager.mode.name, 0))
                return res
            live = live_checkpoint_bytes(manager.tape, True)
            probe.peak_field = max(probe.peak_field, live.get("field", 0))
            live = sum(live.values())
            probe.peak_bytes = max(probe.peak_bytes, live)
            probe.actions.append((type(cp_action).__name__,
                                  getattr(cp_action, "n0", None),
                                  getattr(cp_action, "n1", None),
                                  manager.mode.name, live))
            return res

        CheckpointManager.process_operation = wrapped

    def recomputed_steps(self, nt):
        steps = 0
        for name, n0, n1, mode, _ in self.actions:
            if name == "Forward" and mode != "RECORD":
                steps += min(n1, nt) - n0
        return steps


def install_schedule(kind, snapshots):
    def build(self, total_steps):
        if kind == "single":
            return SingleMemoryStorageSchedule()
        s = max(1, min(snapshots, total_steps - 1))
        if kind == "mixed":
            return MixedCheckpointSchedule(total_steps, s,
                                           storage=StorageType.RAM)
        if kind == "revolve":
            return Revolve(total_steps, s)
        raise ValueError(kind)
    ads.AutomatedAdjoint._build_schedule = build


def rss_now():
    import psutil
    return psutil.Process().memory_info().rss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", default="none",
                   choices=["none", "single", "mixed", "revolve"])
    p.add_argument("--snapshots", type=int, default=None)
    p.add_argument("--abc", default="none", choices=["none", "nrbc"])
    p.add_argument("--final-time", type=float, default=0.8)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--edge-length", type=float, default=0.05)
    p.add_argument("--taylor", action="store_true")
    p.add_argument("--probe-bytes", action="store_true",
                   help="walk the tape after every schedule action "
                        "(accurate peak checkpoint bytes, distorts timings)")
    p.add_argument("--reeval", action="store_true")
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    observed = true_record(args)

    probe = Probe()
    probe.install()
    probe.measure_bytes = args.probe_bytes
    if args.schedule != "none":
        install_schedule(args.schedule, args.snapshots)

    wave = make_wave(args)
    wave.real_shot_record = observed
    wave.set_initial_velocity_model(expression=None,
                                    conditional=guess_velocity(wave),
                                    dg_velocity_model=False)
    wave.enable_automated_adjoint(
        checkpointing=args.schedule != "none",
        snapshots=args.snapshots if args.schedule in ("mixed", "revolve")
        else None)
    nt = int(wave.final_time / wave.dt) + 1
    field_bytes = wave.c.dat.data_ro_with_halos.nbytes

    gc.collect()
    rss0 = rss_now()
    t0 = time.perf_counter()
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    t_fwd = time.perf_counter() - t0
    tape = wave.automated_adjoint._tape
    after_record_cat = live_checkpoint_bytes(tape, True)
    after_record_bytes = sum(after_record_cat.values())
    rss_fwd = rss_now()
    rec_report = inspect_tape(tape, "after recording")

    J = float(wave.functional_value)
    receivers = np.array(wave.forward_solution_receivers, copy=True)
    u_final = np.array(wave.u_n.dat.data_ro, copy=True)
    u_prev = np.array(wave.u_nm1.dat.data_ro, copy=True)

    t0 = time.perf_counter()
    dJ = wave.gradient_solve(adjoint_type=AdjointType.AUTOMATED_ADJOINT)
    t_grad = time.perf_counter() - t0
    grad = np.array(dJ.dat.data_ro, copy=True)
    rev_report = inspect_tape(tape, "after reverse")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes, macOS

    result = {
        "label": args.label, "schedule": args.schedule,
        "snapshots": args.snapshots, "abc": args.abc, "nt": nt,
        "ndofs": int(wave.c.dat.data_ro.size), "field_bytes": int(field_bytes),
        "J": J, "t_forward": t_fwd, "t_gradient": t_grad,
        "t_total": t_fwd + t_grad,
        "rss_before_record": rss0, "rss_after_record": rss_fwd,
        "peak_rss": peak_rss,
        "checkpoint_bytes_after_record": after_record_bytes,
        "checkpoint_bytes_peak_reverse": max(probe.peak_bytes,
                                             after_record_bytes),
        "checkpoint_field_bytes_after_record":
            after_record_cat.get("field", 0),
        "checkpoint_field_bytes_peak_reverse": max(
            probe.peak_field, after_record_cat.get("field", 0)),
        "recomputed_steps": probe.recomputed_steps(nt),
        "action_counts": dict(Counter(a[0] for a in probe.actions)),
        "adj_unit": probe.adj_unit, "ics_unit": probe.ics_unit,
        "tape_after_record": rec_report, "tape_after_reverse": rev_report,
    }

    extra = {}
    if args.reeval:
        # New evaluation after changing the velocity model, through the tape.
        rf = wave.automated_adjoint.reduced_functional
        c_new = fire.Function(wave.c.function_space()).interpolate(
            guess_velocity(wave, shift=0.15))
        J_new = float(rf(c_new))
        g_new = rf.derivative(apply_riesz=True)
        extra["J_reeval"] = np.array(J_new)
        extra["grad_reeval"] = np.array(g_new.dat.data_ro, copy=True)
        result["J_reeval"] = J_new

    if args.taylor:
        rf = wave.automated_adjoint.reduced_functional
        size, = np.shape(wave.c.dat.data[:])
        h = fire.Function(wave.c.function_space(),
                          val=np.random.default_rng(0).random(size))
        result["taylor_rate"] = float(
            wave.automated_adjoint.verify_gradient(wave.c, direction=h))

    np.savez(args.out, receivers=receivers, grad=grad, J=np.array(J),
             u_final=u_final, u_prev=u_prev, **extra)
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in result.items()
                      if not k.startswith("tape_")}, default=str))


if __name__ == "__main__":
    sys.exit(main())
