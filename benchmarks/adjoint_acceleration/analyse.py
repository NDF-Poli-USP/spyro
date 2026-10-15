"""Tabulate the sweep: by number of checkpoints and under equal memory."""
import glob
import json
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "results/sweep"
FORMS = ["orig", "new", "inplace"]


def load(mode):
    out = {}
    for f in glob.glob(f"{D}/{mode}_*.json"):
        _, v, sch, s = os.path.basename(f)[:-5].split("_")
        out[(v, sch, int(s))] = json.load(open(f))
    return out


probe, timing = load("probe"), load("timing")
ref = np.load(f"{D}/probe_orig_single_0.npz")


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


print("== equivalence against orig/single (all configurations)")
worst = {"J": 0, "receivers": 0, "u_final": 0, "grad": 0}
for (v, sch, s) in probe:
    d = np.load(f"{D}/probe_{v}_{sch}_{s}.npz")
    worst["J"] = max(worst["J"], abs(float(d["J"]) - float(ref["J"]))
                     / abs(float(ref["J"])))
    for k in ("receivers", "u_final", "grad"):
        worst[k] = max(worst[k], rel(d[k], ref[k]))
print({k: f"{v:.1e}" for k, v in worst.items()})

fb = next(iter(probe.values()))["field_bytes"]
nt = next(iter(probe.values()))["nt"]
print(f"\nfield = {fb} B, nt = {nt}")
print("\n== by number of checkpoints (peak checkpoint memory in fields; "
      "recomputed steps; time s)")
hdr = f"{'sched':8}{'s':>4} | " + " | ".join(
    f"{v:>7} mem  rec   time" for v in FORMS)
print(hdr)
for sch in ["single", "mixed", "revolve"]:
    for s in sorted({k[2] for k in probe if k[1] == sch}):
        row = f"{sch:8}{s:>4} | "
        cells = []
        for v in FORMS:
            p = probe.get((v, sch, s))
            t = timing.get((v, sch, s))
            if p is None:
                cells.append(" " * 24)
                continue
            mem = p["checkpoint_field_bytes_peak_reverse"] / fb
            tt = t["t_total"] if t else float("nan")
            cells.append(f"{mem:9.1f} {p['recomputed_steps']:5d} {tt:6.1f}")
        print(row + " | ".join(cells))

print("\n== equal memory budget: best configuration with peak <= B fields")
budgets = [25, 50, 100, 200, 400]
for B in budgets:
    print(f"B = {B} fields")
    for v in FORMS:
        best = []
        for sch in ["mixed", "revolve"]:
            ok = [(probe[k]["recomputed_steps"], k) for k in probe
                  if k[0] == v and k[1] == sch
                  and probe[k]["checkpoint_field_bytes_peak_reverse"] / fb
                  <= B]
            if ok:
                r, k = min(ok)
                t = timing.get(k, {}).get("t_total", float("nan"))
                mem = probe[k]["checkpoint_field_bytes_peak_reverse"] / fb
                best.append(f"{sch} s={k[2]:<3} rec={r:<5} mem={mem:6.1f} "
                            f"t={t:5.1f}s")
            else:
                best.append(f"{sch} -- does not fit")
        print(f"  {v:8} " + " || ".join(best))

print("\n== peak RSS (MB) and time, timing runs")
for k in sorted(timing):
    t = timing[k]
    print(f"{k[0]:8}{k[1]:8}{k[2]:>4}  peakRSS={t['peak_rss']/2**20:7.1f} "
          f"fwd={t['t_forward']:6.2f} grad={t['t_gradient']:6.2f} "
          f"tot={t['t_total']:6.2f} rec={t['recomputed_steps']}")
