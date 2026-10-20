"""Single analysis script: receiver comparisons (with or without Gar6more),
interface normal-displacement continuity and sparsity patterns.

The case is chosen with CASE_NAME. Each scheme is identified by the tag used
in the forward script,
    <case>_<scheme>_<dim>d   (e.g. case_small_2d_sequential_2d),
which gives the input files:
    results/spyro_receiver_data_<tag>.npz
    results/sparsity_<tag>.npz
The Gar6more reference of the same case, when it exists, is read from:
    results_gar6/<case>/fluid/P.dat
    results_gar6/<case>/solid/Ux.dat, Uy.dat (Uz.dat in 3D)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import correlate

from spyro.tools.error_measure import MeasureError
from sparsity_plots import load_sparsity_matrices

# ===========================================================================
# Settings
# ===========================================================================
# "case_small_2d", "case_large_2d", "case_3d",
# "case_small_2d_irregular_topography", "case_large_2d_irregular_topography",
# "case_3d_irregular_topography"
CASE_NAME = "case_3d"
DIMENSION = 3 if CASE_NAME.startswith("case_3d") else 2

# Results computed before the axis change (x vertical, z horizontal):
# u_solid columns are (horizontal, vertical[, y]) instead of (vertical, horizontal[, y]).
LEGACY_AXES = False

SCHEMES = {
    "P – u_s (sequential)": ("sequential", dict(color="tab:blue", lw=1.5, ls="-")),
    # "P – u_s (monolithic)": ("monolithic", dict(color="tab:green", lw=1.5, ls="-")),
}
CASES = {label: (f"{CASE_NAME}_{scheme}_{DIMENSION}d", style)
         for label, (scheme, style) in SCHEMES.items()}

NORMALIZE = False
PLOT_CONTINUITY = True
PLOT_SPARSITY = True

FS_SUPTITLE = 22
FS_TITLE    = 22
FS_LABEL    = 18
FS_TICKS    = 18
FS_LEGEND   = 14

RESULTS_DIR = "results"
GAR6_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gar6")
GAR6MORE_DIR_FLUID = f"{GAR6_BASE}/{CASE_NAME}/fluid"
GAR6MORE_DIR_SOLID = f"{GAR6_BASE}/{CASE_NAME}/solid"
GAR6_NAME = "Gar6more3D" if DIMENSION == 3 else "Gar6more2D"
GAR6_STYLE = dict(color="red", lw=1.5, ls="--")
SHOW_GAR6 = os.path.isfile(f"{GAR6MORE_DIR_FLUID}/P.dat")

# The solid receiver is on the larger-x side of the source, which flips the
# sign of the horizontal component relative to Gar6more.
FLIP_UX = True

# (name, column in u_solid, Gar6more file, sign)
COL_UZ, COL_UX = (1, 0) if LEGACY_AXES else (0, 1)
if DIMENSION == 2:
    COMPONENTS = [
        ("uz", COL_UZ, "Uy.dat", -1.0),
        ("ux", COL_UX, "Ux.dat", -1.0 if FLIP_UX else 1.0),
    ]
else:
    COMPONENTS = [
        ("uz", COL_UZ, "Uz.dat", 1.0),
        ("ux", COL_UX, "Ux.dat", 1.0),
        ("uy", 2, "Uy.dat", 1.0),
    ]

SPECFEM_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_specfem")
SPECFEM_DIR = f"{SPECFEM_BASE}/{CASE_NAME}"
SPECFEM_NAME = "SPECFEM2D"
SPECFEM_STYLE = dict(color="black", lw=1.2, ls=":")
SHOW_SPECFEM = DIMENSION == 2 and os.path.isfile(f"{SPECFEM_DIR}/AA.S0001.PRE.semp")

# SPECFEM centers the Ricker at t = 0; spyro centers it at t = delay = 1/f.
SPECFEM_DELAY = {
    "case_small_2d": 1 / 25,
    "case_small_2d_irregular_topography": 1 / 25,
    "case_large_2d": 1 / 10,
    "case_large_2d_irregular_topography": 1 / 10,
}.get(CASE_NAME, 0.0)

# key -> (file, sign). S0001 = fluid receiver, S0002 = solid receiver.
SPECFEM_FILES = {
    "p": ("AA.S0001.PRE.semp", 1.0),
    "uz": ("AA.S0002.BXZ.semd", 1.0),
    "ux": ("AA.S0002.BXX.semd", 1.0),
}

if SHOW_GAR6 or SHOW_SPECFEM:
    NORMALIZE = True
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================================================================
# Helpers
# ===========================================================================
def load_trace(path):
    values = np.loadtxt(path)
    if values.ndim == 1:
        raise ValueError(f"{path}: only 1 column found — expected (time, value).")
    return values[:, 0], values[:, 1]


def normalize(x):
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def estimate_time_shift(time_vector, numerical, reference):
    numerical = numerical - np.mean(numerical)
    reference = reference - np.mean(reference)
    correlation = correlate(numerical, reference, mode="full")
    lag_samples = np.argmax(correlation) - (len(reference) - 1)
    return lag_samples * (time_vector[1] - time_vector[0])


def ref_on(t, t_ref, v_ref, sign=1.0):
    return normalize(sign * np.interp(t, t_ref, v_ref, left=0.0, right=0.0))


def style_axis(ax, title, ylabel, xlabel=None):
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICKS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND)

# ===========================================================================
# Load cases
# ===========================================================================
print(f"=== Case: {CASE_NAME} ({DIMENSION}D) ===")
if SHOW_GAR6:
    print(f"Gar6more fluid: {GAR6MORE_DIR_FLUID}")
    print(f"Gar6more solid: {GAR6MORE_DIR_SOLID}")
    gar6 = {"p": load_trace(f"{GAR6MORE_DIR_FLUID}/P.dat")}
    for comp, _, fname, _ in COMPONENTS:
        gar6[comp] = load_trace(f"{GAR6MORE_DIR_SOLID}/{fname}")
if SHOW_SPECFEM:
    print(f"SPECFEM: {SPECFEM_DIR}  (time shift +{SPECFEM_DELAY:.4f} s)")
    spec = {}
    for key, (fname, _) in SPECFEM_FILES.items():
        t_s, v_s = load_trace(f"{SPECFEM_DIR}/{fname}")
        spec[key] = (t_s + SPECFEM_DELAY, v_s)
if not (SHOW_GAR6 or SHOW_SPECFEM):
    print("No reference for this case — plotting spyro only.")

prep = normalize if NORMALIZE else (lambda x: x)
results = {}
for label, (tag, style) in CASES.items():
    path = f"{RESULTS_DIR}/spyro_receiver_data_{tag}.npz"
    if not os.path.exists(path):
        print(f"[WARNING] {path} not found — '{label}' is skipped.")
        continue
    d = np.load(path)
    p = np.asarray(d["p_spyro"]).ravel()
    u = np.asarray(d["u_solid"])

    dt_case = float(d["dt"])
    t_p = np.arange(len(p)) * dt_case
    t_u = np.arange(len(u)) * dt_case

    r = dict(tag=tag, style=style, t_p=t_p, t_u=t_u, p=prep(p),
             uz_solid_raw=u[:, COL_UZ])
    for comp, col, _, _ in COMPONENTS:
        r[comp] = prep(u[:, col])

    if "u_fluid_z" in d and np.asarray(d["u_fluid_z"]).size:
        r["uz_fluid_raw"] = np.asarray(d["u_fluid_z"]).ravel()

    if SHOW_GAR6:
        r["p_g"] = ref_on(t_p, *gar6["p"], 1.0 if DIMENSION == 2 else -1.0)
        r["err_p_g"] = MeasureError.calculate_normalized_L2_error(r["p"], r["p_g"])
        for comp, _, _, sign in COMPONENTS:
            r[f"{comp}_g"] = ref_on(t_u, *gar6[comp], sign)
            r[f"err_{comp}_g"] = MeasureError.calculate_normalized_L2_error(
                r[comp], r[f"{comp}_g"])
        r["shift_g"] = estimate_time_shift(t_p, r["p"], r["p_g"])

    if SHOW_SPECFEM:
        r["p_s"] = ref_on(t_p, *spec["p"], SPECFEM_FILES["p"][1])
        r["err_p_s"] = MeasureError.calculate_normalized_L2_error(r["p"], r["p_s"])
        for comp in ("uz", "ux"):
            r[f"{comp}_s"] = ref_on(t_u, *spec[comp], SPECFEM_FILES[comp][1])
            r[f"err_{comp}_s"] = MeasureError.calculate_normalized_L2_error(
                r[comp], r[f"{comp}_s"])
        r["shift_s"] = estimate_time_shift(t_p, r["p"], r["p_s"])
    results[label] = r

if not results:
    raise SystemExit("No receiver .npz file found.")

comp_keys = [c for c, _, _, _ in COMPONENTS]
REFERENCES = [(tag, name, style) for tag, name, style, show in (
    ("g", GAR6_NAME, GAR6_STYLE, SHOW_GAR6),
    ("s", SPECFEM_NAME, SPECFEM_STYLE, SHOW_SPECFEM),
) if show]

ref = next(iter(results.values()))
w = max(len(k) for k in results) + 2
for rt, rname, _ in REFERENCES:
    keys = ["p"] + [c for c in comp_keys if f"err_{c}_{rt}" in ref]
    print(f"\n=== Normalized L2 error vs {rname} ===")
    print(f"{'Case':<{w}}" + "".join(f" {('P' if k == 'p' else 'u_' + k[1]):>9}" for k in keys)
          + f" {'shift (s)':>11}")
    for label, r in results.items():
        print(f"{label:<{w}}" + "".join(f" {r[f'err_{k}_{rt}']:9.4f}" for k in keys)
              + f" {r[f'shift_{rt}']:11.6f}")


def curve_label(label, r, key):
    errs = [f"{rname} {r[f'err_{key}_{rt}'] * 100:.2f}%" for rt, rname, _ in REFERENCES
            if f"err_{key}_{rt}" in r]
    return f"{label}  (L2: {', '.join(errs)})" if errs else label


def plot_references(ax, key, t):
    for rt, rname, rstyle in REFERENCES:
        if f"{key}_{rt}" in ref:
            ax.plot(t, ref[f"{key}_{rt}"], label=rname, **rstyle)


amp ="Amplitude (normalized)" if NORMALIZE else "Amplitude"
suffix = "".join(f"_vs_{'gar6' if rt == 'g' else 'specfem'}" for rt, _, _ in REFERENCES)
cases_tag = "__".join(r["tag"] for r in results.values())

# ===========================================================================
# Fluid receiver — pressure
# ===========================================================================
fig_p, ax_p = plt.subplots(figsize=(12, 5))
plot_references(ax_p, "p", ref["t_p"])
for label, r in results.items():
    ax_p.plot(r["t_p"], r["p"], label=curve_label(label, r, "p"), **r["style"])
style_axis(ax_p, "Fluid receiver — pressure", amp, "Time (s)")
fig_p.tight_layout()
p_path = f"{RESULTS_DIR}/receiver_fluid_P_{cases_tag}{suffix}.png"
fig_p.savefig(p_path, dpi=150)
plt.close(fig_p)
print(f"[OK] Saved to: {os.path.abspath(p_path)}")

# ===========================================================================
# Solid receiver — displacement components (2 in 2D, 3 in 3D)
# ===========================================================================
n_comp = len(comp_keys)
fig_u, axes = plt.subplots(n_comp, 1, figsize=(12, 4.5 * n_comp), sharex=True)
for i, (ax, comp) in enumerate(zip(axes, comp_keys)):
    plot_references(ax, comp, ref["t_u"])
    for label, r in results.items():
        ax.plot(r["t_u"], r[comp], label=curve_label(label, r, comp), **r["style"])
    style_axis(ax, f"Solid receiver — displacement u_{comp[1]}", amp,
               "Time (s)" if i == n_comp - 1 else None)
fig_u.tight_layout()
u_path = f"{RESULTS_DIR}/receiver_solid_u_{cases_tag}{suffix}.png"
fig_u.savefig(u_path, dpi=150)
plt.close(fig_u)
print(f"[OK] Saved to: {os.path.abspath(u_path)}")

# ===========================================================================
# Interface normal - Displacement continuity
# ===========================================================================
if PLOT_CONTINUITY:
    for label, r in results.items():
        if "uz_fluid_raw" not in r:
            print(f"[WARNING] '{label}': fluid-side u_z not recorded — no continuity plot.")
            continue
        uf, us = r["uz_fluid_raw"], r["uz_solid_raw"]
        n = min(len(uf), len(us))
        t = r["t_u"][:n]
        mismatch = np.linalg.norm(uf[:n] - us[:n]) / max(np.linalg.norm(us[:n]), 1e-30)

        fig_c, ax_c = plt.subplots(figsize=(12, 5))
        ax_c.plot(t, uf[:n], color=r["style"]["color"], lw=1.5, ls="-",
                  label="fluid side u_z")
        ax_c.plot(t, us[:n], color="black", lw=1.5, ls="--",
                  label="solid side u_z")
        style_axis(ax_c, f"Interface continuity — {label}",
                   "Normal displacement u_z", "Time (s)")
        ax_c.text(0.01, 0.95, f"relative mismatch = {mismatch * 100:.2f}%",
                  transform=ax_c.transAxes, fontsize=FS_LEGEND, va="top")
        fig_c.tight_layout()
        c_path = f"{RESULTS_DIR}/interface_continuity_{r['tag']}.png"
        fig_c.savefig(c_path, dpi=150)
        plt.close(fig_c)
        print(f"[OK] Saved to: {os.path.abspath(c_path)}")

# ===========================================================================
# Sparsity
# ===========================================================================
if PLOT_SPARSITY:
    for label, (tag, _) in CASES.items():
        s_path = f"{RESULTS_DIR}/sparsity_{tag}.npz"
        if not os.path.exists(s_path):
            print(f"[WARNING] {s_path} not found — no sparsity plot for '{label}'.")
            continue
        case, labels, mats, rows, cols, split = load_sparsity_matrices(s_path)

        fig_s, axes_s = plt.subplots(1, len(mats), figsize=(6.5 * len(mats), 7))
        for ax, A, lab in zip(np.atleast_1d(axes_s), mats, labels):
            ax.spy(A, markersize=4 if max(A.shape) < 5000 else 0.3)
            if split > 0:
                ax.axhline(split, color="red", lw=0.8, ls="--")
                ax.axvline(split, color="red", lw=0.8, ls="--")
            ax.set_title(lab, fontsize=FS_TITLE)
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis="both", labelsize=FS_TICKS)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        fig_s.suptitle(f"Sparsity: {label}", fontsize=FS_SUPTITLE)
        fig_s.tight_layout()
        out = f"{RESULTS_DIR}/sparsity_{tag}.png"
        fig_s.savefig(out, dpi=150)
        plt.close(fig_s)
        print(f"[OK] Saved to: {os.path.abspath(out)}")