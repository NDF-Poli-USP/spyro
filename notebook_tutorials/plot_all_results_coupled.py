"""Single analysis script: receiver comparisons (with or without Gar6more2D),
interface normal-displacement continuity and sparsity patterns, for every
case selected in CASES.

Each case is identified by the tag used in the forward script
(<formulation>_<scheme>), which gives the two input files:
    results/spyro_receiver_data_<tag>.npz
    results/sparsity_<tag>.npz
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
# SETTINGS
# ===========================================================================
# Cases to include: label -> (tag, line style).
# Comment out a line to remove a case; append a line to add one.
CASES = {
    # "P – u_s (sequential)": (
    #     "pressure_sequential", dict(color="tab:blue", lw=1.5, ls="-")),
    # "P – u_s (monolithic)": (
    #     "pressure_monolithic", dict(color="tab:green", lw=1.5, ls="-")),
    "u_f – u_s (sequential)": (
        "displacement_sequential", dict(color="tab:orange", lw=1.5, ls="-")),
}

SHOW_GAR6 = True         # include the Gar6more2D reference (and the L2 errors)
NORMALIZE = True         # peak-normalize the curves (forced True if SHOW_GAR6)
PLOT_CONTINUITY = True   # interface normal-displacement continuity, per case
PLOT_SPARSITY = True     # sparsity-pattern figure for each case

# Font sizes (change here to change every figure)
FS_SUPTITLE = 22   # figure title
FS_TITLE    = 22   # subplot title
FS_LABEL    = 18   # axis labels
FS_TICKS    = 18   # tick labels
FS_LEGEND   = 14   # legend

RESULTS_DIR = "results"
GAR6MORE_DIR_FLUID = "/workspaces/spyro/notebook_tutorials/python_files/results_fluid"
GAR6MORE_DIR_SOLID = "/workspaces/spyro/notebook_tutorials/python_files/results_solid"
FLIP_UZ = True
GAR6_STYLE = dict(color="red", lw=1.5, ls="--")

if SHOW_GAR6:
    NORMALIZE = True   # the Gar6more2D amplitude scale differs from Spyro's
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# HELPERS
# ===========================================================================
def load_gar6more_file(path):
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


def gar6_on(t, t_ref, v_ref, sign=1.0):
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
# LOAD CASES
# ===========================================================================
if SHOW_GAR6:
    t_g_p,  p_g  = load_gar6more_file(f"{GAR6MORE_DIR_FLUID}/P.dat")
    t_g_ux, ux_g = load_gar6more_file(f"{GAR6MORE_DIR_SOLID}/Ux.dat")
    t_g_uy, uy_g = load_gar6more_file(f"{GAR6MORE_DIR_SOLID}/Uy.dat")
    uz_sign = -1.0 if FLIP_UZ else 1.0

prep = normalize if NORMALIZE else (lambda x: x)
results = {}
for label, (tag, style) in CASES.items():
    path = f"{RESULTS_DIR}/spyro_receiver_data_{tag}.npz"
    if not os.path.exists(path):
        print(f"[WARNING] {path} not found — '{label}' is skipped.")
        continue
    d = np.load(path)
    T = float(d["final_time"])
    p = np.asarray(d["p_spyro"]).ravel()
    u = np.asarray(d["u_solid"])

    # Records start at t = 0 and end at T, with step dt
    t_p = np.linspace(0.0, T, len(p))
    t_u = np.linspace(0.0, T, len(u))

    r = dict(tag=tag, style=style, t_p=t_p, t_u=t_u,
             p=prep(p), uz=prep(u[:, 0]), ux=prep(u[:, 1]),
             ux_solid_raw=u[:, 1])

    # Fluid-side normal displacement (raw), if it was recorded
    if "u_fluid_x" in d and np.asarray(d["u_fluid_x"]).size:
        r["ux_fluid_raw"] = np.asarray(d["u_fluid_x"]).ravel()

    if SHOW_GAR6:
        r["p_g"]  = gar6_on(t_p, t_g_p, p_g)
        r["uz_g"] = gar6_on(t_u, t_g_ux, ux_g, uz_sign)
        r["ux_g"] = gar6_on(t_u, t_g_uy, uy_g, -1.0)
        r["err_p"]  = MeasureError.calculate_normalized_L2_error(r["p"],  r["p_g"])
        r["err_uz"] = MeasureError.calculate_normalized_L2_error(r["uz"], r["uz_g"])
        r["err_ux"] = MeasureError.calculate_normalized_L2_error(r["ux"], r["ux_g"])
        r["shift"]  = estimate_time_shift(t_p, r["p"], r["p_g"])
    results[label] = r

if not results:
    raise SystemExit("No receiver .npz file found.")

if SHOW_GAR6:
    w = max(len(k) for k in results) + 2
    print("\n=== Normalized L2 error vs Gar6more2D ===")
    print(f"{'Case':<{w}} {'P':>9} {'u_z':>9} {'u_x':>9} {'shift (s)':>11}")
    for label, r in results.items():
        print(f"{label:<{w}} {r['err_p']:9.4f} {r['err_uz']:9.4f} "
              f"{r['err_ux']:9.4f} {r['shift']:11.6f}")


def curve_label(label, r, key):
    return f"{label}  (L2 = {r[key] * 100:.2f}%)" if SHOW_GAR6 else label


ref = next(iter(results.values()))
amp = "Amplitude (normalized)" if NORMALIZE else "Amplitude"
suffix = "_vs_gar6" if SHOW_GAR6 else ""
cases_tag = "__".join(r["tag"] for r in results.values())


# ===========================================================================
# FLUID RECEIVER — pressure
# ===========================================================================
fig_p, ax_p = plt.subplots(figsize=(12, 5))
if SHOW_GAR6:
    ax_p.plot(ref["t_p"], ref["p_g"], label="Gar6more2D", **GAR6_STYLE)
for label, r in results.items():
    ax_p.plot(r["t_p"], r["p"], label=curve_label(label, r, "err_p"), **r["style"])
style_axis(ax_p, "Fluid receiver — pressure", amp, "Time (s)")
fig_p.tight_layout()
p_path = f"{RESULTS_DIR}/receiver_fluid_P_{cases_tag}{suffix}.png"
fig_p.savefig(p_path, dpi=150)
plt.close(fig_p)
print(f"[OK] Saved to: {os.path.abspath(p_path)}")


# ===========================================================================
# SOLID RECEIVER — displacement components (same layout as the fluid one)
# ===========================================================================
fig_u, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
for ax, comp, gkey, ekey, title in (
    (axes[0], "uz", "uz_g", "err_uz", "Solid receiver — displacement u_z"),
    (axes[1], "ux", "ux_g", "err_ux", "Solid receiver — displacement u_x"),
):
    if SHOW_GAR6:
        ax.plot(ref["t_u"], ref[gkey], label="Gar6more2D", **GAR6_STYLE)
    for label, r in results.items():
        ax.plot(r["t_u"], r[comp], label=curve_label(label, r, ekey), **r["style"])
    style_axis(ax, title, amp, "Time (s)" if ax is axes[1] else None)
fig_u.tight_layout()
u_path = f"{RESULTS_DIR}/receiver_solid_u_{cases_tag}{suffix}.png"
fig_u.savefig(u_path, dpi=150)
plt.close(fig_u)
print(f"[OK] Saved to: {os.path.abspath(u_path)}")


# ===========================================================================
# INTERFACE NORMAL-DISPLACEMENT CONTINUITY (one figure per case, raw values)
# ===========================================================================
if PLOT_CONTINUITY:
    for label, r in results.items():
        if "ux_fluid_raw" not in r:
            print(f"[WARNING] '{label}': fluid-side u_x not recorded — no continuity plot.")
            continue
        uf, us = r["ux_fluid_raw"], r["ux_solid_raw"]
        n = min(len(uf), len(us))
        t = r["t_u"][:n]
        mismatch = np.linalg.norm(uf[:n] - us[:n]) / max(np.linalg.norm(us[:n]), 1e-30)

        fig_c, ax_c = plt.subplots(figsize=(12, 5))
        ax_c.plot(t, uf[:n], color=r["style"]["color"], lw=1.5, ls="-",
                  label="fluid side u_x")
        ax_c.plot(t, us[:n], color="black", lw=1.5, ls="--",
                  label="solid side u_x")
        style_axis(ax_c, f"Interface continuity — {label}",
                   "Normal displacement u_x", "Time (s)")
        ax_c.text(0.01, 0.95, f"relative mismatch = {mismatch * 100:.2f}%",
                  transform=ax_c.transAxes, fontsize=FS_LEGEND, va="top")
        fig_c.tight_layout()
        c_path = f"{RESULTS_DIR}/interface_continuity_{r['tag']}.png"
        fig_c.savefig(c_path, dpi=150)
        plt.close(fig_c)
        print(f"[OK] Saved to: {os.path.abspath(c_path)}")

# ===========================================================================
# SPARSITY PATTERNS (one figure per case)
# ===========================================================================
if PLOT_SPARSITY:
    for label, (tag, _) in CASES.items():
        s_path = f"{RESULTS_DIR}/sparsity_{tag}.npz"
        if not os.path.exists(s_path):
            print(f"[WARNING] {s_path} not found — no sparsity plot for '{label}'.")
            continue
        case, labels, mats, rows, cols, split = load_sparsity_matrices(s_path)

        fig_s, axes_s = plt.subplots(1, len(mats), figsize=(6.5 * len(mats), 7))
        for ax, A, lab, rl, cl in zip(np.atleast_1d(axes_s), mats, labels, rows, cols):
            ax.spy(A, markersize=4 if max(A.shape) < 5000 else 0.3)
            if split > 0:  # monolithic: separate the P and u_s blocks
                ax.axhline(split, color="red", lw=0.8, ls="--")
                ax.axvline(split, color="red", lw=0.8, ls="--")
            ax.set_title(lab, fontsize=FS_TITLE)
            # ax.set_ylabel(f"Row: {rl}", fontsize=FS_LABEL)
            # ax.set_xlabel(f"Column: {cl}", fontsize=FS_LABEL)
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