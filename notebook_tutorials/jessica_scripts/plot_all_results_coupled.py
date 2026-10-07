"""Single analysis script: receiver comparisons (with or without Gar6more),
interface normal-displacement continuity and sparsity patterns.

The case (small 2D, large 2D or 3D) is chosen with CASE_NAME. Each scheme is
identified by the tag used in the forward script,
    <case>_<scheme>_<dim>d   (e.g. case_small_2d_sequential_2d),
which gives the input files:
    results/spyro_receiver_data_<tag>.npz
    results/sparsity_<tag>.npz
The Gar6more reference of the same case is read from:
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
CASE_NAME = "case_small_2d"   # "case_small_2d", "case_large_2d" or "case_3d"
DIMENSION = 3 if CASE_NAME == "case_3d" else 2

# Schemes to include. Comment out a line to remove a scheme.
SCHEMES = {
    "P – u_s (sequential)": ("sequential", dict(color="tab:blue", lw=1.5, ls="-")),
    # "P – u_s (monolithic)": ("monolithic", dict(color="tab:green", lw=1.5, ls="-")),
}
CASES = {label: (f"{CASE_NAME}_{scheme}_{DIMENSION}d", style)
         for label, (scheme, style) in SCHEMES.items()}

SHOW_GAR6 = True         # include the Gar6more reference and the L2 errors
NORMALIZE = False        # peak-normalize the curves (forced True if SHOW_GAR6)
PLOT_CONTINUITY = True   # interface normal-displacement continuity
PLOT_SPARSITY = True     # sparsity

# Font sizes
FS_SUPTITLE = 22   # figure title
FS_TITLE    = 22   # subplot title
FS_LABEL    = 18   # axis labels
FS_TICKS    = 18   # tick labels
FS_LEGEND   = 14   # legend

RESULTS_DIR = "results"
GAR6_BASE = "/workspaces/spyro2/notebook_tutorials/jessica_scripts/results_gar6"
GAR6MORE_DIR_FLUID = f"{GAR6_BASE}/{CASE_NAME}/fluid"
GAR6MORE_DIR_SOLID = f"{GAR6_BASE}/{CASE_NAME}/solid"
GAR6_NAME = "Gar6more3D" if DIMENSION == 3 else "Gar6more2D"
GAR6_STYLE = dict(color="red", lw=1.5, ls="--")

# In all three cases the solid receiver is on the larger-z side of the source,
# which flips the sign of the tangential component relative to Gar6more.
FLIP_UZ = True

# Spyro component -> (Gar6more file, sign).
#   Spyro: u_z = along the interface (offset direction), u_x = normal to it,
#          u_y = second horizontal direction (3D only).
#   Gar6more2D: Ux = horizontal, Uy = vertical.
if DIMENSION == 2:
    COMPONENTS = [
        ("uz", "Ux.dat", -1.0 if FLIP_UZ else 1.0),
        ("ux", "Uy.dat", -1.0),
    ]
else:
    # TO CONFIRM with the Gar6more3D documentation/output: assumed here
    # Ux = horizontal (offset direction), Uy = second horizontal, Uz = vertical.
    COMPONENTS = [
        ("uz", "Ux.dat", -1.0 if FLIP_UZ else 1.0),
        ("ux", "Uz.dat", -1.0),
        ("uy", "Uy.dat", 1.0),
    ]

if SHOW_GAR6:
    NORMALIZE = True   # the Gar6more amplitude scale differs from spyro's
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================================================================
# Helpers
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
# Load cases
# ===========================================================================
print(f"=== Case: {CASE_NAME} ({DIMENSION}D) ===")
if SHOW_GAR6:
    print(f"Gar6more fluid: {GAR6MORE_DIR_FLUID}")
    print(f"Gar6more solid: {GAR6MORE_DIR_SOLID}")
    gar6 = {"p": load_gar6more_file(f"{GAR6MORE_DIR_FLUID}/P.dat")}
    for comp, fname, _ in COMPONENTS:
        gar6[comp] = load_gar6more_file(f"{GAR6MORE_DIR_SOLID}/{fname}")

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
    t_p = np.arange(len(p)) * dt_case   # sample k is t = k*dt
    t_u = np.arange(len(u)) * dt_case

    r = dict(tag=tag, style=style, t_p=t_p, t_u=t_u, p=prep(p),
             ux_solid_raw=u[:, 1])
    for i, comp in enumerate(["uz", "ux", "uy"][:u.shape[1]]):
        r[comp] = prep(u[:, i])

    # Fluid-side normal displacement (raw), if it was recorded
    if "u_fluid_x" in d and np.asarray(d["u_fluid_x"]).size:
        r["ux_fluid_raw"] = np.asarray(d["u_fluid_x"]).ravel()

    if SHOW_GAR6:
        r["p_g"] = gar6_on(t_p, *gar6["p"])
        r["err_p"] = MeasureError.calculate_normalized_L2_error(r["p"], r["p_g"])
        for comp, _, sign in COMPONENTS:
            r[f"{comp}_g"] = gar6_on(t_u, *gar6[comp], sign)
            r[f"err_{comp}"] = MeasureError.calculate_normalized_L2_error(
                r[comp], r[f"{comp}_g"])
        r["shift"] = estimate_time_shift(t_p, r["p"], r["p_g"])
    results[label] = r

if not results:
    raise SystemExit("No receiver .npz file found.")

comp_keys = [c for c, _, _ in COMPONENTS]
if SHOW_GAR6:
    w = max(len(k) for k in results) + 2
    print(f"\n=== Normalized L2 error vs {GAR6_NAME} ===")
    header = f"{'Case':<{w}} {'P':>9}" + "".join(
        f" {'u_' + c[1]:>9}" for c in comp_keys) + f" {'shift (s)':>11}"
    print(header)
    for label, r in results.items():
        line = f"{label:<{w}} {r['err_p']:9.4f}" + "".join(
            f" {r['err_' + c]:9.4f}" for c in comp_keys) + f" {r['shift']:11.6f}"
        print(line)


def curve_label(label, r, key):
    return f"{label}  (L2 = {r[key] * 100:.2f}%)" if SHOW_GAR6 else label


ref = next(iter(results.values()))
amp = "Amplitude (normalized)" if NORMALIZE else "Amplitude"
suffix = "_vs_gar6" if SHOW_GAR6 else ""
cases_tag = "__".join(r["tag"] for r in results.values())

# ===========================================================================
# Fluid receiver — pressure
# ===========================================================================
fig_p, ax_p = plt.subplots(figsize=(12, 5))
if SHOW_GAR6:
    ax_p.plot(ref["t_p"], ref["p_g"], label=GAR6_NAME, **GAR6_STYLE)
for label, r in results.items():
    ax_p.plot(r["t_p"], r["p"], label=curve_label(label, r, "err_p"), **r["style"])
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
    if SHOW_GAR6:
        ax.plot(ref["t_u"], ref[f"{comp}_g"], label=GAR6_NAME, **GAR6_STYLE)
    for label, r in results.items():
        ax.plot(r["t_u"], r[comp], label=curve_label(label, r, f"err_{comp}"),
                **r["style"])
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
            if split > 0:  # monolithic: separate the P and u_s blocks
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