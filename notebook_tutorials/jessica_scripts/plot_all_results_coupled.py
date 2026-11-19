"""Receiver comparisons and interface continuity for the coupled solver.

Several runs of the same case (e.g. isotropic, VTI and TTI solids, or the
sequential and monolithic schemes) are drawn on the same figures, together
with the SPECFEM2D / Gar6more references when they exist.

Runs are named as in fluid_solid_coupled_forward.py: the solid is
<rheology>_<solid>, where <solid> is the medium plus, for vti/tti, the C13
relation (e.g. elastic_iso, elastic_vti-exact, viscoelastic_tti-weak), and a
run is <rheology>_<solid>_<scheme> (e.g. elastic_tti-exact_sequential).

Inputs, written by fluid_solid_coupled_forward.py:
    results/<case>/<rheology>_<solid>_<scheme>/receivers.npz
Outputs (in every format of FIGURE_FORMATS of fluid_solid_coupled_forward.py):
    results/<case>/plots/
References:
    results_specfem/case_<case>-<rheology>_<solid>/AA.S0001.PRE.semp,
        AA.S0002.BXZ.semd, AA.S0002.BXX.semd  (one SPECFEM run per solid; each
        spyro run is compared with the SPECFEM run of its own solid)
    results_gar6/<folder>/fluid/P.dat  (folders keep their names, see GAR6_FOLDERS)
    results_gar6/<folder>/solid/Ux.dat, Uy.dat (Uz.dat in 3D)
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate

from spyro.tools.error_measure import MeasureError
from fluid_solid_coupled_forward import save_figure, solid_name

# ===========================================================================
# Settings
# ===========================================================================
# "small_2d_planar", "small_2d_non_planar", "small_2d_visco",
# "large_2d_non_planar", "small_3d_planar", "large_3d_non_planar"
CASE = "small_2d_visco"

# Runs drawn together: (rheology, medium, anisotropy_type, scheme), with the
# same values as the settings of fluid_solid_coupled_forward.py (rheology
# "elastic" or "viscoelastic"; medium "iso", "vti" or "tti"; anisotropy_type
# "exact" or "weak", ignored for "iso"; scheme "sequential" or "monolithic").
# Any subset of the runs already in results/<case>/ can be listed; runs
# without results are skipped. Examples:
#   [("elastic", "tti", "exact", "sequential")]               one run vs SPECFEM
#   [("elastic", "vti", "exact", "sequential"),
#    ("elastic", "vti", "weak", "sequential")]                C13 relations
#   [("elastic", "tti", "exact", "sequential"),
#    ("viscoelastic", "tti", "exact", "sequential")]          rheologies
#   [("elastic", "iso", "exact", "sequential"),
#    ("viscoelastic", "iso", "exact", "sequential"),
#    ("elastic", "vti", "exact", "sequential"),
#    ("viscoelastic", "vti", "exact", "sequential")]          media x rheologies
#   [("elastic", "tti", "exact", "sequential"),
#    ("elastic", "tti", "exact", "monolithic")]               schemes
RUNS = [
    # ("elastic", "iso", "exact", "sequential"),
    # ("viscoelastic", "iso", "exact", "sequential"),
    # ("elastic", "vti", "exact", "sequential"),
    ("viscoelastic", "vti", "exact", "sequential"),
    # ("elastic", "vti", "weak", "sequential"),
    # ("viscoelastic", "vti", "weak", "sequential"),
    # ("elastic", "tti", "exact", "sequential"),
    # ("viscoelastic", "tti", "exact", "sequential"),
    # ("elastic", "tti", "weak", "sequential"),
    # ("viscoelastic", "tti", "weak", "sequential"),
]

# How the runs of RUNS are grouped into figures and drawn:
#   "solids"    compare the solids: one figure per rheology, with every solid
#               of RUNS in it. One colour per medium (MEDIUM_COLORS); exact
#               (and iso) runs are solid lines, weak runs dotted lines of the
#               same colour.
#   "rheology"  compare elastic and viscoelastic: one figure per solid, with
#               both rheologies in it. The elastic run is a solid line in the
#               medium colour, the viscoelastic one a dotted line in a lighter
#               tint of it (VISCOELASTIC_TINT is the fraction of white mixed in).
# In both, monolithic runs are drawn thicker and translucent, so a sequential
# run of the same solid shows on top of them. SPECFEM is a black dotted line.
PLOT_TYPE = "solids"
MEDIUM_COLORS = {"iso": "tab:blue", "vti": "tab:green", "tti": "tab:red"}
DOTTED = (0, (1, 1.5))
VISCOELASTIC_TINT = 0.45
# With more runs than this in a figure, the receiver legends go to the right
# of the axes (and the figure widens) instead of covering the traces.
MAX_LEGEND_INSIDE = 4

# Solid the Gar6more reference describes, as <rheology>_<solid> (e.g.
# "elastic_iso"): its L2 errors are computed only for runs of this solid; None
# computes them for every run. (SPECFEM has one reference per solid, so it is
# always matched by solid.)
REFERENCE_SOLID = None

USE_SPECFEM = True
USE_GAR6 = False

# Results computed before the axis change (x vertical, z horizontal):
# u_solid columns are (horizontal, vertical[, y]) instead of (vertical, horizontal[, y]).
LEGACY_AXES = False

NORMALIZE = False  # forced to True when a reference is shown
PLOT_CONTINUITY = True

FS_SUPTITLE = 22
FS_TITLE    = 22
FS_LABEL    = 18
FS_TICKS    = 18
FS_LEGEND   = 14

DIMENSION = 3 if "_3d_" in CASE else 2
RESULTS_DIR = "results"
CASE_DIR = os.path.join(RESULTS_DIR, CASE)
PLOTS_DIR = os.path.join(CASE_DIR, "plots")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gar6more folders, named as when they were generated.
GAR6_FOLDERS = {
    "small_2d_planar": "case_small_2d",
    "small_2d_non_planar": "case_small_2d_irregular_topography",
    "small_2d_visco": None,
    "large_2d_non_planar": "case_large_2d_irregular_topography",
    "small_3d_planar": "case_3d",
    "large_3d_non_planar": None,
}
GAR6_FOLDER = GAR6_FOLDERS[CASE]

GAR6_BASE = os.path.join(SCRIPT_DIR, "results_gar6")
GAR6MORE_DIR_FLUID = f"{GAR6_BASE}/{GAR6_FOLDER}/fluid"
GAR6MORE_DIR_SOLID = f"{GAR6_BASE}/{GAR6_FOLDER}/solid"
GAR6_NAME = "Gar6more3D" if DIMENSION == 3 else "Gar6more2D"
# Thick translucent grey, drawn behind the runs (red is the TTI colour).
GAR6_STYLE = dict(color="dimgray", lw=3.0, ls="-", alpha=0.5)
SHOW_GAR6 = (USE_GAR6 and GAR6_FOLDER is not None
             and os.path.isfile(f"{GAR6MORE_DIR_FLUID}/P.dat"))

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

SPECFEM_BASE = os.path.join(SCRIPT_DIR, "results_specfem")
SPECFEM_NAME = "SPECFEM2D"
SPECFEM_STYLE = dict(color="black", lw=1.2, ls=":")


def solid_tag(rheology: str, medium: str, anisotropy_type: str) -> str:
    """Return the identifier of a solid, as used in run and SPECFEM folder names.

    Parameters
    ----------
    rheology : str
        Solid rheology: "elastic" or "viscoelastic".
    medium : str
        Solid medium: "iso", "vti" or "tti".
    anisotropy_type : str
        Relation used for C13, "exact" or "weak"; ignored for "iso".

    Returns
    -------
    str
        ``<rheology>_<solid>`` (e.g. "elastic_iso", "elastic_tti-exact").
    """
    return f"{rheology}_{solid_name(medium, anisotropy_type)}"


def specfem_dir(solid: str) -> str | None:
    """Return the SPECFEM folder of ``solid`` for CASE, or None if it has no traces.

    Parameters
    ----------
    solid : str
        Solid identifier ``<rheology>_<solid>`` (see `solid_tag`).

    Returns
    -------
    str or None
        ``results_specfem/case_<CASE>-<solid>`` (also accepting "non-planar"
        written with a hyphen), or None when its solid traces are missing.
    """
    for case in (CASE, CASE.replace("non_planar", "non-planar")):
        folder = os.path.join(SPECFEM_BASE, f"case_{case}-{solid}")
        if os.path.isfile(os.path.join(folder, "AA.S0002.BXZ.semd")):
            return folder
    return None


# Solids of RUNS, in order.
SOLIDS = list(dict.fromkeys(solid_tag(*run[:3]) for run in RUNS))

# solid -> SPECFEM folder, for the solids listed in RUNS that have one.
SPECFEM_DIRS = {}
if USE_SPECFEM and DIMENSION == 2:
    for solid in SOLIDS:
        folder = specfem_dir(solid)
        if folder is None:
            print(
                f"[WARNING] no SPECFEM traces in {SPECFEM_BASE}/case_{CASE}-{solid}"
                f" — '{solid}' runs are not compared with SPECFEM."
            )
        else:
            SPECFEM_DIRS[solid] = folder
SHOW_SPECFEM = bool(SPECFEM_DIRS)

# SPECFEM centers the Ricker at t = 0; spyro centers it at t = delay = 1/f.
SPECFEM_DELAY = {
    "small_2d_planar": 1 / 25,
    "small_2d_non_planar": 1 / 25,
    "small_2d_visco": 1 / 25,
    "large_2d_non_planar": 1 / 10,
}.get(CASE, 0.0)

GAR6_DELAY = {
    "small_3d_planar": 2.8e-5,
}.get(CASE, 0.0)

# key -> (file, sign). S0001 = fluid receiver, S0002 = solid receiver.
SPECFEM_FILES = {
    "p": ("AA.S0001.PRE.semp", 1.0),
    "uz": ("AA.S0002.BXZ.semd", 1.0),
    "ux": ("AA.S0002.BXX.semd", 1.0),
}

if SHOW_GAR6 or SHOW_SPECFEM:
    NORMALIZE = True
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===========================================================================
# Helpers
# ===========================================================================
def load_trace(path):
    values = np.loadtxt(path)
    if values.ndim == 1:
        raise ValueError(f"{path}: only 1 column found — expected (time, value).")
    return values[:, 0], values[:, 1]


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour with white.

    Parameters
    ----------
    color : str
        Any matplotlib colour.
    amount : float
        Fraction of white, from 0 (``color`` unchanged) to 1 (white).

    Returns
    -------
    tuple of float
        The mixed colour as RGB.
    """
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def run_style(
    rheology: str, medium: str, anisotropy_type: str, scheme: str
) -> tuple[dict, str]:
    """Return the line style of a run and the figure group it belongs to.

    Parameters
    ----------
    rheology : str
        Solid rheology: "elastic" or "viscoelastic".
    medium : str
        Solid medium: "iso", "vti" or "tti".
    anisotropy_type : str
        Relation used for C13, "exact" or "weak"; ignored for "iso".
    scheme : str
        Coupling scheme: "sequential" or "monolithic".

    Returns
    -------
    dict
        Keyword arguments of ``Axes.plot``.
    str
        Group of the run: its rheology for PLOT_TYPE "solids", its solid
        (e.g. "vti-weak") for PLOT_TYPE "rheology".

    Raises
    ------
    ValueError
        If PLOT_TYPE is not "solids" or "rheology".
    """
    color = MEDIUM_COLORS.get(medium, "tab:gray")
    if PLOT_TYPE == "solids":
        weak = medium != "iso" and anisotropy_type == "weak"
        style = dict(color=color, ls=DOTTED if weak else "-")
        group = rheology
    elif PLOT_TYPE == "rheology":
        if rheology == "viscoelastic":
            style = dict(color=lighten(color, VISCOELASTIC_TINT), ls=DOTTED)
        else:
            style = dict(color=color, ls="-")
        group = solid_name(medium, anisotropy_type)
    else:
        raise ValueError(f"PLOT_TYPE = {PLOT_TYPE!r}: use 'solids' or 'rheology'.")
    style["lw"] = 4.0 if scheme == "monolithic" else 2.0
    if scheme == "monolithic":
        style["alpha"] = 0.5
    return style, group


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


def run_label(
    rheology: str,
    medium: str,
    anisotropy_type: str,
    scheme: str,
    medium_parameters: dict,
    viscoelastic_parameters: dict,
) -> str:
    """Return the legend label of a run.

    Parameters
    ----------
    rheology : str
        Solid rheology: "elastic" or "viscoelastic".
    medium : str
        Solid medium: "iso", "vti" or "tti".
    anisotropy_type : str
        Relation used for C13, "exact" or "weak"; ignored for "iso".
    scheme : str
        Coupling scheme: "sequential" or "monolithic".
    medium_parameters : dict
        Solid parameters stored in receivers.npz (the tilt is read from "theta").
    viscoelastic_parameters : dict
        Viscoelastic parameters stored in receivers.npz (the quality factors
        are read from "quality_factors"); empty for elastic runs.

    Returns
    -------
    str
        Label such as "elastic TTI-exact (θ = 30°) – sequential" or
        "viscoelastic VTI-exact (Q = 30) – sequential".
    """
    solid = solid_name(medium, anisotropy_type).replace(medium, medium.upper(), 1)
    details = []
    if "theta" in medium_parameters:
        details.append(f"θ = {medium_parameters['theta']:g}°")
    quality_factors = viscoelastic_parameters.get("quality_factors", {})
    if medium == "iso":
        quality_factors = {
            k: v for k, v in quality_factors.items() if k in ("Q_vp", "Q_vs")
        }
    if len(set(quality_factors.values())) == 1:
        details.append(f"Q = {next(iter(quality_factors.values())):g}")
    else:
        details += [f"{k} = {v:g}" for k, v in quality_factors.items()]
    label = f"{rheology} {solid}"
    if details:
        label += f" ({', '.join(details)})"
    return f"{label} – {scheme}"


def style_axis(ax, title, ylabel, xlabel=None, legend_outside=False):
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICKS)
    ax.grid(True, alpha=0.3)
    if legend_outside:
        ax.legend(fontsize=FS_LEGEND, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    else:
        ax.legend(fontsize=FS_LEGEND)

# ===========================================================================
# Load cases
# ===========================================================================
print(f"=== Case: {CASE} ({DIMENSION}D) ===")
if SHOW_GAR6:
    print(f"Gar6more fluid: {GAR6MORE_DIR_FLUID}")
    print(f"Gar6more solid: {GAR6MORE_DIR_SOLID}")
    # gar6 = {"p": load_trace(f"{GAR6MORE_DIR_FLUID}/P.dat")}
    # for comp, _, fname, _ in COMPONENTS:
    #     gar6[comp] = load_trace(f"{GAR6MORE_DIR_SOLID}/{fname}")
    t_g, v_g = load_trace(f"{GAR6MORE_DIR_FLUID}/P.dat")
    gar6 = {"p": (t_g + GAR6_DELAY, v_g)}
    for comp, _, fname, _ in COMPONENTS:
        t_g, v_g = load_trace(f"{GAR6MORE_DIR_SOLID}/{fname}")
        gar6[comp] = (t_g + GAR6_DELAY, v_g)
# solid -> {key: (t, v)}. The fluid pressure (S0001.PRE) is compared only
# when SPECFEM wrote it.
spec = {}
for solid, folder in SPECFEM_DIRS.items():
    print(f"SPECFEM ({solid}): {folder}  (time shift +{SPECFEM_DELAY:.4f} s)")
    spec[solid] = {}
    for key, (fname, _) in SPECFEM_FILES.items():
        if os.path.isfile(f"{folder}/{fname}"):
            t_s, v_s = load_trace(f"{folder}/{fname}")
            spec[solid][key] = (t_s + SPECFEM_DELAY, v_s)
if not (SHOW_GAR6 or SHOW_SPECFEM):
    print("No reference for this case — plotting spyro only.")

prep = normalize if NORMALIZE else (lambda x: x)
results = {}
for rheology, medium, anisotropy_type, scheme in RUNS:
    solid = solid_tag(rheology, medium, anisotropy_type)
    tag = f"{solid}_{scheme}"
    path = os.path.join(CASE_DIR, tag, "receivers.npz")
    if not os.path.exists(path):
        print(f"[WARNING] {path} not found — '{tag}' is skipped.")
        continue
    d = np.load(path)
    params = json.loads(str(d["medium_parameters"])) if "medium_parameters" in d else {}
    visco_params = (
        json.loads(str(d["viscoelastic_parameters"]))
        if "viscoelastic_parameters" in d
        else {}
    )
    label = run_label(rheology, medium, anisotropy_type, scheme, params, visco_params)
    style, group = run_style(rheology, medium, anisotropy_type, scheme)
    compare = REFERENCE_SOLID is None or solid == REFERENCE_SOLID
    p = np.asarray(d["p_spyro"]).ravel()
    u = np.asarray(d["u_solid"])

    dt_case = float(d["dt"])
    t_p = np.arange(len(p)) * dt_case
    t_u = np.arange(len(u)) * dt_case

    r = dict(
        tag=tag,
        solid=solid,
        style=style,
        group=group,
        rheology=rheology,
        solid_key=solid_name(medium, anisotropy_type),
        scheme=scheme,
        t_p=t_p,
        t_u=t_u,
        p=prep(p),
        uz_solid_raw=u[:, COL_UZ],
    )
    for comp, col, _, _ in COMPONENTS:
        r[comp] = prep(u[:, col])

    if "u_fluid_z" in d and np.asarray(d["u_fluid_z"]).size:
        r["uz_fluid_raw"] = np.asarray(d["u_fluid_z"]).ravel()

    if SHOW_GAR6 and compare:
        r["p_g"] = ref_on(t_p, *gar6["p"], 1.0 if DIMENSION == 2 else -1.0)
        r["err_p_g"] = MeasureError.calculate_normalized_L2_error(r["p"], r["p_g"])
        for comp, _, _, sign in COMPONENTS:
            r[f"{comp}_g"] = ref_on(t_u, *gar6[comp], sign)
            r[f"err_{comp}_g"] = MeasureError.calculate_normalized_L2_error(
                r[comp], r[f"{comp}_g"])
        r["shift_g"] = estimate_time_shift(t_p, r["p"], r["p_g"])

    if solid in spec:
        spec_m = spec[solid]
        for comp in ("uz", "ux"):
            r[f"{comp}_s"] = ref_on(t_u, *spec_m[comp], SPECFEM_FILES[comp][1])
            r[f"err_{comp}_s"] = MeasureError.calculate_normalized_L2_error(
                r[comp], r[f"{comp}_s"])
        if "p" in spec_m:
            r["p_s"] = ref_on(t_p, *spec_m["p"], SPECFEM_FILES["p"][1])
            r["err_p_s"] = MeasureError.calculate_normalized_L2_error(r["p"], r["p_s"])
            r["shift_s"] = estimate_time_shift(t_p, r["p"], r["p_s"])
        else:
            r["shift_s"] = estimate_time_shift(t_u, r["uz"], r["uz_s"])
    results[label] = r

if not results:
    raise SystemExit(f"No receivers.npz found in {CASE_DIR}.")

comp_keys = [c for c, _, _, _ in COMPONENTS]
REFERENCES = [(tag, name, style) for tag, name, style, show in (
    ("g", GAR6_NAME, GAR6_STYLE, SHOW_GAR6),
    ("s", SPECFEM_NAME, SPECFEM_STYLE, SHOW_SPECFEM),
) if show]

w = max(len(k) for k in results) + 2
for rt, rname, _ in REFERENCES:
    keys = [c for c in ["p"] + comp_keys
            if any(f"err_{c}_{rt}" in r for r in results.values())]
    print(f"\n=== Normalized L2 error vs {rname} ===")
    print(f"{'Run':<{w}}" + "".join(f" {('P' if k == 'p' else 'u_' + k[1]):>9}" for k in keys)
          + f" {'shift (s)':>11}")
    for label, r in results.items():
        if f"shift_{rt}" not in r:
            continue
        errs = (f" {r[f'err_{k}_{rt}']:9.4f}" if f"err_{k}_{rt}" in r else f" {'-':>9}"
                for k in keys)
        print(f"{label:<{w}}" + "".join(errs) + f" {r[f'shift_{rt}']:11.6f}")


def curve_label(label, r, key):
    errs = [f"{rname} {r[f'err_{key}_{rt}'] * 100:.2f}%" for rt, rname, _ in REFERENCES
            if f"err_{key}_{rt}" in r]
    return f"{label}  (L2: {', '.join(errs)})" if errs else label


def plot_references(ax, key, t_key, runs):
    """Draw the Gar6more trace and one SPECFEM trace per solid on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    key : str
        Trace key: "p", "uz", "ux" or "uy".
    t_key : str
        Time-vector key of the runs: "t_p" or "t_u".
    runs : dict
        Runs of the figure, label -> run data.

    Returns
    -------
    None
    """
    compared = [r for r in runs.values() if f"{key}_g" in r]
    if compared:
        ref = compared[0]
        ax.plot(ref[t_key], ref[f"{key}_g"], label=GAR6_NAME, **GAR6_STYLE)
    # Runs of the same solid (e.g. sequential and monolithic) share a trace.
    spec_runs = {r["solid"]: r for r in runs.values() if f"{key}_s" in r}
    for solid, r in spec_runs.items():
        if len(spec_runs) == 1:
            label, color = SPECFEM_NAME, SPECFEM_STYLE["color"]
        else:
            label, color = f"{SPECFEM_NAME} – {solid}", r["style"]["color"]
        ax.plot(r[t_key], r[f"{key}_s"], label=label, **dict(SPECFEM_STYLE, color=color))


amp ="Amplitude (normalized)" if NORMALIZE else "Amplitude"
suffix = "".join(f"_vs_{'gar6' if rt == 'g' else 'specfem'}" for rt, _, _ in REFERENCES)


def figure_tag(group: str, runs: dict) -> str:
    """Return the part of the figure file names that identifies the runs.

    Parameters
    ----------
    group : str
        Group of the runs (a rheology or a solid, see `run_style`).
    runs : dict
        Runs of the figure, label -> run data.

    Returns
    -------
    str
        ``<PLOT_TYPE>-<group>__<members>_<schemes>``, where the members are
        the solids ("solids") or the rheologies ("rheology") of the runs,
        e.g. "solids-elastic__iso_vti-exact_tti-exact_sequential" or
        "rheology-vti-weak__elastic_viscoelastic_sequential". A short, stable
        hash replaces the members when the name would be too long (file names
        are limited to 255 characters).
    """
    member_key = "solid_key" if PLOT_TYPE == "solids" else "rheology"
    members = list(dict.fromkeys(r[member_key] for r in runs.values()))
    schemes = list(dict.fromkeys(r["scheme"] for r in runs.values()))
    runs_tag = "_".join(members + schemes)
    if len(runs_tag) > 150:
        digest = hashlib.sha1(runs_tag.encode()).hexdigest()[:8]
        runs_tag = f"{len(runs)}runs_{digest}"
        print(f"Figures of {', '.join(runs)} are tagged '{runs_tag}'.")
    return f"{PLOT_TYPE}-{group}__{runs_tag}"


def plot_receivers(group: str, runs: dict) -> None:
    """Save the fluid-pressure and solid-displacement figures of some runs.

    Parameters
    ----------
    group : str
        Group of the runs (a rheology or a solid, see `run_style`).
    runs : dict
        Runs drawn together, label -> run data.

    Returns
    -------
    None
    """
    tag = figure_tag(group, runs)
    legend_outside = len(runs) > MAX_LEGEND_INSIDE
    fig_width = 20 if legend_outside else 12

    # Fluid receiver — pressure
    fig_p, ax_p = plt.subplots(figsize=(fig_width, 5))
    plot_references(ax_p, "p", "t_p", runs)
    for label, r in runs.items():
        ax_p.plot(r["t_p"], r["p"], label=curve_label(label, r, "p"), **r["style"])
    style_axis(
        ax_p,
        "Fluid receiver — pressure",
        amp,
        "Time (s)",
        legend_outside=legend_outside,
    )
    fig_p.tight_layout()
    p_path = Path(PLOTS_DIR, f"receiver_fluid_P__{tag}{suffix}")
    save_figure(fig_p, p_path)
    plt.close(fig_p)
    print(f"[OK] Saved to: {p_path.resolve()}.*")

    # Solid receiver — displacement components (2 in 2D, 3 in 3D)
    n_comp = len(comp_keys)
    fig_u, axes = plt.subplots(
        n_comp, 1, figsize=(fig_width, 4.5 * n_comp), sharex=True
    )
    for i, (ax, comp) in enumerate(zip(axes, comp_keys)):
        plot_references(ax, comp, "t_u", runs)
        for label, r in runs.items():
            ax.plot(r["t_u"], r[comp], label=curve_label(label, r, comp), **r["style"])
        style_axis(
            ax,
            f"Solid receiver — displacement u_{comp[1]}",
            amp,
            "Time (s)" if i == n_comp - 1 else None,
            legend_outside=legend_outside,
        )
    fig_u.tight_layout()
    u_path = Path(PLOTS_DIR, f"receiver_solid_u__{tag}{suffix}")
    save_figure(fig_u, u_path)
    plt.close(fig_u)
    print(f"[OK] Saved to: {u_path.resolve()}.*")


# ===========================================================================
# Receivers — one set of figures per group of PLOT_TYPE
# ===========================================================================
groups = {}
for label, r in results.items():
    groups.setdefault(r["group"], {})[label] = r
for group, runs in groups.items():
    print(f"\n--- {PLOT_TYPE}: {group} ({len(runs)} runs) ---")
    plot_receivers(group, runs)

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
        c_path = Path(PLOTS_DIR, f"interface_continuity__{r['tag']}")
        save_figure(fig_c, c_path)
        print(f"[OK] Saved to: {c_path.resolve()}.*")
