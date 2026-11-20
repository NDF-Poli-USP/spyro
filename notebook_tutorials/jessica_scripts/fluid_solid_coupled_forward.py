"""Forward runs of the coupled acoustic-elastic (P - u_s) solver.

Choose the case, the solid medium and the scheme in "Settings" and run from
the repository root (one run per execution):

    python3 notebook_tutorials/jessica_scripts/fluid_solid_coupled_forward.py

Every run is identified by <rheology>_<solid>_<scheme>, where <rheology> is
"elastic" or "viscoelastic" and <solid> is the medium plus, for vti/tti, the
C13 relation (e.g. elastic_iso_sequential, viscoelastic_vti-exact_sequential,
elastic_tti-weak_monolithic), and the outputs are organised by case, so
different runs never overwrite each other:

    results/<case>/mesh/                         mesh, velocity model, mesh/model
                                                 figures
    results/<case>/<rheology>_<solid>_<scheme>/  receivers.npz, pressure/sigma_zz
                                                 .pvd, snapshots/
    results/<case>/plots/                        figures from
                                                 plot_all_results_coupled.py

Figures are written in every format of FIGURE_FORMATS.
"""
import json
import os
import time
import resource
from collections.abc import Callable
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import firedrake as fire
from firedrake import Function, FunctionSpace, triplot, tripcolor

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.solvers.acoustic_solver_construction_no_pml import velocity_fluid
from spyro.utils.utils import communicate
from spyro.io.segy_io import create_segy_from_grid

SCRIPT_DIR = Path(__file__).resolve().parent

# ===========================================================================
# Settings
# ===========================================================================
# "small_2d_planar", "small_2d_non_planar", "small_2d_visco",
# "large_2d_non_planar", "small_3d_planar", "large_3d_non_planar"
CASE = "large_3d_non_planar"

# Solid medium: "iso", "vti" or "tti" (parameters in MEDIA_PARAMETERS).
MEDIUM = "iso"

# C13 from the "exact" or the "weak" Thomsen relation (vti and tti only).
ANISOTROPY_TYPE = "exact"

# "elastic" or "viscoelastic" (parameters in "Viscoelastic solid").
RHEOLOGY = "elastic"

USE_MONOLITHIC = False
RESULTS_DIR = Path("results")

# Figures are saved once per format. Vector formats ("svg", "pdf") keep axes
# and text sharp at any zoom; the dense mesh/model drawings inside them are
# rasterized at FIGURE_DPI so the files stay small.
FIGURE_FORMATS = ("svg",)
FIGURE_DPI = 300
matplotlib.rcParams["svg.fonttype"] = "none"  # editable text in the SVG

# ===========================================================================
# Solid media (Thomsen parameters added to the isotropic properties)
# ===========================================================================
VTI = {"epsilon": 0.3, "delta": 0.15, "gamma": 0.1}
MEDIA_PARAMETERS = {
    "iso": {},
    "vti": VTI,
    "tti": dict(VTI, theta=30.0),  # theta: tilt from the vertical towards +x
}


# ===========================================================================
# Viscoelastic solid (used when RHEOLOGY = "viscoelastic")
# ===========================================================================
# Quality factors of the solid. Q_vp and Q_vs act on every medium; Q_epsilon,
# Q_delta and Q_gamma act only on vti/tti.
QUALITY_FACTORS = {
    "Q_vp": 30.0,
    "Q_vs": 30.0,
    "Q_epsilon": 30.0,
    "Q_delta": 30.0,
    "Q_gamma": 30.0,
}
# Generalized standard linear solid: number of relaxation mechanisms and the
# band, as multiples of the source frequency, where Q is fitted as constant.
GSLS_BRANCHES = 5
GSLS_BAND = (0.1, 10.0)


def gsls_coefficients(
    fmin: float, fmax: float, branches: int, samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the GSLS weights that give a constant Q in a frequency band.

    The relaxation frequencies are spread logarithmically over the band and
    the weights solve, in the least-squares sense,
    ``sum_l y_l omega_l omega / (omega**2 + omega_l**2) = 1`` on ``samples``
    frequencies of the band (as in viscoelastic_wave_propagation.ipynb).

    Parameters
    ----------
    fmin : float
        Lowest frequency of the band, in Hz.
    fmax : float
        Highest frequency of the band, in Hz.
    branches : int
        Number of relaxation mechanisms.
    samples : int, optional
        Number of frequencies used in the fit.

    Returns
    -------
    numpy.ndarray
        Weights ``y_l`` of the mechanisms.
    numpy.ndarray
        Relaxation angular frequencies ``omega_l``, in rad/s.
    """
    omega_relax = 2 * np.pi * np.logspace(np.log10(fmin), np.log10(fmax), branches)
    omega = 2 * np.pi * np.linspace(fmin, fmax, samples)
    A = omega_relax * omega[:, None] / (omega[:, None] ** 2 + omega_relax**2)
    y = np.linalg.lstsq(A, np.ones(samples), rcond=None)[0]
    return y, omega_relax


def viscoelastic_parameters(frequency: float, dt: float) -> tuple[dict, dict]:
    """Return the spyro inputs of the viscoelastic solid.

    Parameters
    ----------
    frequency : float
        Peak frequency of the source, in Hz.
    dt : float
        Time step of the run.

    Returns
    -------
    dict
        Reciprocal quality factors, to be added to ``synthetic_data``.
    dict
        The ``viscoelasticity`` block of the spyro dictionary.

    Raises
    ------
    ValueError
        If ``dt`` is too large for the explicit update of the memory
        variables (``dt * omega_l`` must stay below 1).
    """
    fmin, fmax = (factor * frequency for factor in GSLS_BAND)
    y, omega = gsls_coefficients(fmin, fmax, GSLS_BRANCHES)
    if dt * omega.max() >= 1.0:
        raise ValueError(
            f"dt * omega_max = {dt * omega.max():.3f} >= 1: the memory-variable "
            "update is unstable. Reduce dt or the upper end of GSLS_BAND."
        )
    inverse_Q = {name: 1.0 / Q for name, Q in QUALITY_FACTORS.items()}
    viscoelasticity = {
        "visco_type": "maxwell_gsls_Q",
        "branches": GSLS_BRANCHES,
        "omega_gsls": omega.tolist(),
        "y_gsls": y.tolist(),
    }
    return inverse_Q, viscoelasticity


def solid_parameters(medium: str, anisotropy_type: str) -> dict:
    """Return the synthetic-data parameters of the solid.

    Parameters
    ----------
    medium : str
        Solid medium: "iso", "vti" or "tti".
    anisotropy_type : str
        Relation used for C13, "exact" or "weak"; ignored for "iso".

    Returns
    -------
    dict
        Thomsen parameters (and tilt) of ``medium`` with ``anisotropy_type``
        added for anisotropic media; empty for "iso".
    """
    parameters = MEDIA_PARAMETERS[medium]
    if not parameters:
        return {}
    return dict(parameters, anisotropy_type=anisotropy_type)


def solid_name(medium: str, anisotropy_type: str) -> str:
    """Return the identifier of the solid used in file and folder names.

    Parameters
    ----------
    medium : str
        Solid medium: "iso", "vti" or "tti".
    anisotropy_type : str
        Relation used for C13, "exact" or "weak"; ignored for "iso".

    Returns
    -------
    str
        ``medium`` for "iso", ``<medium>-<anisotropy_type>`` otherwise
        (e.g. "vti-exact").
    """
    return medium if medium == "iso" else f"{medium}-{anisotropy_type}"


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save ``fig`` once per format in FIGURE_FORMATS and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : pathlib.Path
        Output path; its suffix is replaced by each format.

    Returns
    -------
    None
    """
    for fmt in FIGURE_FORMATS:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=FIGURE_DPI)
    plt.close(fig)


# ===========================================================================
# Helpers: velocity models and acquisition
# ===========================================================================
def build_segy(filename, length_z, length_x, dz, interface, v_fluid, v_solid, dx=None):
    dx = dz if dx is None else dx
    z = np.linspace(0.0, -length_z, int(round(length_z / dz)) + 1)
    x = np.linspace(0.0, length_x, int(round(length_x / dx)) + 1)
    X, Z = np.meshgrid(x, z, indexing="ij")
    if callable(interface):
        z_int = np.broadcast_to(interface(x), x.shape)
    elif isinstance(interface, (str, Path)):
        d = np.loadtxt(interface)
        z_int = np.interp(x, d[:, 0], d[:, 1])
    else:
        z_int = np.full(x.shape, float(interface))
    field = lambda v: np.broadcast_to(v(Z, X), Z.shape) if callable(v) else np.full(Z.shape, float(v))
    v = np.where(Z > z_int[:, None], field(v_fluid), field(v_solid))
    create_segy_from_grid(v[:, ::-1].astype(np.float32), str(filename))


def build_binary_3d(
    filename: str | Path,
    nz: int,
    nx: int,
    ny: int,
    length_z: float,
    length_x: float,
    length_y: float,
    interface_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    v_fluid: float,
    v_solid: float,
    top_first: bool = False,
) -> None:
    """Write a raw big-endian float32 model in (nz, nx, ny) C order.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output binary file.
    nz : int
        Number of samples in z.
    nx : int
        Number of samples in x.
    ny : int
        Number of samples in y.
    length_z : float
        Depth of the model, in km.
    length_x : float
        Length of the model in x, in km.
    length_y : float
        Length of the model in y, in km.
    interface_func : callable
        Interface elevation z(x, y), negative downward, in km.
    v_fluid : float
        Velocity above the interface.
    v_solid : float
        Velocity below the interface.
    top_first : bool, optional
        If True, the first z sample is the top (z = 0), as the gmsh
        water-interface mesher reads it. Otherwise it is the bottom, as the
        solver reads the model to split fluid and solid cells when the mesh
        has no cell tags. Default is False.
    """
    if top_first:
        z_arr = np.linspace(0.0, -length_z, nz)
    else:
        z_arr = np.linspace(-length_z, 0.0, nz)
    x_arr = np.linspace(0.0, length_x, nx)
    y_arr = np.linspace(0.0, length_y, ny)
    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")
    Z_int = interface_func(X, Y)
    cube = np.where(
        z_arr[:, np.newaxis, np.newaxis] > Z_int[np.newaxis, :, :],
        v_fluid, v_solid,
    ).astype(">f4")
    cube.tofile(filename)
    print(f"[OK] Binary model written: {filename}  ({nz}z x {nx}x x {ny}y)")


def specfem_interface(filename, index, length_z):
    with open(filename) as f:
        rows = [line.split() for line in f
                if line.strip() and not line.lstrip().startswith("#")]
    it = iter(rows[1:])
    interfaces = []
    for _ in range(int(rows[0][0])):
        n = int(next(it)[0])
        interfaces.append(np.array([[float(a), float(b)] for a, b in (next(it) for _ in range(n))]))
    xs = interfaces[index][:, 0] / 1000.0
    zs = interfaces[index][:, 1] / 1000.0 - length_z
    return lambda x: np.interp(x, xs, zs)


def acquisition_on_interface(z_int, x_source, height_source, offset, distance):
    z_at = lambda x: float(z_int(np.array([x]))[0])
    return {
        "source_locations": [(z_at(x_source) + height_source, x_source)],
        "receiver_location": [(z_at(x_source - offset) + distance, x_source - offset)],
        "solid_receiver_locations": [(z_at(x_source + offset) - distance, x_source + offset)],
    }


def interface_depth(case, x, y=None):
    """Interface z at the given horizontal positions (planar or not)."""
    interface = case["interface"]
    if not callable(interface):
        return np.full(np.shape(x), float(interface))
    return interface(x) if case["dimension"] == 2 else interface(x, y)


def fluid_wave_velocity(case):
    if case["velocity_fluid"] is not None:
        return case["velocity_fluid"]
    return np.sqrt(case["bulk_modulus"] / case["density_fluid"])


# ===========================================================================
# Helpers: figures (2D)
# ===========================================================================
def plot_mesh_xz(meshes, title, filename, case):
    fig, ax = plt.subplots(figsize=(8, 8))
    swap = Affine2D().from_values(0, 1, 1, 0, 0, 0) + ax.transData
    for mesh, color in meshes:
        kw = {"edgecolors": color} if color else {}
        for coll in triplot(mesh, axes=ax, interior_kw=kw):
            coll.set_transform(swap)
            coll.set_rasterized(True)
    ax.set_xlim(0.0, case["length_x"])
    ax.set_ylim(-case["length_z"], 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (km)", fontsize=18)
    ax.set_ylabel("Z (km)", fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    fig.tight_layout()
    save_figure(fig, Path(filename))


def plot_model_xz(wave, case, acquisition, filename):
    vf = Function(FunctionSpace(wave.submesh_fluid, "CG", 1)).interpolate(velocity_fluid(wave))
    vs = Function(FunctionSpace(wave.submesh_solid, "CG", 1)).interpolate(
        ((wave.lmbda + 2 * wave.mu) / wave.rho) ** 0.5)
    vmin = min(vf.dat.data_ro.min(), vs.dat.data_ro.min())
    vmax = max(vf.dat.data_ro.max(), vs.dat.data_ro.max())

    fig, ax = plt.subplots(figsize=(9, 9))
    swap = Affine2D().from_values(0, 1, 1, 0, 0, 0) + ax.transData
    for f in (vf, vs):
        img = tripcolor(f, axes=ax, vmin=vmin, vmax=vmax)
        img.set_transform(swap)
        img.set_rasterized(True)

    for key, marker, color, label in (
        ("source_locations", "*", "green", "Source"),
        ("receiver_locations", "v", "red", "Receiver (fluid)"),
        ("solid_receiver_locations", "^", "orange", "Receiver (solid)"),
    ):
        pts = np.asarray(acquisition[key])
        ax.scatter(pts[:, 1], pts[:, 0], marker=marker, c=color, s=120, label=label, zorder=5)

    ax.set_xlim(0.0, case["length_x"])
    ax.set_ylim(-case["length_z"], 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (km)", fontsize=18)
    ax.set_ylabel("Z (km)", fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="best", fontsize=14)
    cbar = fig.colorbar(img, ax=ax, orientation="horizontal")
    cbar.set_label("Velocity (km/s)", fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    fig.tight_layout()
    save_figure(fig, Path(filename))


# ===========================================================================
# Cases
# ===========================================================================
small_2d_planar = {
    "dimension": 2,
    "mesh_type": "firedrake_mesh",  # "firedrake_mesh" or "gmsh_mesh"
    "cell_type": "T",
    "degree": 4,
    "length_z": 1.0,
    "length_x": 1.0,
    "length_y": 0.0,
    "edge_length": 0.010,
    "interface": -0.5,
    "source_locations": [(-0.45, 0.5)],
    "frequency": 25,
    "receiver_location": [(-0.49, 0.49)],
    "solid_receiver_locations": [(-0.51, 0.51)],
    "final_time": 0.22,
    "dt": 0.0001,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.0,
    "p_wave_velocity": 2.0,
    "s_wave_velocity": 1.2,
    "segy_dz": 0.002,
    "cells_per_wavelength": 6.0,
}

small_2d_non_planar = dict(
    small_2d_planar,
    mesh_type="gmsh_mesh",
    cell_type="Q",
    interface=lambda x: -0.5 + 0.02 * np.sin(2 * np.pi * x / 0.25),
    segy_dz=0.0005,
    segy_dx=0.010,
)

# Planar small 2D case with the solid receiver far from the interface, so that
# the wave travels about 3.6 P (6 S) wavelengths in the solid and the GSLS
# attenuation is visible when comparing the elastic and viscoelastic runs.
small_2d_visco = dict(
    small_2d_non_planar,
    solid_receiver_locations=[(-0.75, 0.65)],
    final_time=0.40,
)

# Planar large 2D case: only used as the base of large_2d_non_planar.
_large_2d = {
    "dimension": 2,
    "mesh_type": "firedrake_mesh",
    "cell_type": "Q",
    "degree": 4,
    "length_z": 12.0,
    "length_x": 30.0,
    "length_y": 0.0,
    "edge_length": 0.040,
    "interface": -6.0,
    "source_locations": [(-5.4, 15.0)],
    "frequency": 10,
    "receiver_location": [(-5.99, 14.99)],
    "solid_receiver_locations": [(-6.01, 15.01)],
    "final_time": 1.0,
    "dt": 0.0005,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
    "segy_dz": 0.005,
    "cells_per_wavelength": 6.0,
}


def large_2d_non_planar():
    interface = specfem_interface(SCRIPT_DIR / "specfem_large_2d_with_bathymetry.dat", 1, 9.6)
    return dict(
        _large_2d,
        mesh_type="gmsh_mesh",
        cell_type="T",
        length_z=9.6,
        length_x=20.0,
        interface=interface,
        segy_dz=0.0005,
        segy_dx=0.040,
        dt=0.00025,
        **acquisition_on_interface(interface, 10.0, 0.6, 0.05, 0.01),
    )


small_3d_planar = {
    "dimension": 3,
    "mesh_type": "firedrake_mesh",
    "cell_type": "Q",
    "degree": 3,
    "length_z": 3.0,
    "length_x": 3.0,
    "length_y": 3.0,
    "edge_length": 0.035,
    "interface": -1.5,
    "source_locations": [(-1.35, 1.5, 1.5)],
    "frequency": 10,
    "receiver_location": [(-1.49, 1.49, 1.49)],
    "solid_receiver_locations": [(-1.51, 1.51, 1.51)],
    "final_time": 0.8,
    "dt": 2.8e-5,
    "velocity_fluid": None,
    "bulk_modulus": 2.25,
    "density_fluid": 1.0,
    "density_solid": 2.5,
    "p_wave_velocity": 3.4,
    "s_wave_velocity": 1.963,
}


def large_3d_non_planar():
    """Sinusoidal 3D interface (formerly case_3d_irregular.py).

    z_int(x, y) = -1.5 + 0.05 sin(2 pi x / 1.5) sin(2 pi y / 1.5), in km.
    Source 0.15 km above the interface at the domain centre; receivers 0.01 km
    above/below it with a 0.01 km lateral offset in x and y.
    """
    interface = lambda x, y: -1.5 + 0.05 * np.sin(2 * np.pi * x / 1.5) * np.sin(2 * np.pi * y / 1.5)
    xc, yc, lateral = 1.5, 1.5, 0.01
    zc = float(interface(xc, yc))
    xr, yr = xc - lateral, yc - lateral
    return dict(
        small_3d_planar,
        mesh_type="gmsh_mesh",
        interface=interface,
        water_interface=True,
        source_locations=[(zc + 0.15, xc, yc)],
        receiver_location=[(zc + lateral, xr, yr)],
        solid_receiver_locations=[(zc - 0.1, xr, yr)],
        output_frequency=2500,
        dt=2.5e-5,
        segy_dz=0.005,
        segy_dx=0.025,
        segy_dy=0.025,
        cells_per_wavelength=4.0,  # >= 3.72 required for degree-3 tetrahedra (KMV)
        min_element_size=0.025,
        padding=0.3,
    )


def small_3d_non_planar() -> dict:
    """Return a 0.5 km cube case with a sinusoidal interface the mesh follows.

    Returns
    -------
    dict
        Case parameters, with the interface
        z_int(x, y) = -0.25 + 0.02 sin(2 pi x / 0.25) sin(2 pi y / 0.25), in km.

    Notes
    -----
    The gmsh mesher reads the interface from the binary model, so its depth is
    rounded down to the segy_dz grid (error up to segy_dz). segy_dx and segy_dy
    only set the number of interface points.
    """

    def interface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the interface elevation.

        Parameters
        ----------
        x : numpy.ndarray
            x coordinates, in km.
        y : numpy.ndarray
            y coordinates, in km.

        Returns
        -------
        numpy.ndarray
            Interface elevation z, in km.
        """
        return -0.25 + 0.02 * np.sin(2 * np.pi * x / 0.25) * np.sin(
            2 * np.pi * y / 0.25
        )

    xc, yc, lateral = 0.25, 0.25, 0.01
    zc = float(interface(xc, yc))
    xr, yr = xc - lateral, yc - lateral
    return dict(
        small_3d_planar,
        mesh_type="gmsh_mesh",
        length_z=0.5,
        length_x=0.5,
        length_y=0.5,
        interface=interface,
        water_interface=True,
        source_locations=[(zc + 0.15, xc, yc)],
        receiver_location=[(zc + lateral, xr, yr)],
        solid_receiver_locations=[(zc - lateral, xr, yr)],
        final_time=0.4,
        dt=2.5e-5,
        output_frequency=2000,
        segy_dz=0.002,
        segy_dx=0.010,
        segy_dy=0.010,
        cells_per_wavelength=4.0,  # >= 3.72 required for degree-3 tetrahedra (KMV)
        min_element_size=0.025,
        padding=0.0,
    )


CASES = {
    "small_2d_planar": small_2d_planar,
    "small_2d_non_planar": small_2d_non_planar,
    "small_2d_visco": small_2d_visco,
    "large_2d_non_planar": large_2d_non_planar,
    "small_3d_planar": small_3d_planar,
    "small_3d_non_planar": small_3d_non_planar,
    "large_3d_non_planar": large_3d_non_planar,
}


# ===========================================================================
# Dictionary
# ===========================================================================
def mesh_dictionary(case, mesh_dir):
    interface = case["interface"]
    mesh = {
        "length_z": case["length_z"],
        "length_x": case["length_x"],
        "length_y": case["length_y"],
        "mesh_file": None,
        "mesh_type": case["mesh_type"],
        "interface_z": None if callable(interface) else interface,
    }
    if case["mesh_type"] == "firedrake_mesh":
        mesh["edge_length"] = case["edge_length"]
        return mesh

    vp_water = fluid_wave_velocity(case)
    mesh.update({
        "output_filename": str(mesh_dir / "mesh.msh"),
        "cells_per_wavelength": case["cells_per_wavelength"],
        "frequency": case["frequency"],
        "min_element_size": case.get("min_element_size", case["edge_length"]),
        "grade": 0.05,
        "apply_winslow": False,
        "winslow_iterations": 2000,
        "vp_water": vp_water,
        "water_search_value": vp_water,
    })
    if case["dimension"] == 2:
        model = mesh_dir / "model.segy"
        build_segy(model, case["length_z"], case["length_x"], case["segy_dz"],
                   interface, vp_water, case["s_wave_velocity"], dx=case.get("segy_dx"))
        mesh.update({
            "segy_velocity_model": str(model),
            "structured_mesh": case["cell_type"] == "Q",
            "water_interface": True,
        })
    else:
        model = mesh_dir / "model.bin"
        water_interface = case.get("water_interface", False)
        n = {ax: int(round(case[f"length_{ax}"] / case[f"segy_d{ax}"])) + 1 for ax in "zxy"}
        build_binary_3d(
            model,
            n["z"],
            n["x"],
            n["y"],
            case["length_z"],
            case["length_x"],
            case["length_y"],
            interface,
            vp_water,
            case["p_wave_velocity"],
            top_first=water_interface,
        )
        if water_interface:
            mesh["segy_axes_order_sort"] = "C"
        mesh.update({
            "segy_velocity_model": str(model),
            "structured_mesh": False,
            "water_interface": water_interface,
            **{f"segy_n{ax}": n[ax] for ax in "zxy"},
            **{f"segy_d{ax}": case[f"segy_d{ax}"] for ax in "zxy"},
            **{f"padding_{ax}": case["padding"] for ax in "xyz"},
        })
    return mesh


def build_dictionary(case, medium, anisotropy_type, rheology, mesh_dir, run_dir):
    dt = case["dt"]
    dictionary = {
        "options": {
            "cell_type": case["cell_type"],
            "variant": "lumped",
            "degree": case["degree"],
            "dimension": case["dimension"],
        },
        "parallelism": {"type": "automatic"},
        "mesh": mesh_dictionary(case, mesh_dir),
        "acquisition": {
            "source_type": "ricker",
            "source_locations": case["source_locations"],
            "frequency": case["frequency"],
            "delay": 1.0 / case["frequency"],
            "delay_type": "time",
            "amplitude": 1.0,
            "receiver_locations": case["receiver_location"],
            "solid_receiver_locations": case["solid_receiver_locations"],
            "use_vertex_only_mesh": False,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": case["final_time"],
            "dt": dt,
            "output_frequency": case.get("output_frequency", int(round(0.005 / dt))),
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": True,
            "forward_output_filename": str(run_dir / "pressure.pvd"),
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "graadient_output": False,
            "gradient_filename": None,
            "debug_output": False,
            "displacement_output": False,
            "displacement_output_filename": str(run_dir / "displacement.pvd"),
            "snapshot_frequency": 20,
            "snapshot_output_dir": str(run_dir / "snapshots"),
            "p_equivalent_output": False,
            "p_equivalent_output_filename": str(run_dir / "p_equivalent.pvd"),
            "interface_error_frequency": 20,
            "sigma_zz_output": True,
            "sigma_zz_output_filename": str(run_dir / "sigma_zz.pvd"),
        },
        "synthetic_data": {
            "type": "object",
            "velocity_fluid": case["velocity_fluid"],
            "bulk_modulus": case["bulk_modulus"],
            "density_fluid": case["density_fluid"],
            "density_solid": case["density_solid"],
            "p_wave_velocity": case["p_wave_velocity"],
            "s_wave_velocity": case["s_wave_velocity"],
            "real_velocity_file": None,
            **solid_parameters(medium, anisotropy_type),
        },
        "absorving_boundary_conditions": {
            "status": False,
            "abc_type": "nrbc",
            "absorb_top": True,
            "absorb_bottom": True,
            "absorb_right": True,
            "absorb_left": True,
        },
    }
    if rheology == "viscoelastic":
        inverse_Q, viscoelasticity = viscoelastic_parameters(case["frequency"], dt)
        dictionary["synthetic_data"].update(inverse_Q)
        dictionary["viscoelastic"] = True
        dictionary["viscoelasticity"] = viscoelasticity
    elif rheology != "elastic":
        raise ValueError(f"RHEOLOGY = {rheology!r}: use 'elastic' or 'viscoelastic'.")
    return dictionary


# ===========================================================================
# Run
# ===========================================================================
def check_interface(wave, case):
    """Print submesh markers, cells on the wrong side and the interface size."""
    print("Fluid exterior facet markers:",
          wave.scalar_function_space.mesh().exterior_facets.unique_markers)
    print("Solid exterior facet markers:",
          wave.vector_function_space.mesh().exterior_facets.unique_markers)
    for name, sub in (("fluid", wave.submesh_fluid), ("solid", wave.submesh_solid)):
        V = fire.VectorFunctionSpace(sub, "DG", 0)
        c = fire.Function(V).interpolate(fire.SpatialCoordinate(sub)).dat.data_ro
        y = c[:, 2] if case["dimension"] == 3 else None
        zi = interface_depth(case, c[:, 1], y)
        wrong = (c[:, 0] < zi) if name == "fluid" else (c[:, 0] > zi)
        print(f"{name} - cells on the wrong side: {wrong.sum()} of {len(c)}")
    size = fire.assemble(fire.Constant(1.0) * wave.ds_int(wave.interface_id))
    print(f"interface {'area' if case['dimension'] == 3 else 'length'}: {size}")


def run(case_name, medium, anisotropy_type, rheology, use_monolithic):
    case = CASES[case_name]
    if callable(case):
        case = case()
    scheme = "monolithic" if use_monolithic else "sequential"
    solid = solid_name(medium, anisotropy_type)
    run_name = f"{rheology}_{solid}_{scheme}"

    case_dir = RESULTS_DIR / case_name
    mesh_dir = case_dir / "mesh"
    run_dir = case_dir / run_name
    for d in (mesh_dir, run_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n========== {case_name} | {run_name} ==========")
    dictionary = build_dictionary(
        case, medium, anisotropy_type, rheology, mesh_dir, run_dir
    )

    wave = AcousticElasticWave(dictionary=dictionary)
    check_interface(wave, case)
    wave.use_monolithic = use_monolithic

    t_start = time.perf_counter()
    wave.forward_solve()
    elapsed = time.perf_counter() - t_start
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"Case: {case_name} | run: {run_name}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Memory: {mem_mb:.2f} MB")

    p_spyro = np.asarray(wave.forward_solution_receivers)[:, 0]
    u_all = communicate(np.asarray(wave.solid_receiver_history), wave.comm)
    u_solid = u_all[:, 0, :]

    n_procs = wave.comm.comm.size
    rank = wave.comm.comm.rank
    if rank == 0:
        out = run_dir / "receivers.npz"
        np.savez(
            out,
            dt=dictionary["time_axis"]["dt"],
            final_time=dictionary["time_axis"]["final_time"],
            p_spyro=p_spyro,
            u_solid=u_solid,
            case=case_name,
            medium=medium,
            solid=solid,
            rheology=rheology,
            scheme=scheme,
            dimension=case["dimension"],
            medium_parameters=json.dumps(solid_parameters(medium, anisotropy_type)),
            viscoelastic_parameters=json.dumps(
                {
                    "quality_factors": QUALITY_FACTORS,
                    **dictionary["viscoelasticity"],
                }
                if rheology == "viscoelastic"
                else {}
            ),
            elapsed_time=elapsed,
            memory_mb=mem_mb,
        )
        print(f"[OK] Saved: {out}")

    # The mesh and the velocity figure do not depend on the medium.
    if case["dimension"] == 2 and n_procs == 1:
        plot_model_xz(wave, case, dictionary["acquisition"], mesh_dir / "model")
        plot_mesh_xz([(wave.mesh, None)], "Parent mesh", mesh_dir / "mesh_parent", case)
        plot_mesh_xz(
            [(wave.submesh_fluid, "blue"), (wave.submesh_solid, "orange")],
            "Child submeshes",
            mesh_dir / "mesh_child",
            case,
        )


if __name__ == "__main__":
    run(CASE, MEDIUM, ANISOTROPY_TYPE, RHEOLOGY, USE_MONOLITHIC)
