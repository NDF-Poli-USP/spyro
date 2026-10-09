import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECFEM_FILE = "specfem_large_2d_with_bathymetry.dat"  # SPECFEM2D interfaces file
SEAFLOOR_INDEX = 1 # 0 = bottom, 1 = seafloor, 2 = top

# ===========================================================================
# Cases (km)
# ===========================================================================
CASES = {
    "case_small_2d_without_bathymetry": dict(
        dim=2, length_z=1.0, length_x=1.0, x0=0.5, a=0.0, l=0.25,
        edge_length=0.010),
    "case_large_2d_without_bathymetry": dict(
        dim=2, length_z=30.0, length_x=12.0, x0=6.0, a=0.0, l=2.5,
        edge_length=0.040),
    "case_3d_without_bathymetry": dict(
        dim=3, length_z=6.0, length_x=4.0, length_y=6.0, x0=2.5, a=0.0, l=1.0,
        edge_length=0.050),
    "case_small_2d_with_bathymetry": dict(
        dim=2, length_z=1.0, length_x=1.0, x0=0.5, a=0.02, l=0.25,
        edge_length=0.010),
    "case_large_2d_with_bathymetry": dict(
        dim=2, length_z=20.0, length_x=9.6, specfem=SPECFEM_FILE),
    "case_3d_with_bathymetry": dict(
        dim=3, length_z=6.0, length_x=4.0, length_y=6.0, x0=2.5, a=0.05, l=1.5,
        edge_length=0.050),
}


def n_points(length, edge_length):
    return int(round(length / edge_length)) + 1


def interface_2d(c):
    z = np.linspace(-c["length_z"], 0.0, n_points(c["length_z"], c["edge_length"]))
    x = c["x0"] + c["a"] * np.sin(2 * np.pi * z / c["l"])
    return z, x


def interface_3d(c):
    z, y = np.meshgrid(
        np.linspace(-c["length_z"], 0.0, n_points(c["length_z"], c["edge_length"])),
        np.linspace(0.0, c["length_y"], n_points(c["length_y"], c["edge_length"])),
        indexing="ij")
    x = (c["x0"] + c["a"] * np.sin(2 * np.pi * z / c["l"])
         * np.sin(2 * np.pi * y / c["l"]))
    return z, y, x


def interface_specfem(c):
    with open(os.path.join(OUT_DIR, c["specfem"])) as f:
        rows = [line.split() for line in f
                if line.strip() and not line.lstrip().startswith("#")]
    it = iter(rows[1:])
    interfaces = []
    for _ in range(int(rows[0][0])):
        n = int(next(it)[0])
        interfaces.append(np.array([[float(a), float(b)]
                                    for a, b in (next(it) for _ in range(n))]))
    # SPECFEM (x, z) in metres -> Spyro (z, x) in km
    sea = interfaces[SEAFLOOR_INDEX]
    return sea[:, 0] / 1000.0 - c["length_z"], sea[:, 1] / 1000.0


def plot_2d(name, c, z, x):
    fig, ax = plt.subplots(figsize=(10, 10 * c["length_x"] / c["length_z"] + 1.5))
    ax.fill_between(z, 0.0, x, color="tan", alpha=0.6, label="solid")
    ax.fill_between(z, x, c["length_x"], color="lightblue", alpha=0.6, label="fluid")
    ax.plot(z, x, color="black", lw=1.2, label="interface")
    ax.set_xlim(-c["length_z"], 0.0)
    ax.set_ylim(0.0, c["length_x"])
    ax.set_aspect("equal")
    ax.set_xlabel("Z (km)", fontsize=14)
    ax.set_ylabel("X (km)", fontsize=14)
    ax.set_title(name, fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    return fig


def plot_3d(name, c, z, y, x):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(z, y, x, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="interface X (km)")
    ax.set_aspect("equal")
    ax.set_xlabel("Z (km)", fontsize=14)
    ax.set_ylabel("Y (km)", fontsize=14)
    ax.set_title(name, fontsize=14)
    return fig


for name, c in CASES.items():
    txt_path = os.path.join(OUT_DIR, f"interface_points_{name}.txt")
    png_path = os.path.join(OUT_DIR, f"interface_points_{name}.png")

    if c["dim"] == 3:
        z, y, x = interface_3d(c)
        np.savetxt(txt_path, np.column_stack([z.ravel(), y.ravel(), x.ravel()]),
                   fmt="%.6f",
                   header=f"z (km)   y (km)   x (km)   [grid {z.shape[0]} x {z.shape[1]}]")
        fig = plot_3d(name, c, z, y, x)
    else:
        z, x = interface_specfem(c) if "specfem" in c else interface_2d(c)
        np.savetxt(txt_path, np.column_stack([z, x]), fmt="%.6f",
                   header="z (km)   x (km)")
        fig = plot_2d(name, c, z, x)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[OK] {name}: {os.path.basename(txt_path)}, {os.path.basename(png_path)}")