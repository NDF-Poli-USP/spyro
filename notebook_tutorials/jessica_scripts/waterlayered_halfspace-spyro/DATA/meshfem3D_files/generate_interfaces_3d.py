"""
Generate interfaces.dat and interface_sine.dat for SPECFEM3D
Case: large_3d_non_planar
Interface: z(x,y) = -1500 + 50*sin(2*pi*x/1500)*sin(2*pi*y/1500) meters
Domain: 3000 x 3000 x 3000 m
"""
import numpy as np

# Domain parameters (in meters)
LX = 3000.0
LY = 3000.0

# Interface parameters
z0 = -1500.0       # mean depth (m)
amp = 50.0         # amplitude (m)
lam = 1500.0       # wavelength (m)

# Grid resolution for interface (must match NEX_XI and NEX_ETA or be finer)
NXI  = 41   # number of points along X
NETA = 41   # number of points along Y

x = np.linspace(0.0, LX, NXI)
y = np.linspace(0.0, LY, NETA)

SPACING_XI  = x[1] - x[0]
SPACING_ETA = y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing='ij')
Z = z0 + amp * np.sin(2*np.pi*X/lam) * np.sin(2*np.pi*Y/lam)

print(f"Interface z range: {Z.min():.2f} to {Z.max():.2f} m")
print(f"Grid: {NXI} x {NETA}, spacing {SPACING_XI:.2f} x {SPACING_ETA:.2f} m")

# Write interface_sine.dat (Z values, row by row in ETA direction)
with open("interface_sine.dat", "w") as f:
    for ix in range(NXI):
        for ieta in range(NETA):
            f.write(f"{Z[ix, ieta]:.4f}\n")

print("Written: interface_sine.dat")

# Write interfaces.dat
# Format: 2 interfaces needed:
#   1 = sinusoidal fluid-solid interface
#   2 = top surface (flat at z=0)
# NZ layers: 20 elements in solid (bottom), 20 in fluid (top)
NZ_SOLID = 20
NZ_FLUID = 20

with open("interfaces.dat", "w") as f:
    f.write("# number of interfaces\n")
    f.write(" 2\n\n")

    f.write("# Interface 1: sinusoidal fluid-solid interface\n")
    f.write("# SUPPRESS_UTM_PROJECTION  NXI  NETA  LONG_MIN  LAT_MIN  SPACING_XI  SPACING_ETA\n")
    f.write(f" .true.  {NXI}  {NETA}  0.d0  0.d0  {SPACING_XI:.4f}d0  {SPACING_ETA:.4f}d0\n")
    f.write(" interface_sine.dat\n\n")

    f.write("# Interface 2: top surface (flat at z=0)\n")
    f.write("# SUPPRESS_UTM_PROJECTION  NXI  NETA  LONG_MIN  LAT_MIN  SPACING_XI  SPACING_ETA\n")
    f.write(f" .true.  2  2  0.d0  0.d0  {LX:.1f}d0  {LY:.1f}d0\n")
    f.write(" interface_top.dat\n\n")

    f.write("# number of spectral elements in vertical direction for each layer\n")
    f.write(f"# layer 1 (solid, bottom)\n")
    f.write(f" {NZ_SOLID}\n")
    f.write(f"# layer 2 (fluid, top)\n")
    f.write(f" {NZ_FLUID}\n")

print("Written: interfaces.dat")

# Write interface_top.dat (flat top surface at z=0)
with open("interface_top.dat", "w") as f:
    for _ in range(4):  # 2x2 grid
        f.write("0.0\n")

print("Written: interface_top.dat")
print("Done!")
