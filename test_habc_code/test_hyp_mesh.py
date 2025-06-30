from firedrake import *
from netgen.geom2d import SplineGeometry
import numpy as np


# Function to define the hyperellipse boundary points
def parametric_hyperellipse(a, b, n, num_pts):
    r_ang = np.linspace(0, 2 * np.pi, num_pts)

    rc_zero = [np.pi / 2., 3 * np.pi / 2.]
    rs_zero = [0., np.pi, 2 * np.pi]

    cr = np.cos(r_ang)
    sr = np.sin(r_ang)
    cr = np.where(np.isin(r_ang, rc_zero), 0, cr)
    sr = np.where(np.isin(r_ang, rs_zero), 0, sr)

    x = a * np.sign(cr) * np.abs(cr)**(2/n)
    y = b * np.sign(sr) * np.abs(sr)**(2/n)
    return np.column_stack((x, y))


# Parameters for the hyperellipse
a = 2.0          # Semi-major axis
b = 1.0          # Semi-minor axis
n = 192          # Degree of the hyperellipse
lmax = 0.05      # Maximum edge length
lmin = lmax / 2  # Minimum edge length

# Generate the hyperellipse boundary points
num_bnd_pts = 512  # 16 24
num_bnd_pts += 1 if num_bnd_pts % 2 == 0 else 2  # to close the curve
bnd_pts = parametric_hyperellipse(a, b, n, num_bnd_pts)
# print(bnd_pts)

geo = SplineGeometry()
# Append points to the geometry
[geo.AppendPoint(*pnt) for pnt in bnd_pts]

# Generate the boundary curves
curves = []
for idp in range(0, num_bnd_pts - 1, 2):
    p1 = geo.PointData()[2][idp]
    p2 = geo.PointData()[2][idp + 1]
    p3 = geo.PointData()[2][idp + 2]
    curves.append(["spline3", p1, p2, p3])
    # print(p1, p2, p3)
[geo.Append(c, bc="hyp", maxh=lmin) for c in curves]

# Generate the mesh
ngmsh = geo.GenerateMesh(maxh=lmax, quad_dominated=False)
msh = Mesh(ngmsh)
VTKFile("output/hyp_test.pvd").write(msh)

# f"O valor  vale {r}"
# f"The price is {price:.2f} dollars"
