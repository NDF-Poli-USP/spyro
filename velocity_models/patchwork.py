import firedrake as fire
from firedrake import And, VTKFile
import ipdb


def apply_box(mesh, c, x1, y1, x2, y2, value, tol=1e-3):
    x1r = round(x1, 2) - tol
    x2r = round(x2, 2) + tol
    y1r = -(round(2.4 - y1, 2) + tol)
    y2r = -(round(2.4 - y2, 2) - tol)
    y, x = fire.SpatialCoordinate(mesh)
    box = fire.conditional(
        And(And(x >= x1r, x <= x2r),
            And(y >= y1r, y <= y2r)), value, c)
    c.interpolate(box, allow_missing_dofs=True)

    return c


def apply_slope(mesh, c, x1, y1, x3, y3, value):
    y, x = fire.SpatialCoordinate(mesh)
    y1 = round(2.4 - y1, 2)
    y3 = round(2.4 - y3, 2)
    slope = (y3 - y1) / (x3 - x1)
    slope = fire.conditional(
        And((-y - y1) / (x - x1) >= slope, x > x1), value, c)
    c.interpolate(slope, allow_missing_dofs=True)

    return c


def apply_vs_from_list(velmat, mesh, Lx, Ly, c):
    # (x1, y1, x2, y2, cm)
    for box in velmat:
        x1 = round(box[0] * Lx, 2)
        y1 = round(box[1] * Ly, 2)
        x2 = round(box[2] * Lx, 2)
        y2 = round(box[3] * Ly, 2)
        cm = box[4]
        c = apply_box(mesh, c, x1, y1, x2, y2, cm)

    return c


def get_velocity_model(mesh, V, output=True, units='km/s'):
    if units == 'km/s':
        multiplier = 1.0
    elif units == 'm/s':
        multiplier = 1000.0
    velmat = []
    velmat.append([0.00, 0.00, 0.35, 0.10, 2.9 * multiplier])
    velmat.append([0.00, 0.10, 0.25, 0.30, 2.9 * multiplier])
    velmat.append([0.00, 0.30, 0.25, 0.70, 2.0 * multiplier])
    velmat.append([0.00, 0.70, 0.10, 1.00, 3.7 * multiplier])
    velmat.append([0.10, 0.70, 0.30, 0.90, 3.7 * multiplier])
    velmat.append([0.25, 0.10, 0.75, 0.30, 2.5 * multiplier])
    velmat.append([0.25, 0.30, 0.40, 0.70, 2.5 * multiplier])
    velmat.append([0.35, 0.00, 0.70, 0.10, 2.1 * multiplier])
    velmat.append([0.70, 0.00, 0.90, 0.10, 3.4 * multiplier])
    velmat.append([0.80, 0.10, 0.90, 0.35, 3.4 * multiplier])
    velmat.append([0.90, 0.00, 1.00, 0.20, 3.4 * multiplier])
    velmat.append([0.90, 0.20, 1.00, 0.65, 2.6 * multiplier])
    velmat.append([0.75, 0.10, 0.80, 0.50, 4.0 * multiplier])
    velmat.append([0.80, 0.35, 0.90, 0.80, 4.0 * multiplier])
    velmat.append([0.85, 0.80, 0.90, 0.95, 3.6 * multiplier])
    velmat.append([0.90, 0.65, 1.00, 1.00, 3.6 * multiplier])

    Lx = 4.8
    Ly = 2.4

    c = fire.Function(V)
    c.dat.data[:] = 1.5 * multiplier
    c = apply_slope(mesh, c, 0.4 * Lx, 0.3 * Ly, 0.75 * Lx, 0.65 * Ly,
                    3.3 * multiplier)
    c = apply_vs_from_list(velmat, mesh, Lx, Ly, c)

    if output is True:
        VTKFile("testing.pvd").write(c)

    return c
