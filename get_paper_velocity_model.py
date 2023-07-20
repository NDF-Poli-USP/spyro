import firedrake as fire
from firedrake import And, conditional, File
import numpy as np

def apply_box(mesh, c, x1, y1, x2, y2, value):
    x, y = fire.SpatialCoordinate(mesh)
    box = fire.conditional(And(And(x1<=x, x<=x2), And(y1<=y, y<=y2)), value, c)
    print("a")
    c.interpolate(box)
    return c

# def apply_triangle(mesh, c, x1, y1, x3, y3, value):
#     x, y = fire.SpatialCoordinate(mesh)
#     slope = (y3-y1)/(x3-x1)
#     print(slope)
#     # triangle = fire.conditional(And( (y-y1)/(x-x1) <= slope , And(y1<=y, x<=x3)), value, c)
#     triangle = fire.conditional(And( (y-y1)/(x-x1) <= slope, (y-y1)/(x-x1) >= 0.1) , value, c)
#     c.interpolate(triangle)
#     return c

def apply_slope(mesh, c, x1, y1, x3, y3, value):
    x, y = fire.SpatialCoordinate(mesh)
    slope = (y3-y1)/(x3-x1)
    print(slope)
    slope = fire.conditional(And( (y-y1)/(x-x1) <= slope, x >= x1 ), value, c)
    c.interpolate(slope)
    return c

def apply_vs_from_list(velmat, mesh, Lx, Ly, c):
    # (x1, y1, x2, y2, cm)
    for box in velmat:
        x1 = box[0]*Lx
        y1 = box[1]*Ly
        x2 = box[2]*Lx
        y2 = box[3]*Ly
        cm = box[4]
        c = apply_box(mesh, c, x1, y1, x2, y2, cm)
    
    return c

def get_velocity_model(V):

    velmat = []
    velmat.append([0.00, 0.00, 0.35, 0.10, 2.9])
    velmat.append([0.00, 0.10, 0.25, 0.30, 2.9])
    velmat.append([0.00, 0.30, 0.25, 0.70, 2.0])
    velmat.append([0.00, 0.70, 0.10, 1.00, 3.7])
    velmat.append([0.10, 0.70, 0.30, 0.90, 3.7])
    velmat.append([0.25, 0.10, 0.75, 0.30, 2.5])
    velmat.append([0.25, 0.30, 0.40, 0.70, 2.5])
    velmat.append([0.35, 0.00, 0.70, 0.10, 2.1])
    velmat.append([0.70, 0.00, 0.90, 0.10, 3.4])
    velmat.append([0.80, 0.10, 0.90, 0.35, 3.4])
    velmat.append([0.90, 0.00, 1.00, 0.20, 3.4])
    velmat.append([0.90, 0.20, 1.00, 0.65, 2.6])
    velmat.append([0.75, 0.10, 0.80, 0.50, 4.0])
    velmat.append([0.80, 0.35, 0.90, 0.80, 4.0])
    velmat.append([0.85, 0.80, 0.90, 0.95, 3.6])
    velmat.append([0.90, 0.65, 1.00, 1.00, 3.6])
    velmat.append([0.00, 0.00, 0.00, 0.00, 1.5])  

    Lx = 4.8
    Ly = 2.4
    mesh = fire.RectangleMesh(480,240,Lx,Ly, diagonal="crossed")
    c = fire.Function(V)

    c.dat.data[:] = 1.5

    c = apply_slope(mesh, c, 0.4*Lx, 0.3*Ly, 0.75*Lx, 0.65*Ly, 3.3)
    # c = apply_vs_from_list(velmat, mesh, Lx, Ly, c)

    File("testingvelmodel.pvd").write(c)
    return c


Lx = 4.8
Ly = 2.4

mesh = fire.RectangleMesh(480,240,Lx,Ly, diagonal="crossed")
V = fire.FunctionSpace(mesh, 'DG', 0)

c = get_velocity_model(V)



