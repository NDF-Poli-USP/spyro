import firedrake as fire
from firedrake import And, conditional, File
import numpy as np
import spyro
from SeismicMesh import write_velocity_model
import SeismicMesh
import meshio


def apply_box(mesh, c, x1, y1, x2, y2, value):
    y1 = 2.4 - y1
    y2 = 2.4 - y2
    x1 = 4.8 - x1
    x2 = 4.8 - x2
    y, x = fire.SpatialCoordinate(mesh)
    box = fire.conditional(And(And(x1>=x, x>=x2), And(y1>=-y, -y>=y2)), value, c)
    print("a")
    c.interpolate(box)
    return c

def apply_slope(mesh, c, x1, y1, x3, y3, value):
    y, x = fire.SpatialCoordinate(mesh)
    y1 = 2.4 - y1
    y3 = 2.4 - y3
    x1 = 4.8 - x1
    x3 = 4.8 - x3
    slope = (y3-y1)/(x3-x1)
    print(slope)
    slope = fire.conditional(And( (-y-y1)/(x-x1) <= slope, x < x1 ), value, c)
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


def get_paper_velocity(mesh, V, output=False):

    velmat = []
    velmat.append([0.00, 0.00, 0.35, 0.10, 2.9*1000])
    velmat.append([0.00, 0.10, 0.25, 0.30, 2.9*1000])
    velmat.append([0.00, 0.30, 0.25, 0.70, 2.0*1000])
    velmat.append([0.00, 0.70, 0.10, 1.00, 3.7*1000])
    velmat.append([0.10, 0.70, 0.30, 0.90, 3.7*1000])
    velmat.append([0.25, 0.10, 0.75, 0.30, 2.5*1000])
    velmat.append([0.25, 0.30, 0.40, 0.70, 2.5*1000])
    velmat.append([0.35, 0.00, 0.70, 0.10, 2.1*1000])
    velmat.append([0.70, 0.00, 0.90, 0.10, 3.4*1000])
    velmat.append([0.80, 0.10, 0.90, 0.35, 3.4*1000])
    velmat.append([0.90, 0.00, 1.00, 0.20, 3.4*1000])
    velmat.append([0.90, 0.20, 1.00, 0.65, 2.6*1000])
    velmat.append([0.75, 0.10, 0.80, 0.50, 4.0*1000])
    velmat.append([0.80, 0.35, 0.90, 0.80, 4.0*1000])
    velmat.append([0.85, 0.80, 0.90, 0.95, 3.6*1000])
    velmat.append([0.90, 0.65, 1.00, 1.00, 3.6*1000])
    velmat.append([0.00, 0.00, 0.00, 0.00, 1.5*1000])  

    Lx = 4.8
    Ly = 2.4

    c = fire.Function(V)

    c.dat.data[:] = 1.5*1000

    c = apply_slope(mesh, c, 0.4*Lx, 0.3*Ly, 0.75*Lx, 0.65*Ly, 3.3*1000)
    c = apply_vs_from_list(velmat, mesh, Lx, Ly, c)

    if output == True:
        File("testing.pvd").write(c)

    return c


