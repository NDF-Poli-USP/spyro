import firedrake as fire
import SeismicMesh as sm
import numpy as np

def RectangleMesh(Lx, Ly, nx, ny, pad = None, comm=None, quadrilateral=False):
    if pad != None:
        Lx += pad
        Ly += 2*pad
    else:
        pad = 0
    mesh = fire.RectangleMesh(Lx, Ly, nx, ny, quadrilateral=quadrilateral)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh

def PeriodicRectangleMesh(Lx, Ly, nx, ny, pad = None, comm=None, quadrilateral=False):
    if pad != None:
        Lx += pad
        Ly += 2*pad
    else:
        pad = 0
    mesh = fire.PeriodicRectangleMesh(Lx, Ly, nx, ny, quadrilateral=quadrilateral)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh

def UnitSquareMesh(nx, ny, quadrilateral=False, comm=None):
    mesh = fire.UnitSquareMesh(nx, ny, quadrilateral=quadrilateral, reorder=reorder, distribution_parameters=distribution_parameters, comm=comm, **kwargs)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    return mesh





