import firedrake as fire
import numpy as np

def apply_box(mesh, c, x1, y1, x2, y2, value):
    x, y = fire.SpatialCoordinate(mesh)
    box = fire.conditional(x>x1  )