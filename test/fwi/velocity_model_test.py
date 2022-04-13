from firedrake import *
import spyro

mesh = UnitSquareMesh(10,10)

x, y = SpatialCoordinate(mesh)

c = conditional(y<0.5, 3.0, 1.5)
V = FunctionSpace(mesh, 'DG', 0)
U = Function(V).interpolate(c)

spyro.io.create_segy(U,"teste1")