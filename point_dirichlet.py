import firedrake as fire
from firedrake import dx
import numpy as np
# import ipdb

class MyBC(fire.DirichletBC):
    def __init__(self, V, value, nodes):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(MyBC, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = nodes


mesh = fire.UnitSquareMesh(20, 20)
mesh_x, mesh_y = fire.SpatialCoordinate(mesh)
V = fire.FunctionSpace(mesh, "CG", 1)
u = fire.TrialFunction(V)
vy = fire.TestFunction(V)
yp = fire.Function(V)

f = fire.Constant(1.0)
cond = fire.conditional(mesh_x < 0.5, 3.0, 1.5)
c = fire.Function(V).interpolate(cond)

F1 = fire.inner(fire.grad(u), fire.grad(vy)) * dx - f / c * vy * dx

point = (0.25, 0.5)

x_f = fire.Function(V).interpolate(mesh_x)
y_f = fire.Function(V).interpolate(mesh_y)
x_data = x_f.dat.data[:]
y_data = y_f.dat.data[:]
boolean_vector = np.isclose(x_data, 0.25) & np.isclose(y_data, 0.5)
true_indices = np.where(boolean_vector)[0]

bcs_eik = [MyBC(V, 0.0, true_indices)]
fire.solve(fire.lhs(F1) == fire.rhs(F1), yp, bcs=bcs_eik)
out_vel = fire.VTKFile("velocity.pvd")
out_eik = fire.VTKFile("eik.pvd")
out_vel.write(c)
out_eik.write(yp)

print("TEST")


