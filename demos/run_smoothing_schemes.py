from firedrake import *
import spyro
import sys


mesh = UnitSquareMesh(20, 20, diagonal="crossed")
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)
u_n = Function(V)   # timestep n

x,y = SpatialCoordinate(mesh)

#u_n.interpolate(sin(3.14*(x-0.5)))
u_n.interpolate(1.5 / (2.5 + tanh(200 * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2)))))

lamb =  0.01
mu   = -0.0000105 
k = Constant(0.001)

m = inner((u - u_n), v) * dx 
#a = k * inner(grad(u_n), grad(v)) * dx # explicit
a = k * inner(grad(u), grad(v)) * dx # implicit
F = m + a

lhs_ = lhs(F)
rhs_ = rhs(F)

X = Function(V)
B = Function(V)
A = assemble(lhs_)

params = {"ksp_type": "preonly", "pc_type": "lu"}
solver = LinearSolver(A, solver_parameters=params)

outfile = File("u.pvd")

nt = 1
k.assign(lamb)
for step in range(nt):
    B = assemble(rhs_, tensor=B)
    solver.solve(X, B)
    u_n.assign(X)
    outfile.write(u_n,time=step)
    #if step%2 == 0:
    #    k.assign(mu)
    #else:
    #    k.assign(lamb)

