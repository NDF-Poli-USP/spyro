from firedrake import *

mesh = RectangleMesh(10, 10, 1, 1, diagonal="crossed")

V = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
v = TestFunction(V)

fext = Constant(1)

a = inner(grad(u), grad(v)) * dx
f = fext * v * dx 

bcs = DirichletBC(V, Constant((0.0)), (1,2,3,4))

s = Function(V, name="sol")
par = {"ksp_type": "preonly", "pc_type": "lu"}
solve(a == f, s, bcs=bcs, solver_parameters=par)

G = VectorFunctionSpace(mesh,"Lagrange", 1)
grad_s = Function(G, name="Grad s").interpolate(grad(s))

File("poisson_sol.pvd").write(s)
File("poisson_vec_sol.pvd").write(grad_s)

alpha = - 10
deform = Function(G)
deform.dat.data[:,0] = alpha * grad_s.dat.data[:,0] * s.dat.data[:]
deform.dat.data[:,1] = alpha * grad_s.dat.data[:,1] * s.dat.data[:]

mesh.coordinates.assign(mesh.coordinates + deform)

s_new = Function(V, name="sol new")
solve(a == f, s_new, bcs=bcs, solver_parameters=par)
File("poisson_sol_new.pvd").write(s_new)
