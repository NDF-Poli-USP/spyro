from firedrake import *

mesh = UnitSquareMesh(10, 10) 

# random perturbation
X = SpatialCoordinate(mesh)
W = VectorFunctionSpace(mesh, "CG", 1)
w = Function(W)
w.interpolate(as_vector([sin(X[0]+X[1]), cos(X[0]*X[1])]))

# state problem
V = FunctionSpace(mesh, "CG", 1)
u, v = Function(V), TestFunction(V)
f = sin(X[1]) * cos(X[0])

for jj in range(0, 1):
    # two possible Dirichlet data
    g = (1 - jj)*Constant(1.) + jj*f/2.
    # print(g)
    for ii in range(2):
        
        # two ways to impose DirichletBC
        bc = DirichletBC(V, (1-ii)*g, "on_boundary")
        
        F = (inner(grad(u+ii*g), grad(v)) + (u+ii*g-f)*v)*dx
        J = (u+ii*g)*dx
        print(bc.__dict__)
        # evaluate dJdw
        solve(F == 0, u, bcs=bc)
        bil_form = adjoint(derivative(F, u)) 
        rhs = -derivative(J, u)
        u_adj = Function(V)
        solve(assemble(bil_form, bcs=DirichletBC(V, 0, "on_boundary")),
        u_adj, assemble(rhs))
        L = J + replace(F, {v: u_adj})
        dJdw = assemble(derivative(L, X, w)) 
        output = (jj, ii, assemble(J), dJdw)
        print("bcs %u, method %u, J = %1.4f, dJdW = %1.4f" % output)
        