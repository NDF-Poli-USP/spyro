import firedrake as fire
import numpy as np
import FIAT
import finat
from spyro.utils.cost import comp_cost

# Geometry
length_x = 2.0
length_y = 2.0
length_z = 1.5

# Mesh
h = 0.1
nx = int(length_x / h)
nz = int(length_z / h)
ny = int(length_y / h)
mesh = fire.BoxMesh(nx, ny, nz, length_x, length_y, length_z)


# Reference to resource usage
tRef = comp_cost("tini")

V = fire.FunctionSpace(mesh, "KMV", 3)
quad_rule = finat.quadrature.make_quadrature(
    V.finat_element.cell, V.ufl_element().degree(), "KMV")
dx = fire.dx(scheme=quad_rule)

# Initialize velocity field
c = fire.Function(V, name='c [km/s])')
x, y, z = fire.SpatialCoordinate(mesh)
cond = fire.conditional(x < length_x / 2.0, 3.0, 1.5)
c.interpolate(cond, allow_missing_dofs=True)

# Bilinear forms
u, v = fire.TrialFunction(V), fire.TestFunction(V)
a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * dx
m = fire.inner(u, v) * dx


# Solver
n_eig = 2
ksp_type = "cg"
pc_type = "hypre"
opts = {
    "eps_type": "krylovschur",          # Iterative
    "eps_tol": 1e-6,                 # Tight tolerance for accuracy
    "eps_max_it": 200,               # Reasonable iteration cap
    "st_type": "sinvert",            # Useful for interior eigenvalues
    "st_shift": 1e-6,                # Stabilizes Neumann BC null space
    "eps_smallest_magnitude": None,  # Smallest eigenvalues in magnitude
    "eps_monitor": "ascii",          # Print convergence info
    "ksp_type": ksp_type,            # Options for large problems
    "pc_type": pc_type               # Options for large problems
}
eigenproblem = fire.LinearEigenproblem(a, M=m)
eigensolver = fire.LinearEigensolver(eigenproblem, n_evals=n_eig, solver_parameters=opts)
nconv = eigensolver.solve()
lam = eigensolver.eigenvalue(1)
Lsp = np.asarray([eigensolver.eigenvalue(mod) for mod in range(n_eig)])


comp_cost("tfin", tRef=tRef, user_name="mod_cost")
