import finat
from firedrake import *
import numpy as np


def analytical_solution(t, V, mesh_z, mesh_x):
    analytical = Function(V)
    x = mesh_z
    y = mesh_x
    analytical.interpolate(x * (x + 1) * y * (y - 1) * t)

    return analytical


def isDiag(M):
    i, j = np.nonzero(M)
    return np.all(i == j)

degree = 4
mesh = RectangleMesh(50, 50, 1.0, 1.0)
mesh.coordinates.dat.data[:, 0] *= -1.0
mesh_z, mesh_x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "KMV", degree)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

u = TrialFunction(V)
v = TestFunction(V)

c = Function(V, name="velocity")
c.interpolate(1 + sin(pi*-mesh_z)*sin(pi*mesh_x))
u_n = Function(V)
u_nm1 = Function(V)
dt = 0.0005
t = 0.0
final_time = 1.0
u_nm1.assign(analytical_solution((t - 2 * dt), V, mesh_z, mesh_x))
u_n.assign(analytical_solution((t - dt), V, mesh_z, mesh_x))

q_xy = Function(V)
q_xy.interpolate(-(mesh_z**2) - mesh_z - mesh_x**2 + mesh_x)
q = q_xy * Constant(2 * t)

nt = int((final_time - t) / dt) + 1

m1 = (
    1
    / (c * c)
    * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
    * v
    * dx(scheme=quad_rule)
)
a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
le = q * v * dx(scheme=quad_rule)

form = m1 + a - le

B = Cofunction(V.dual())

boundary_ids = (1, 2, 3, 4)
bcs = DirichletBC(V, 0.0, boundary_ids)
A = assemble(lhs(form), bcs=bcs)
solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}
solver = LinearSolver(
    A, solver_parameters=solver_parameters
)
As = solver.A
petsc_matrix = As.petscmat
diagonal = petsc_matrix.getDiagonal()
Mdiag = diagonal.array

np.save("/home/olender/Development/issue_50_compatibility/spyro-1/new_diag", Mdiag)
out = File("new_firedrake_u2.pvd")

u_np1 = Function(V)
for step in range(nt):
    q = q_xy * Constant(2 * t)
    m1 = (
        1
        / (c * c)
        * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
        * v
        * dx(scheme=quad_rule)
    )
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    le = q * v * dx(scheme=quad_rule)

    form = m1 + a - le

    B = assemble(rhs(form), tensor=B)

    solver.solve(u_np1, B)

    if (step - 1) % 100 == 0:
        print(f"Time : {t}")
        out.write(u_n)
        assert (
            norm(u_n) < 1
        ), "Numerical instability. Try reducing dt or building the \
            mesh differently"

    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    t = step * float(dt)

u_an = analytical_solution(t, V, mesh_z, mesh_x)
error = errornorm(u_n, u_an)
print(f"Error: {error}")

print("END")