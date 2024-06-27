import finat
import FIAT
from firedrake import *
import numpy as np


def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    points = finat_ps(fiat_rule.get_points())
    weights = fiat_rule.get_weights()
    return finat.quadrature.QuadratureRule(points, weights)

def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result



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
mesh = RectangleMesh(50, 50, 1.0, 1.0, quadrilateral=True)
mesh.coordinates.dat.data[:, 0] *= -1.0
mesh_z, mesh_x = SpatialCoordinate(mesh)
element = FiniteElement('CG', mesh.ufl_cell(), degree=degree, variant='spectral')
V = FunctionSpace(mesh, element)
quad_rule = gauss_lobatto_legendre_cube_rule(dimension=2, degree=degree)

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

# old_Mdiag = np.load("/home/olender/Development/issue_50_compatibility/spyro-1/old_diag.npy")
# old_qxy = np.load("/home/olender/Development/issue_50_compatibility/spyro-1/old_qxy.npy") # OK
# old_un = np.load("/home/olender/Development/issue_50_compatibility/spyro-1/old_un.npy") # OK
# old_unm1 = np.load("/home/olender/Development/issue_50_compatibility/spyro-1/old_unm1.npy") # OK
# old_c = np.load("/home/olender/Development/issue_50_compatibility/spyro-1/old_c.npy") # OK

# out = File("new_firedrake_u2.pvd")

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
        # out.write(u_n)
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