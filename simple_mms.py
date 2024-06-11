import firedrake as fire
from firedrake import dot, grad, sin, pi, dx
import FIAT
import finat
import math


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
    analytical = fire.Function(V)
    x = mesh_z
    y = mesh_x
    analytical.interpolate(x * (x + 1) * y * (y - 1) * t)

    return analytical


quad_rule = gauss_lobatto_legendre_cube_rule(dimension=2, degree=4)

length_z = 1.0
length_x = 1.0

mesh_dx = 0.02
nz = int(length_z / mesh_dx)
nx = int(length_x / mesh_dx)
mesh = fire.RectangleMesh(
    nz,
    nx,
    length_z,
    length_x,
    quadrilateral=True
)
mesh_z, mesh_x = fire.SpatialCoordinate(mesh)

output = fire.File("debug.pvd")

element = fire.FiniteElement(  # noqa: F405
    "CG", mesh.ufl_cell(), degree=4, variant="spectral"
)
V = fire.FunctionSpace(mesh, element)

X = fire.Function(V)
u_nm1 = fire.Function(V)
u_n = fire.Function(V)

# Setting C
c = fire.Function(V, name="velocity")
c.interpolate(1 + sin(pi*-mesh_z)*sin(pi*mesh_x))

# setting xy part of source
q_xy = fire.Function(V)
xy = fire.project((-(mesh_z**2) - mesh_z - mesh_x**2 + mesh_x), V)
q_xy.assign(xy)

# Starting time integration
t = 0.0
final_time = 1.0
dt = 0.0001
nt = int((final_time - t) / dt) + 1
u_nm1.assign(analytical_solution((t - 2 * dt), V, mesh_z, mesh_x))
u_n.assign(analytical_solution((t - dt), V, mesh_z, mesh_x))
u_np1 = fire.Function(V, name="pressure t +dt")
u = fire.TrialFunction(V)
v = fire.TestFunction(V)

m1 = (
    (1 / (c * c))
    * ((u - 2.0 * u_n + u_nm1) / fire.Constant(dt**2))
    * v
    * dx(scheme=quad_rule)
)
a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)  # explicit

B = fire.Function(V)

form = m1 + a
lhs = fire.lhs(form)
rhs = fire.rhs(form)

A = fire.assemble(lhs, mat_type="matfree")
solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}
solver = fire.LinearSolver(
    A, solver_parameters=solver_parameters
)

for step in range(nt):
    q = q_xy * fire.Constant(2 * t)
    m1 = (
        1
        / (c * c)
        * ((u - 2.0 * u_n + u_nm1) / fire.Constant(dt**2))
        * v
        * dx(scheme=quad_rule)
    )
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    le = q * v * dx(scheme=quad_rule)

    form = m1 + a - le
    rhs = fire.rhs(form)

    B = fire.assemble(rhs, tensor=B)

    solver.solve(X, B)

    u_np1.assign(X)

    if (step - 1) % 100 == 0:
        assert (
            fire.norm(u_n) < 1
        ), "Numerical instability. Try reducing dt or building the \
            mesh differently"
        output.write(u_n, time=t, name="Pressure")
        if t > 0:
            print(f"Simulation time is: {t:{10}.{4}} seconds", flush=True)

    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    t = step * float(dt)

last_analytical_sol = analytical_solution(t, V, mesh_z, mesh_x)
error = fire.errornorm(u_n, last_analytical_sol)
test = math.isclose(error, 0.0, abs_tol=1e-7)

print(F"Test is {test}")
