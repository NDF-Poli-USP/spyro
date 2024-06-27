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



def isDiag(M):
    i, j = np.nonzero(M)
    return np.all(i == j)

degree = 2
# mesh = RectangleMesh(3, 2, 2.0, 1.0, quadrilateral=True)
mesh = RectangleMesh(1, 1, 1.0, 1.0, quadrilateral=True)
element = FiniteElement('CG', mesh.ufl_cell(), degree=degree, variant='spectral')
V = FunctionSpace(mesh, element)
quad_rule = gauss_lobatto_legendre_cube_rule(dimension=2, degree=degree)

u = TrialFunction(V)
v = TestFunction(V)

form = u*v*dx(scheme=quad_rule)
A = assemble(form)
M = A.M.values
Mdiag = M.diagonal()

x_mesh, y_mesh = SpatialCoordinate(mesh)
x_func = Function(V)
y_func = Function(V)
x_func.interpolate(x_mesh)
y_func.interpolate(y_mesh)
x = x_func.dat.data[:]
y = y_func.dat.data[:]

print(f"Matrix is diagonal:{isDiag(M)}")
old_diag = np.load("/home/olender/Development/issue_50_compatibility/spyro-1/old_sem_diag.npy")
dif = Mdiag-old_diag
print("END")