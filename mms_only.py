from firedrake import *
import FIAT, finat

def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())

def gll_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

# Output files
output_file = File("just_mms_output.pvd")

# Input variables
dx = 0.02
dt = 0.001
final_time = 2.0
order = 4

# Create mesh and define function space
nx = int(1.0/dx)
ny = nx
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)
x, y = SpatialCoordinate(mesh)

element = FiniteElement("CG", mesh.ufl_cell(), order, variant="spectral")
V = FunctionSpace(mesh, element)

# Define source and boundary conditions
q_xy = Function(V)
q_time = Constant()


