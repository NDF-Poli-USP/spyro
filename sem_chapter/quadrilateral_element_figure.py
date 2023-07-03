from firedrake import *
import FIAT
import finat
import matplotlib.pyplot as plt

def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())

def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

def quadrature_rules(V):
    """ Quadrature rule - Gauss-Lobatto-Legendre, Gauss-Legendre and Equi-spaced, KMV"""
    degree = V.ufl_element().degree()
    dimension = V.mesh().geometric_dimension()
    cell_geometry = V.mesh().ufl_cell()

    # Getting method, this returns the names used in Firedrake and UFL
    # current implementation supports 'Lagrange' ('CG'),
    # 'Kong-Mulder-Veldhuizen' ('KMV'), 'Discontinuous Lagrange' ('DG'),
    # 'DQ' ('DG' with quads).

    ufl_method = V.ufl_element().family()

    # Dealing with mixed function spaces
    if ufl_method == "Mixed":
        ufl_method = V.sub(1).ufl_element().family()

    if (cell_geometry == quadrilateral) and ufl_method == "Q":
        # In this case, for the spectral element method we use GLL quadrature
        qr_x = gauss_lobatto_legendre_cube_rule(dimension=dimension, degree=degree)
        qr_k = qr_x
        qr_s = gauss_lobatto_legendre_cube_rule(
            dimension=(dimension - 1), degree=degree
        )
    # elif (cell_geometry == quadrilateral) and ufl_method == "DQ":
    #     # In this case, we use GL quadrature
    #     qr_x = gauss_legendre_cube_rule(dimension=dimension, degree=degree)
    #     qr_k = qr_x
    #     qr_s = gauss_legendre_cube_rule(dimension=(dimension - 1), degree=degree)
    elif (cell_geometry == triangle) and (
        ufl_method == "Lagrange" or ufl_method == "Discontinuous Lagrange"
    ):
        qr_x = None
        qr_s = None
        qr_k = None
    elif (cell_geometry == tetrahedron) and (
        ufl_method == "Lagrange" or ufl_method == "Discontinuous Lagrange"
    ):
        qr_x = None
        qr_s = None
        qr_k = None
    elif ufl_method == "Kong-Mulder-Veldhuizen":
        qr_x = finat.quadrature.make_quadrature(
            V.finat_element.cell, V.ufl_element().degree(), "KMV"
        )
        qr_s = None
        qr_k = None
    else:
        raise ValueError("Unrecognized quadrature scheme")
    return qr_x, qr_s, qr_k

mesh = UnitSquareMesh(1,1, quadrilateral=True)
x_mesh, y_mesh = SpatialCoordinate(mesh)
element = FiniteElement('CG', mesh.ufl_cell(), degree=4, variant='spectral')
V = FunctionSpace(mesh, element)
qr_x, qr_s, qr_k = quadrature_rules(V)

x_func = Function(V)
y_func = Function(V)

x_func.interpolate(x_mesh)
y_func.interpolate(y_mesh)

x = x_func.dat.data_ro_with_halos[:]
y = y_func.dat.data_ro_with_halos[:]

plt.plot(x, y, 'o', 'black')

a = x[3]-x[2]
print(a*0.02)
plt.plot(a,a,'o', 'red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Quadrilateral element')
plt.axis('equal')
plt.axis('off')
plt.show()

print("END")

