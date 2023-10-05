from firedrake import *
import FIAT
import finat
import numpy as np


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


element_type = 'KMV'
dimension = 2
degrees = [1, 2, 3, 4, 5]

if element_type == 'spectral':
    quadrilateral = True
    method = 'CG'
    variant = 'spectral'
elif element_type == 'KMV':
    quadrilateral = False
    method = 'KMV'
    variant = 'KMV'

if dimension == 2:
    n = 1000
    mesh = UnitSquareMesh(n, n, quadrilateral=quadrilateral)
elif dimension == 3:
    n = 200
    mesh = UnitCubeMesh(n, n, n)

for degree in degrees:

    print('For a '+method+' '+str(dimension)+'D'+' element with degree = ' + str(degree))
    element = FiniteElement(method, mesh.ufl_cell(), degree=degree, variant=variant)
    V = FunctionSpace(mesh, element)
    u = Function(V)
    udat = u.dat.data[:]
    dof = len(udat)

    if dimension == 2 and quadrilateral is False:
        num_elements = (2*n**2)
        new_alpha = np.sqrt(dof / (num_elements))
    elif dimension == 2 and quadrilateral is True:
        num_elements = (n**2)
        new_alpha = np.sqrt(dof / (num_elements))
    elif dimension == 3 and quadrilateral is False:
        num_elements = (6*n**3)
        new_alpha = np.cbrt(dof / (num_elements))
    elif dimension == 3 and quadrilateral is True:
        num_elements = (n**3)
        new_alpha = np.cbrt(dof / (num_elements))

    print(new_alpha)
