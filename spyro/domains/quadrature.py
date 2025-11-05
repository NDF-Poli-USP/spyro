import FIAT
import finat
from firedrake import *  # noqa:F403


def quadrature_rules(V):
    """Quadrature rule - Gauss-Lobatto-Legendre, Gauss-Legendre and Equi
    spaced, KMV

    Parameters:
    -----------
    V: Firedrake FunctionSpace
        Function space to be used in the quadrature rule.

    Returns:
    --------
    qr_x: FIAT quadrature rule
        Quadrature rule for the spatial domain.
    qr_s: FIAT quadrature rule
        Quadrature rule for the boundary of the spatial domain.
    qr_k: FIAT quadrature rule
        Quadrature rule for the spatial domain stiffness matrix.
    """
    cell_geometry = V.mesh().ufl_cell()

    # Getting method, this returns the names used in Firedrake and UFL
    # current implementation supports 'Lagrange' ('CG'),
    # 'Kong-Mulder-Veldhuizen' ('KMV'), 'Discontinuous Lagrange' ('DG'),
    # 'DQ' ('DG' with quads).

    # Dealing with mixed function spaces
    family = set(V_.ufl_element().family() for V_ in V)
    degree = max(V_.ufl_element().degree() for V_ in V)
    try:
        degree = max(degree)
    except TypeError:
        pass

    if (cell_geometry in {triangle, tetrahedron}
            and family <= {"Lagrange", "Discontinuous Lagrange"}):
        qr_x = {}
        qr_s = {}
        qr_k = {}
    elif family == {"Kong-Mulder-Veldhuizen"}:
        qr_x = {"scheme": "KMV", "degree": degree}
        qr_s = {}
        qr_k = {}
    elif (cell_geometry in {quadrilateral, hexahedron, TensorProductCell(quadrilateral, interval)}
            and family <= {"Q", "DQ"}):
        dimension = cell_geometry._tdim
        # In this case, for the spectral element method we use GLL quadrature
        qr_x_rule = gauss_lobatto_legendre_cube_rule(
            dimension=dimension, degree=degree
        )
        qr_s_rule = gauss_lobatto_legendre_cube_rule(
            dimension=(dimension - 1), degree=degree
        )
        # Convert to dictionary format for consistent interface
        qr_x = {"scheme": qr_x_rule}
        qr_k = {"scheme": qr_x_rule}
        qr_s = {"scheme": qr_s_rule}
    else:
        raise ValueError("Unrecognized quadrature scheme")
    return qr_x, qr_k, qr_s


# -------------------------- #
# Spectral method - Gauss-Lobatto-Legendre rule
# 1D
def gauss_lobatto_legendre_line_rule(degree):
    """Returns GLL quad rule for a given degree in a line

    Parameters
    ----------
    degree : int
        degree of the polynomial

    Returns
    -------
    result : obj
        quadrature rule
    """
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())


# 3D
def gauss_lobatto_legendre_cube_rule(dimension, degree):
    """Returns GLL quad rule for a given degree in a multidimensional space

    Parameters
    ----------
    dimension : int
        dimension of the space
    degree : int
        degree of the polynomial

    Returns
    -------
    result : obj
        quadrature rule
    """
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result
