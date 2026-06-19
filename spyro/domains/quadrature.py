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
            and family <= {"Q", "DQ", "TensorProductElement"}):
        # In this case, for the spectral element method we use GLL quadrature
        qr_x = {"scheme": "lump", "degree": degree}
        qr_k = qr_x
        qr_s = qr_x
    else:
        raise ValueError("Unrecognized quadrature scheme")
    return qr_x, qr_k, qr_s
