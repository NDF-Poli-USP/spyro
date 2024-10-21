import numpy as np


def change_to_reference_triangle(p, cell_vertices):
    """
    Changes variables to reference triangle

    Parameters
    ----------
    p : tuple
        Point in original triangle
    cell_vertices : list
        List of vertices, in tuple format, of original triangle

    Returns
    -------
    tuple
        Point location in reference triangle
    """
    (xa, ya) = cell_vertices[0]
    (xb, yb) = cell_vertices[1]
    (xc, yc) = cell_vertices[2]
    (px, py) = p

    xna = 0.0
    yna = 0.0
    xnb = 1.0
    ynb = 0.0
    xnc = 0.0
    ync = 1.0

    div = xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb
    a11 = (
        -(xnb * ya - xnc * ya - xna * yb + xnc * yb + xna * yc - xnb * yc) / div
    )
    a12 = (
        xa * xnb - xa * xnc - xb * xna + xb * xnc + xc * xna - xc * xnb
    ) / div
    a13 = (
        xa * xnc * yb
        - xb * xnc * ya
        - xa * xnb * yc
        + xc * xnb * ya
        + xb * xna * yc
        - xc * xna * yb
    ) / div
    a21 = (
        -(ya * ynb - ya * ync - yb * yna + yb * ync + yc * yna - yc * ynb) / div
    )
    a22 = (
        xa * ynb - xa * ync - xb * yna + xb * ync + xc * yna - xc * ynb
    ) / div
    a23 = (
        xa * yb * ync
        - xb * ya * ync
        - xa * yc * ynb
        + xc * ya * ynb
        + xb * yc * yna
        - xc * yb * yna
    ) / div

    pnx = px * a11 + py * a12 + a13
    pny = px * a21 + py * a22 + a23

    return (pnx, pny)


def change_to_reference_tetrahedron(
    p, cell_vertices, reference_coordinates=None
):
    """
    Changes variables to reference tetrahedron

    Parameters
    ----------
    p : tuple
        Point in original tetrahedron
    cell_vertices : list
        List of vertices, in tuple format, of original tetrahedron
    reference_coordinates : list, optional
        List of reference coordinates, in tuple format, of original tetrahedron

    Returns
    -------
    tuple
        Point location in reference tetrahedron
    """
    (xa, ya, za) = cell_vertices[0]
    (xb, yb, zb) = cell_vertices[1]
    (xc, yc, zc) = cell_vertices[2]
    (xd, yd, zd) = cell_vertices[3]
    (px, py, pz) = p

    if reference_coordinates is None:
        ra = (0.0, 0.0, 0.0)
        rb = (1.0, 0.0, 0.0)
        rc = (0.0, 1.0, 0.0)
        rd = (0.0, 0.0, 1.0)
        reference_coordinates = [
            ra,
            rb,
            rc,
            rd,
        ]

    xna, yna, zna = reference_coordinates[0]
    xnb, ynb, znb = reference_coordinates[1]
    xnc, ync, znc = reference_coordinates[2]
    xnd, ynd, znd = reference_coordinates[3]

    det = (
        xa * yb * zc
        - xa * yc * zb
        - xb * ya * zc
        + xb * yc * za
        + xc * ya * zb
        - xc * yb * za
        - xa * yb * zd
        + xa * yd * zb
        + xb * ya * zd
        - xb * yd * za
        - xd * ya * zb
        + xd * yb * za
        + xa * yc * zd
        - xa * yd * zc
        - xc * ya * zd
        + xc * yd * za
        + xd * ya * zc
        - xd * yc * za
        - xb * yc * zd
        + xb * yd * zc
        + xc * yb * zd
        - xc * yd * zb
        - xd * yb * zc
        + xd * yc * zb
    )
    a11 = (
        (xnc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (xnd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (xnb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (xna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a12 = (
        (xnd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (xnc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (xnb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (xna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a13 = (
        (xnc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (xnd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (xnb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (xna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
    )
    a14 = (
        (
            xnd
            * (
                xa * yb * zc
                - xa * yc * zb
                - xb * ya * zc
                + xb * yc * za
                + xc * ya * zb
                - xc * yb * za
            )
        )
        / det
        - (
            xnc
            * (
                xa * yb * zd
                - xa * yd * zb
                - xb * ya * zd
                + xb * yd * za
                + xd * ya * zb
                - xd * yb * za
            )
        )
        / det
        + (
            xnb
            * (
                xa * yc * zd
                - xa * yd * zc
                - xc * ya * zd
                + xc * yd * za
                + xd * ya * zc
                - xd * yc * za
            )
        )
        / det
        - (
            xna
            * (
                xb * yc * zd
                - xb * yd * zc
                - xc * yb * zd
                + xc * yd * zb
                + xd * yb * zc
                - xd * yc * zb
            )
        )
        / det
    )
    a21 = (
        (ync * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (ynd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (ynb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (yna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a22 = (
        (ynd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (ync * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (ynb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (yna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a23 = (
        (ync * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (ynd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (ynb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (yna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
    )
    a24 = (
        (
            ynd
            * (
                xa * yb * zc
                - xa * yc * zb
                - xb * ya * zc
                + xb * yc * za
                + xc * ya * zb
                - xc * yb * za
            )
        )
        / det
        - (
            ync
            * (
                xa * yb * zd
                - xa * yd * zb
                - xb * ya * zd
                + xb * yd * za
                + xd * ya * zb
                - xd * yb * za
            )
        )
        / det
        + (
            ynb
            * (
                xa * yc * zd
                - xa * yd * zc
                - xc * ya * zd
                + xc * yd * za
                + xd * ya * zc
                - xd * yc * za
            )
        )
        / det
        - (
            yna
            * (
                xb * yc * zd
                - xb * yd * zc
                - xc * yb * zd
                + xc * yd * zb
                + xd * yb * zc
                - xd * yc * zb
            )
        )
        / det
    )
    a31 = (
        (znc * (ya * zb - yb * za - ya * zd + yd * za + yb * zd - yd * zb))
        / det
        - (znd * (ya * zb - yb * za - ya * zc + yc * za + yb * zc - yc * zb))
        / det
        - (znb * (ya * zc - yc * za - ya * zd + yd * za + yc * zd - yd * zc))
        / det
        + (zna * (yb * zc - yc * zb - yb * zd + yd * zb + yc * zd - yd * zc))
        / det
    )
    a32 = (
        (znd * (xa * zb - xb * za - xa * zc + xc * za + xb * zc - xc * zb))
        / det
        - (znc * (xa * zb - xb * za - xa * zd + xd * za + xb * zd - xd * zb))
        / det
        + (znb * (xa * zc - xc * za - xa * zd + xd * za + xc * zd - xd * zc))
        / det
        - (zna * (xb * zc - xc * zb - xb * zd + xd * zb + xc * zd - xd * zc))
        / det
    )
    a33 = (
        (znc * (xa * yb - xb * ya - xa * yd + xd * ya + xb * yd - xd * yb))
        / det
        - (znd * (xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb))
        / det
        - (znb * (xa * yc - xc * ya - xa * yd + xd * ya + xc * yd - xd * yc))
        / det
        + (zna * (xb * yc - xc * yb - xb * yd + xd * yb + xc * yd - xd * yc))
        / det
    )
    a34 = (
        (
            znd
            * (
                xa * yb * zc
                - xa * yc * zb
                - xb * ya * zc
                + xb * yc * za
                + xc * ya * zb
                - xc * yb * za
            )
        )
        / det
        - (
            znc
            * (
                xa * yb * zd
                - xa * yd * zb
                - xb * ya * zd
                + xb * yd * za
                + xd * ya * zb
                - xd * yb * za
            )
        )
        / det
        + (
            znb
            * (
                xa * yc * zd
                - xa * yd * zc
                - xc * ya * zd
                + xc * yd * za
                + xd * ya * zc
                - xd * yc * za
            )
        )
        / det
        - (
            zna
            * (
                xb * yc * zd
                - xb * yd * zc
                - xc * yb * zd
                + xc * yd * zb
                + xd * yb * zc
                - xd * yc * zb
            )
        )
        / det
    )

    pnx = px * a11 + py * a12 + pz * a13 + a14
    pny = px * a21 + py * a22 + pz * a23 + a24
    pnz = px * a31 + py * a32 + pz * a33 + a34

    return (pnx, pny, pnz)


def change_to_reference_quad(p, cell_vertices):
    """
    Changes varibales to reference quadrilateral

    Parameters
    ----------
    p : tuple
        Point in original quadrilateral
    cell_vertices : list
        List of vertices, in tuple format, of original quadrilateral

    Returns
    -------
    tuple
        Point location in reference quadrilateral
    """
    (px, py) = p
    # Irregular quad
    (x0, y0) = cell_vertices[0]
    (x1, y1) = cell_vertices[1]
    (x2, y2) = cell_vertices[2]
    (x3, y3) = cell_vertices[3]

    # Reference quad
    # xn0 = 0.0
    # yn0 = 0.0
    # xn1 = 1.0
    # yn1 = 0.0
    # xn2 = 1.0
    # yn2 = 1.0
    # xn3 = 0.0
    # yn3 = 1.0

    dx1 = x1 - x2
    dx2 = x3 - x2
    dy1 = y1 - y2
    dy2 = y3 - y2
    sumx = x0 - x1 + x2 - x3
    sumy = y0 - y1 + y2 - y3

    gover = np.array([[sumx, dx2], [sumy, dy2]])

    g_under = np.array([[dx1, dx2], [dy1, dy2]])

    gunder = np.linalg.det(g_under)

    hover = np.array([[dx1, sumx], [dy1, sumy]])

    hunder = gunder

    g = np.linalg.det(gover) / gunder
    h = np.linalg.det(hover) / hunder
    i = 1.0

    a = x1 - x0 + g * x1
    b = x3 - x0 + h * x3
    c = x0
    d = y1 - y0 + g * y1
    e = y3 - y0 + h * y3
    f = y0

    A = e * i - f * h
    B = c * h - b * i
    C = b * f - c * e
    D = f * g - d * i
    E = a * i - c * g
    F = c * d - a * f
    G = d * h - e * g
    H = b * g - a * h
    Ij = a * e - b * d

    pnx = (A * px + B * py + C) / (G * px + H * py + Ij)
    pny = (D * px + E * py + F) / (G * px + H * py + Ij)

    return (pnx, pny)


def change_to_reference_hexa(p, cell_vertices, based_on_extruded=True):
    """
    Changes variables to reference hexahedron.
    Parameters
    ----------
    p : tuple
        Point in original hexahedron
    cell_vertices : list
        List of vertices, in tuple format, of original hexahedron

    Returns
    -------
    tuple
        Point location in reference hexahedron
    """

    if based_on_extruded:
        a = cell_vertices[0]
        b = cell_vertices[1]
        c = cell_vertices[2]
        d = cell_vertices[4]

        ra = (0.0, 0.0, 0.0)
        rb = (0.0, 0.0, 1.0)
        rc = (0.0, 1.0, 0.0)
        rd = (1.0, 0.0, 0.0)

        reference_coordinates = [ra, rb, rc, rd]
        tet_cell_vertices = [a, b, c, d]

        return change_to_reference_tetrahedron(
            p, tet_cell_vertices, reference_coordinates=reference_coordinates
        )
    else:
        raise ValueError("Not yet implemented.")
