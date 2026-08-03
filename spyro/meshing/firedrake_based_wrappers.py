import firedrake as fire


def rectangle_mesh(nx, ny, length_x, length_y, pad=None, comm=None, quadrilateral=False):
    """Create a rectangle mesh based on the Firedrake mesh.

    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

    Parameters
    ----------
    length_x : float
      Length of the domain in the x direction.
    length_y : float
      Length of the domain in the y direction.
    nx : int
      Number of elements in the x direction.
    ny : int
      Number of elements in the y direction.
    pad : float, optional
      Padding to be added to the domain. The default is None.
    comm : MPI communicator, optional
      MPI communicator. The default is None.
    quadrilateral : bool, optional
      If True, the mesh is quadrilateral. The default is False.

    Returns
    -------
    mesh : Firedrake Mesh
      Mesh
    """
    if pad is not None:
        length_x += pad
        length_y += 2 * pad
    else:
        pad = 0

    if comm is None:
        mesh = fire.RectangleMesh(nx, ny, length_x, length_y,
                                  quadrilateral=quadrilateral)
    else:
        mesh = fire.RectangleMesh(nx, ny, length_x, length_y,
                                  quadrilateral=quadrilateral, comm=comm)

    # Adjusting to Spyro's reference system (z, x) with origin at (0, 0)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def periodic_rectangle_mesh(
    nx, ny, length_x, length_y, pad=None, comm=None, quadrilateral=False
):
    """Create a periodic rectangle mesh based on the Firedrake mesh.

    First axis is negative, second axis is positive. If there is a pad, both
    axis are dislocated by the pad.

    Parameters
    ----------
    length_x : float
        Length of the domain in the x direction.
    length_y : float
        Length of the domain in the y direction.
    nx : int
        Number of elements in the x direction.
    ny : int
        Number of elements in the y direction.
    pad : float, optional
        Padding to be added to the domain. The default is None.
    comm : MPI communicator, optional
        MPI communicator. The default is None.
    quadrilateral: bool, optional
        If True, the mesh is quadrilateral. The default is False.

    Returns
    -------
    mesh : Firedrake Mesh
        Mesh
    """
    if pad is not None:
        length_x += pad
        length_y += 2 * pad
    else:
        pad = 0

    if comm is None:
        mesh = fire.PeriodicRectangleMesh(nx, ny, length_x, length_y,
                                          quadrilateral=quadrilateral)
    else:
        mesh = fire.PeriodicRectangleMesh(nx, ny, length_x, length_y,
                                          quadrilateral=quadrilateral, comm=comm)

    # Adjusting to Spyro's reference system (z, x) with origin at (0, 0)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    mesh.coordinates.dat.data[:, 1] -= pad

    return mesh


def box_mesh(nx, ny, nz, length_x, length_y, length_z, pad=None,
             quadrilateral=False, comm=None):
    """
    Create a 3D box mesh based on Firedrake mesh utilities.

    Parameters
    ----------
    nx : int
        Number of elements in the x direction.
    ny : int
        Number of elements in the y direction.
    nz : int
        Number of elements in the z direction.
    length_x : float
        Length of the domain in the x direction.
    length_y : float
        Length of the domain in the y direction.
    length_z : float
        Length of the domain in the z direction.
    pad : float, optional
        Padding to be added to the domain. The default is None.
    quadrilateral : bool, optional
        If True, the mesh is created by extruding a quadrilateral mesh.
        The default is False.
    comm : MPI communicator, optional
      MPI communicator. The default is None.

    Returns
    -------
    mesh : Firedrake Mesh
        The generated 3D box mesh.

    Notes
    -----
    The first coordinate is negated(multiplied by - 1) to match the expected
    coordinate system. If quadrilateral is True, the mesh is created by
    extruding a 2D quadrilateral mesh in the z direction.
    """
    if pad is not None:
        length_x += pad
        length_y += 2 * pad
        length_z += 2 * pad
    else:
        pad = 0

    if quadrilateral:

        if comm is None:
            quad_mesh = fire.RectangleMesh(nx, ny, length_x, length_y,
                                           quadrilateral=quadrilateral)

        else:
            quad_mesh = fire.RectangleMesh(nx, ny, length_x, length_y,
                                           quadrilateral=quadrilateral, comm=comm)

        # Adjusting to Spyro's reference system (z, x, y) with origin at (0, 0, 0)
        quad_mesh.coordinates.dat.data[:, 0] *= -1.0
        quad_mesh.coordinates.dat.data[:, 1] -= pad
        layer_height = length_z / nz
        mesh = fire.ExtrudedMesh(quad_mesh, nz, layer_height=layer_height)
    else:

        if comm is None:
            mesh = fire.BoxMesh(nx, ny, nz, length_x, length_y, length_z)

        else:
            mesh = fire.BoxMesh(nx, ny, nz, length_x, length_y, length_z, comm=comm)

        # Adjusting to Spyro's reference system (z, x, y) with origin at (0, 0, 0)
        mesh.coordinates.dat.data[:, 0] *= -1.0
        mesh.coordinates.dat.data[:, 1] -= pad

    # Offset to respect the origin of the domain
    mesh.coordinates.dat.data_with_halos[:, 2] -= pad

    return mesh
