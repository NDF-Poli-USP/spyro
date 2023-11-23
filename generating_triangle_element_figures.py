import firedrake as fire


def create_element_points_2d(degree, element_type="equispaced"):
    """Creates 2D single cell element point locations with a given degree and element type.

    Parameters
    ----------
    degree : int
        The degree of the element.
    element_type : str
        The type of element to create. Options are "equispaced" and "mass_lumped_triangle".

    Returns
    -------
    points : list
        A list of points that define the element.
    """
    mesh = fire.UnitTriangleMesh()
    x_mesh, y_mesh = fire.SpatialCoordinate(mesh)

    if element_type == "equispaced":
        V = fire.FunctionSpace(mesh, "CG", degree)
    elif element_type == "mass_lumped_triangle":
        V = fire.FunctionSpace(mesh, "KMV", degree)

    ux = fire.Function(V)
    uy = fire.Function(V)
    ux.interpolate(x_mesh)
    uy.interpolate(y_mesh)
    xs = ux.dat.data[:]
    ys = uy.dat.data[:]
    points = list(zip(xs, ys))
    return points


def create_element_points_3d(degree, element_type="equispaced"):
    """Creates 2D single cell element point locations with a given degree and element type.

    Parameters
    ----------
    degree : int
        The degree of the element.
    element_type : str
        The type of element to create. Options are "equispaced" and "mass_lumped_triangle".

    Returns
    -------
    points : list
        A list of points that define the element.
    """
    mesh = fire.UnitTetrahedronMesh()
    x_mesh, y_mesh, z_mesh = fire.SpatialCoordinate(mesh)

    if element_type == "equispaced":
        V = fire.FunctionSpace(mesh, "CG", degree)
    elif element_type == "mass_lumped_triangle":
        V = fire.FunctionSpace(mesh, "KMV", degree)

    ux = fire.Function(V)
    uy = fire.Function(V)
    uz = fire.Function(V)
    ux.interpolate(x_mesh)
    uy.interpolate(y_mesh)
    uz.interpolate(z_mesh)
    xs = ux.dat.data[:]
    ys = uy.dat.data[:]
    zs = uz.dat.data[:]
    points = list(zip(xs, ys, zs))
    return points


def convert_to_tikz_2d(points):
    """Converts a list of points to tikz commands.

    Parameters
    ----------
    points : list
        A list of points that define the element.

    Returns
    -------
    tikz_commands : str
        A string of tikz commands that define the element.
    """
    tikz_commands = []
    tikz_commands.append("\\begin{tikzpicture}[scale=1.0,>=latex]")
    tikz_commands.append("\t\\draw[](0.0, 0.0) -- (1.0, 0.0) -- (0.0, 1.0) -- cycle;")
    for point in points:
        tikz_commands.append(f"\t\\draw[fill=black] ({point[0]},{point[1]}) circle (0.02);")
    tikz_commands.append("\\end{tikzpicture} \\\\ \\small")
    return "\n".join(tikz_commands)


def basic_tetrahedron_tikz():
    tikz_commands = []
    tikz_commands.append("\t\\coordinate (a) at (0.0, 0.0, 0.0);")
    tikz_commands.append("\t\\coordinate (b) at (1.0, 0.0, 0.0);")
    tikz_commands.append("\t\\coordinate (c) at (0.0, 1.0, 0.0);")
    tikz_commands.append("\t\\coordinate (d) at (0.0, 0.0, 1.0);")
    tikz_commands.append("\t\\draw[] (b) -- (c) -- (d) -- cycle;")
    tikz_commands.append("\t\\draw[dashed] (d) -- (a) -- (c);")
    tikz_commands.append("\t\\draw[dashed] (a) -- (b);")
    return "\n".join(tikz_commands)


def convert_to_tikz_3d(points):
    """Converts a list of points to tikz commands.

    Parameters
    ----------
    points : list
        A list of points that define the element.

    Returns
    -------
    tikz_commands : str
        A string of tikz commands that define the element.
    """
    tikz_commands = []
    tikz_commands.append(basic_tetrahedron_tikz())
    for point in points:
        tikz_commands.append(f"\t\\draw[fill=black, opacity={opacity}] ({point[0]}, {point[1]}, {point[2]}) circle (0.02);")
    tikz_commands.append("\\end{tikzpicture} \\\\ \\small")
    return "\n".join(tikz_commands)


if __name__ == "__main__":
    degree = 5
    points = create_element_points_2d(degree, element_type="equispaced")
    tikz_commands = convert_to_tikz_2d(points)
    print(tikz_commands)
    print("END")
