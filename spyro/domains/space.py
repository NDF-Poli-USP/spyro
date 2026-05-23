from firedrake import (FiniteElement, FunctionSpace, VectorElement)


def create_function_space(mesh, method, degree, dim=1):
    """Create a Firedrake function space based on the specified
    finite element method.

    Parameters:
    -----------
    mesh: Firedrake Mesh
        Mesh to be used in the finite element space.
    method: str
        Method to be used for the finite element space.
    degree: int
        Degree of the finite element space.
    dim: int
        Number of degrees of freedom per node.

    Returns:
    --------
    function_space: Firedrake FunctionSpace
        Function space.
    """

    if method == "mass_lumped_triangle":
        element = FiniteElement(
            "KMV", mesh.ufl_cell(), degree=degree,
        )
    elif method == "spectral_quadrilateral":
        element = FiniteElement(
            "CG", mesh.ufl_cell(), degree=degree, variant="spectral"
        )
    elif method in ["DG_triangle", "DG_quadrilateral", "DG"]:
        element = FiniteElement(
            "DG", mesh.ufl_cell(), degree=degree
        )
    elif method in ["CG_triangle", "CG_quadrilateral", "CG"]:
        element = FiniteElement(
            "CG", mesh.ufl_cell(), degree=degree
        )
    elif method in ["DQ_quadrilateral", "DQ"]:
        element = FiniteElement(
            "DQ", mesh.ufl_cell(), degree=degree, variant="spectral"
        )

    if dim > 1:
        element = VectorElement(element, dim=dim)

    return FunctionSpace(mesh, element)


def check_function_space_type(function_space):
    # Check if wave.function_space is a generates vector os scalar fields:
    if function_space.value_size == 1:
        return "scalar"
    elif function_space.value_size > 1:
        if len(function_space.topological.subspaces) == 1:
            return "vector"
        elif len(function_space.topological.subspaces) > 1:
            return "mixed"
        else:
            raise ValueError(
                f"Function space topology of {function_space.topological} not supported",
            )
    else:
        raise ValueError(f"Function space size of {function_space.value_size} not supported")
