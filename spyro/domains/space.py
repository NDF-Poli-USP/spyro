"""Function-space construction and classification helpers."""

from firedrake import FiniteElement, FunctionSpace, VectorElement


def create_function_space(mesh, method, degree, dim=1):
    """Create a Firedrake function space.

    The finite-element method selects the underlying element family.

    Parameters
    ----------
    mesh : Firedrake Mesh
        Mesh to be used in the finite element space.
    method : str
        Method to be used for the finite element space.
    degree : int
        Degree of the finite element space.
    dim : int
        Number of degrees of freedom per node.

    Returns
    -------
    function_space : Firedrake FunctionSpace
        Function space.
    """
    if method == "mass_lumped_triangle":
        element = FiniteElement(
            "KMV",
            mesh.ufl_cell(),
            degree=degree,
        )
    elif method == "spectral_quadrilateral":
        element = FiniteElement(
            "CG", mesh.ufl_cell(), degree=degree, variant="spectral"
        )
    elif method == "DG_triangle" or "DG_quadrilateral" or "DG":
        element = FiniteElement("DG", mesh.ufl_cell(), degree=degree)
    elif method == "CG_triangle" or "CG_quadrilateral" or "CG":
        element = FiniteElement("CG", mesh.ufl_cell(), degree=degree)

    if dim > 1:
        element = VectorElement(element, dim=dim)

    return FunctionSpace(mesh, element)


def check_function_space_type(function_space):
    """Return whether a function space is scalar, vector, or mixed.

    Parameters
    ----------
    function_space : Firedrake FunctionSpace
        Function space.
    """
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
                "Function space topology of "
                f"{function_space.topological} not supported"
            )
    else:
        raise ValueError(
            f"Function space size of {function_space.value_size} not supported"
        )
