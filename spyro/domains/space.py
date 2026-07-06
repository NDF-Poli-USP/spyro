from numbers import Integral

from firedrake import (
    FiniteElement,
    FunctionSpace,
    VectorElement,
)
from ufl.finiteelement import AbstractFiniteElement


# Each entry is (family, variant, fixed_degree):
# - family: Firedrake finite element family passed to FiniteElement.
# - variant: optional Firedrake variant. None uses Firedrake's default.
# - fixed_degree: required degree for aliases such as DG0. None means the
#   caller-provided degree is used.
_ELEMENT_SPECS = {
    "mass_lumped_triangle": ("KMV", None, None),
    "KMV": ("KMV", None, None),
    "Kong-Mulder-Veldhuizen": ("KMV", None, None),
    "spectral_quadrilateral": ("CG", "spectral", None),
    "DG0": ("DG", None, 0),
    "DG_triangle": ("DG", None, None),
    "DG_quadrilateral": ("DG", None, None),
    "DG": ("DG", None, None),
    "CG_triangle": ("CG", None, None),
    "CG_quadrilateral": ("CG", None, None),
    "CG": ("CG", None, None),
    "DQ_quadrilateral": ("DQ", "spectral", None),
    "DQ": ("DQ", "spectral", None),
}


def _is_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def create_function_space(mesh, method, degree=None, dim=1):
    """Create a Firedrake function space from a mesh and finite element.

    Parameters:
    -----------
    mesh: Firedrake Mesh
        Mesh to be used in the finite element space.
    method: str or FiniteElement
        Method to be used for the finite element space, or an already
        constructed finite element.
    degree: int or None
        Degree of the finite element space. Required when ``method`` is a
        supported method name. Ignored only when ``method`` is an already
        constructed finite element.
    dim: int
        Number of vector components. If ``dim`` is 1, a scalar function space
        is created. If ``dim`` is greater than 1, the selected element is
        wrapped in a vector element with ``dim`` components.

    Returns:
    --------
    function_space: Firedrake FunctionSpace
        Function space.
    """

    if not _is_integer(dim) or dim < 1:
        raise ValueError("Function space dimension must be a positive integer")
    dim = int(dim)

    if isinstance(method, AbstractFiniteElement):
        if degree is not None:
            raise ValueError(
                "degree must be None when method is an already constructed "
                "finite element"
            )
        element = method
    else:
        try:
            family, variant, fixed_degree = _ELEMENT_SPECS[method]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Finite element method {method} not supported"
            ) from exc

        if fixed_degree is not None:
            if degree != fixed_degree:
                raise ValueError(
                    f"Finite element method {method} requires degree {fixed_degree}"
                )
            degree = fixed_degree
        elif not _is_integer(degree) or degree < 0:
            raise ValueError(
                f"Finite element method {method} requires a non-negative integer degree"
            )
        else:
            degree = int(degree)

        element = FiniteElement(
            family, mesh.ufl_cell(), degree=degree, variant=variant,
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
