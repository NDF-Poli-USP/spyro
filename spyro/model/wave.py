from dataclasses import dataclass, field
from typing import Any
from firedrake import (
    FiniteElement, FunctionSpace, VectorElement, Function
)
from spyro.model.mesh import FiniteElementMethod, MeshDefinition
from ufl.finiteelement import AbstractFiniteElement

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

@dataclass
class Wave:
    function_space: FunctionSpace = field(init=False)
    scalar_function_space: FunctionSpace = field(init=False)
    velocity_model: Function = field(init=False) # remeber to set this later
    quadrature_rule: Any
    c: Any
    current_time: float = field(default=0, init=False)

    def __post_init__(self, mesh: MeshDefinition, initial_velocity_file: str):
        self._validate_args(mesh.finite_element_method, mesh.degree, mesh.dimension)
        self.function_space = self._get_function_space(mesh.get_mesh(), mesh.method, mesh.degree, mesh.dimension)
        
    def _validate_args(method: FiniteElementMethod, degree: int, dimension: int):
        if dimension is None or dimension < 1:
            raise ValueError(f"Dimension must be greater than 1, but found {dimension}")
        
        if degree is None or degree < 0:
            raise ValueError(f"Degree must be a positive integer, but found {dimension}")

        if method not in _ELEMENT_SPECS:
            raise ValueError(f"Finite element method {method} not supported")

    @classmethod
    def from_finite_element(self, mesh_definition: MeshDefinition, initial_velocity_file: str, element: AbstractFiniteElement):
        self.function_space = FunctionSpace(mesh_definition.get_mesh(), element)

    def _get_function_space(self, mesh: object, method: FiniteElementMethod, degree: int, dimension: int = 1):
        family, variant, fixed_degree = _ELEMENT_SPECS[method]

        degree = degree if fixed_degree is not None else fixed_degree

        element = FiniteElement(family, mesh.ufl_cell(), degree=degree, variant=variant)

        if dimension > 1:
            return FunctionSpace(mesh, VectorElement(element, dim=dimension))

        return FunctionSpace(mesh, element)