
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator

class ListEnum(Enum):
    @classmethod
    def from_string(cls, value):
        for member in cls:
            if value in member.value:
                return member
        raise ValueError(f"Unknown value: {value}")

class Method(ListEnum):
    MASS_LUMPED_TRIANGLE = (
        "KMV", "MLT", "mass_lumped_triangle", "mass_lumped_tetrahedra"
    )
    SPECTRAL_QUADRILATERAL = (
        "spectral", "SEM", "spectral_quadrilateral"
    )
    DISCONTINUOUS_GALERKIN_TRIANGLE = (
        "DG_triangle", "DGT", "discontinuous_galerkin_triangle"
    )
    DISCONTINUOUS_GALERKIN_QUADRILATERAL = (
        "DG_quadrilateral", "DGQ", "discontinuous_galerkin_quadrilateral"
    )

    CG = ("CG",)

class CellType(ListEnum):
    TRIANGLE = ("T", "triangle", "triangles", "tetrahedra", "tetrahedron")
    QUADRILATERAL = ("Q", "quadrilateral", "quadrilaterals", "hexahedra", "hexahedron")

class Variant(Enum):
    LUMPED = "lumped"
    EQUISPACED = "equispaced"
    DG = "DG"

class Read_options(BaseModel):
    degree: int
    dimension: int
    variant: Variant | None = None
    method: Method | None =  None
    cell_type: CellType | None = None
    automatic_adjoint: bool = False

    """
    Read the options section of the dictionary.

    Attributes
    ----------
    options_dictionary : dict
        Dictionary containing the options information.
    cell_type : str
        The cell type to be used.
    method : str
        The FEM method to be used.
    variant : str
        The quadrature variant to be used.
    degree : int
        The polynomial degree of the FEM method.
    dimension : int
        The spatial dimension of the problem.
    automatic_adjoint : bool
        Whether to automatically compute the adjoint.

    Methods
    -------
    check_valid_degree()
        Check that the degree is valid for the method.
    _check_valid_degree_for_mlt()
        Check that the degree is valid for the MLT method.
    check_mismatch_cell_type_variant_method()
        Check that the user has not specified both the method and the cell type.
    get_from_method()
        Get the method, cell type and variant from the method.
    get_from_cell_type_variant()
        Get the method, cell type and variant from the cell type and variant.
    """

    @field_validator("degree")
    @classmethod
    def validate_degree(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Degree should be greater than 0.")
        return value

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value not in (2, 3):
            raise ValueError(f"Dimension of {value} not 2 or 3.")
        return value

    @model_validator(mode="after")
    def validate_model(self):
        if (
            self.method == Method.CG
            and (self.variant is None or self.cell_type is None)
        ):
            raise ValueError(
                "Can't use CG without specifying cell type and variant."
            )

        if self.cell_type is None:
            self._set_cell_type()

        self._validate_cell_type()

        if self.variant is not None and self.method is None:
            self._set_default_method()

        return self


    def _set_cell_type(self):
        if self.method in (
            Method.MASS_LUMPED_TRIANGLE,
            Method.DISCONTINUOUS_GALERKIN_TRIANGLE,
        ):
            self.cell_type = CellType.TRIANGLE

        elif self.method in (
            Method.SPECTRAL_QUADRILATERAL,
            Method.DISCONTINUOUS_GALERKIN_QUADRILATERAL,
        ):
            self.cell_type = CellType.QUADRILATERAL

    def _validate_cell_type(self):
        if (
            self.cell_type == CellType.TRIANGLE and self.method
            not in (
                Method.MASS_LUMPED_TRIANGLE,
                Method.DISCONTINUOUS_GALERKIN_TRIANGLE,
            )
        ):
            raise ValueError(
                f"Cell type '{self.cell_type}' is not "
                f"compatible with method '{self.method}'."
            )

        if (
            self.cell_type == CellType.QUADRILATERAL and self.method
            not in (
                Method.DISCONTINUOUS_GALERKIN_QUADRILATERAL,
                Method.SPECTRAL_QUADRILATERAL,
            )
        ):
            raise ValueError(
                f"Cell type '{self.cell_type}' is not "
                f"compatible with method '{self.method}'."
            )

    def _set_default_method(self):
        default_method = {
            CellType.TRIANGLE: {
                Variant.LUMPED: Method.MASS_LUMPED_TRIANGLE,
                Variant.DG: Method.DISCONTINUOUS_GALERKIN_TRIANGLE,
                Variant.EQUISPACED: Method.CG,
            },
            CellType.QUADRILATERAL: {
                Variant.LUMPED: Method.SPECTRAL_QUADRILATERAL,
                Variant.DG: Method.DISCONTINUOUS_GALERKIN_QUADRILATERAL,
                Variant.EQUISPACED: Method.CG,
            },
        }

        try:
            self.method = default_method[self.cell_type][self.variant]
        except KeyError:
            raise ValueError(
                f"Cell type '{self.cell_type}' not compatible "
                f"with variant '{self.variant}'."
            )

class Read_outputs(BaseModel):
    forward_output_filename: str = "results/forward.pvd"
    gradient_filename: str = "results/gradient.pvd"
    adjoint_filename: str = "results/adjoint.pvd"
    debug_output: bool = False
