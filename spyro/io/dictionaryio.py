"""Class for reading option section int he input dictionary."""

from enum import Enum
from pydantic import BaseModel, field_validator, model_validator, ConfigDict


class ListEnum(Enum):
    def __new__(cls, value, *aliases):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.aliases = aliases
        return obj

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if value in member.aliases:
                return member

        return None


class Method(ListEnum):
    MASS_LUMPED_TRIANGLE = (
        "mass_lumped_triangle",
        "KMV",
        "MLT",
        "mass_lumped_tetrahedra",
    )
    SPECTRAL_QUADRILATERAL = ("spectral_quadrilateral", "spectral", "SEM")
    DISCONTINUOUS_GALERKIN_TRIANGLE = (
        "DG_triangle",
        "DGT",
        "discontinuous_galerkin_triangle",
    )
    DISCONTINUOUS_GALERKIN_QUADRILATERAL = (
        "DG_quadrilateral",
        "DGQ",
        "discontinuous_galerkin_quadrilateral",
    )

    CG = ("CG",)


class CellType(ListEnum):
    TRIANGLE = ("triangle", "T", "triangles", "tetrahedra", "tetrahedron")
    QUADRILATERAL = ("quadrilateral", "Q", "quadrilaterals", "hexahedra", "hexahedron")


class Variant(Enum):
    LUMPED = "lumped"
    EQUISPACED = "equispaced"
    DG = "DG"


class Analysis(Enum):
    MODAL = "modal"
    EIKONAL = "eikonal"
    TRANSIENT = "transient"


class Read_options(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    degree: int
    dimension: int
    analysis: Analysis = Analysis.TRANSIENT.value
    variant: Variant | None = None
    method: Method | None = None
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
    analysis : `str`
        The type of analysis to be performed. Can be 'transient', 'modal' or 'eikonal'.

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
        if self.method == Method.CG.value and (
            self.variant is None or self.cell_type is None
        ):
            raise ValueError("Can't use CG without specifying cell type and variant.")

        if self.cell_type is None:
            self._set_cell_type()

        self._validate_cell_type()

        if (
            self.variant is not None
            and self.cell_type is not None
            and self.method is None
        ):
            self._set_default_method()

        return self

    def _set_cell_type(self):
        if self.method is None:
            return

        if self.method in (
            Method.MASS_LUMPED_TRIANGLE.value,
            Method.DISCONTINUOUS_GALERKIN_TRIANGLE.value,
        ):
            self.cell_type = CellType.TRIANGLE

        elif self.method in (
            Method.SPECTRAL_QUADRILATERAL.value,
            Method.DISCONTINUOUS_GALERKIN_QUADRILATERAL.value,
        ):
            self.cell_type = CellType.QUADRILATERAL

    def _validate_cell_type(self):
        if self.method is None or self.method == Method.CG.value:
            return

        if (
            self.cell_type == CellType.TRIANGLE.value
            and self.method
            not in Method.MASS_LUMPED_TRIANGLE.value
            + Method.DISCONTINUOUS_GALERKIN_TRIANGLE.value
        ):
            raise ValueError(
                f"Cell type '{self.cell_type}' is not "
                f"compatible with method '{self.method}'."
            )

        if (
            self.cell_type == CellType.QUADRILATERAL.value
            and self.method
            not in Method.DISCONTINUOUS_GALERKIN_QUADRILATERAL.value
            + Method.SPECTRAL_QUADRILATERAL.value
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
            self.method = default_method[CellType(self.cell_type)][
                Variant(self.variant)
            ].value
        except KeyError:
            raise ValueError(
                f"Cell type '{self.cell_type}' not compatible "
                f"with variant '{self.variant}'."
            )


class Read_outputs(BaseModel):
    forward_output: bool = True
    forward_output_filename: str = "results/forward_output.pvd"
    gradient_filename: str | None = None
    adjoint_filename: str | None = None
    time_filename: str | None = None
    acoustic_energy_filename: str | None = None
    acoustic_energy: bool = False
    output_folder: str = "output/"
    debug_output: bool = False

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        if hasattr(self, key):
            return getattr(self, key)

        if key.endswith("_output") and hasattr(
            self, key.replace("_output", "_filename")
        ):
            return getattr(self, key.replace("_output", "_filename")) is not None

        return default
