import pytest

from spyro.io.dictionaryio import (
    Read_options,
    Method,
    Variant,
    CellType,
)


def test_read_options_minimal():
    options = Read_options(
        degree=2,
        dimension=2,
    )

    assert options.degree == 2
    assert options.dimension == 2
    assert options.variant is None
    assert options.method is None
    assert options.cell_type is None
    assert options.automatic_adjoint is False


def test_read_options_all_fields():
    options = Read_options(
        degree=4,
        dimension=3,
        variant=Variant.LUMPED,
        method=Method.MASS_LUMPED_TRIANGLE,
        cell_type=CellType.TRIANGLE,
        automatic_adjoint=True,
    )

    assert options.degree == 4
    assert options.dimension == 3
    assert options.variant == "lumped"
    assert options.method == "mass_lumped_triangle"
    assert options.cell_type == "triangle"
    assert options.automatic_adjoint is True


def test_read_options_all_fields_with_strings():
    options = Read_options(
        degree=4,
        dimension=3,
        variant="lumped",
        method="mass_lumped_tetrahedra",
        cell_type="triangle",
        automatic_adjoint=True,
    )

    assert options.degree == 4
    assert options.dimension == 3
    assert options.variant == "lumped"
    assert options.method == "mass_lumped_triangle"
    assert options.cell_type == "triangle"
    assert options.automatic_adjoint is True


@pytest.mark.parametrize(
    "degree,dimension,variant,method,cell_type",
    [
        (0, 2, None, None, None),  # invalid degree
        (2, 1, None, None, None),  # invalid dimension
        (2, 2, "invalid", None, None),  # invalid variant
        (2, 2, None, "invalid", None),  # invalid method
        (
            2,
            2,
            Variant.LUMPED,
            Method.SPECTRAL_QUADRILATERAL,
            "invalid",
        ),  # invalid cell type
        (
            2,
            2,
            "lumped",
            Method.SPECTRAL_QUADRILATERAL,
            "invalid",
        ),  # invalid cell type
        (
            2,
            2,
            None,
            Method.CG,
            None,
        ),  # invalid, should specify both variant and cell_type
        (
            2,
            2,
            None,
            Method.MASS_LUMPED_TRIANGLE,
            CellType.QUADRILATERAL,
        ),  # invalid, cell_type should be TRIANGLE
    ],
)
def test_invalid_fields(
    degree,
    dimension,
    variant,
    method,
    cell_type,
):
    with pytest.raises(ValueError):
        Read_options(
            degree=degree,
            dimension=dimension,
            variant=variant,
            method=method,
            cell_type=cell_type,
            automatic_adjoint=True,
        )
