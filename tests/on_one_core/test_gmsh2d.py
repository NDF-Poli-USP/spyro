import importlib.util
import math

import pytest

from spyro.meshing.meshing_functions import AutomaticMesh
from spyro.meshing.meshing_parameters import MeshingParameters


HAS_NUMBA = importlib.util.find_spec("numba") is not None

CELL_COUNT_TOLERANCE = 0.25


WATER_INTERFACE_CASES = pytest.mark.parametrize(
    "water_interface", [True, False], ids=["water", "no_water"]
)
PADDING_CASES = pytest.mark.parametrize(
    "padding_type", [None, "rectangular", "hyperelliptical"]
)


# Winslow smoothing moves nodes but does not change the topology, so the
# structured cell count does not depend on the Winslow implementation.
STRUCTURED_EXPECTED_CELLS = {
    # (water_interface, padding_type): expected_cells
    (True, None): 770,
    (True, "rectangular"): 1421,
    (True, "hyperelliptical"): 1323,
    (False, None): 735,
    (False, "rectangular"): 1372,
    (False, "hyperelliptical"): 1225,
}


UNSTRUCTURED_EXPECTED_CELLS = {
    # (extend_segy, water_interface, padding_type): expected_cells
    (True, True, None): 17196,
    (True, True, "rectangular"): 24932,
    (True, True, "hyperelliptical"): 24526,
    (True, False, None): 17304,
    (True, False, "rectangular"): 25113,
    (True, False, "hyperelliptical"): 24653,
    (False, True, None): 17196,
    (False, True, "rectangular"): 18086,
    (False, True, "hyperelliptical"): 18115,
    (False, False, None): 17304,
    (False, False, "rectangular"): 18218,
    (False, False, "hyperelliptical"): 18215,
}


def _base_gmsh2d_parameters(output_filename):
    """Parameters for the 2-D velocity model."""
    return {
        "mesh_type": "gmsh_mesh",
        "dimension": 2,
        "length_z": 3000.0,
        "length_x": 5000.0,
        "output_filename": str(output_filename),
        "segy_velocity_model": "velocity_models/vs_example_2D.segy",
        "cells_per_wavelength": 2.0,
        "source_frequency": 10.0,
        "padding_x": 1000.0,
        "padding_z": 1000.0,
        "hyper_n": 4.0,
        "hmin_segy": 0.0,
        "grade": 0.1,
        "water_search_value": 0.0,
        "vp_water": 500.0,
        "min_element_size": 150.0,
        "winslow_iterations": 100,
        "winslow_omega": 0.5,
        "h_padding": 500.0,
    }


def _create_mesh_and_count_cells(parameters):
    mesh_params = MeshingParameters(input_mesh_dictionary=parameters)
    mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)
    firedrake_mesh = mesh_generator.create_mesh()

    if (
        not hasattr(firedrake_mesh.topology, "_entity_classes")
        and hasattr(firedrake_mesh, "init")
    ):
        firedrake_mesh.init()

    return firedrake_mesh.cell_set.core_size


def _assert_cell_count(
    actual_cells: int,
    expected_cells: int,
    relative_tolerance: float = CELL_COUNT_TOLERANCE,
) -> None:
    """Assert that a cell count lies within a relative tolerance of a baseline.

    Parameters
    ----------
    actual_cells : int
        Number of cells in the generated mesh.
    expected_cells : int
        Baseline number of cells.
    relative_tolerance : float, optional
        Allowed relative deviation from ``expected_cells``.

    Raises
    ------
    AssertionError
        If ``actual_cells`` is outside the allowed interval.
    """
    lower_bound = math.floor(expected_cells * (1.0 - relative_tolerance))
    upper_bound = math.ceil(expected_cells * (1.0 + relative_tolerance))

    assert lower_bound <= actual_cells <= upper_bound, (
        f"Got {actual_cells} cells, expected approximately "
        f"{expected_cells} cells "
        f"(allowed interval: {lower_bound} to {upper_bound}, "
        f"tolerance: +/-{100.0 * relative_tolerance:.1f}%)."
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "winslow_implementation",
    [
        None,
        "default",
        "fast",
        pytest.param(
            "numba",
            marks=pytest.mark.skipif(
                not HAS_NUMBA, reason="Numba is not installed."
            ),
        ),
    ],
    ids=["no_winslow", "default", "fast", "numba"],
)
@PADDING_CASES
@WATER_INTERFACE_CASES
def test_gmsh2d_structured(
    tmp_path, water_interface, padding_type, winslow_implementation
):
    """Structured quadrilateral meshes, with and without Winslow smoothing."""
    parameters = _base_gmsh2d_parameters(tmp_path / "mesh.msh")
    parameters.update(
        {
            "padding_type": padding_type,
            "water_interface": water_interface,
            "structured_mesh": True,
            "apply_winslow": winslow_implementation is not None,
            "winslow_implementation": winslow_implementation or "default",
            "extend_segy": False,
        }
    )

    _assert_cell_count(
        _create_mesh_and_count_cells(parameters),
        STRUCTURED_EXPECTED_CELLS[(water_interface, padding_type)],
    )


@pytest.mark.slow
@PADDING_CASES
@WATER_INTERFACE_CASES
@pytest.mark.parametrize(
    "extend_segy", [True, False], ids=["extend_segy", "no_extend_segy"]
)
def test_gmsh2d_unstructured(
    tmp_path, extend_segy, water_interface, padding_type
):
    """Unstructured triangular meshes."""
    parameters = _base_gmsh2d_parameters(tmp_path / "mesh.msh")
    parameters.update(
        {
            "padding_type": padding_type,
            "water_interface": water_interface,
            "structured_mesh": False,
            "apply_winslow": False,
            "winslow_implementation": "default",
            "extend_segy": extend_segy,
        }
    )

    _assert_cell_count(
        _create_mesh_and_count_cells(parameters),
        UNSTRUCTURED_EXPECTED_CELLS[
            (extend_segy, water_interface, padding_type)
        ],
    )
