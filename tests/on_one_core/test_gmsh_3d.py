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


# Winslow smoothing moves nodes but does not change the topology, so the
# structured cell count is the same with and without Winslow.
STRUCTURED_EXPECTED_CELLS = {
    # (water_interface, padding_type): expected_cells
    (True, None): 26950,
    (True, "rectangular"): 69629,
    (False, None): 25725,
    (False, "rectangular"): 67228,
}


UNSTRUCTURED_EXPECTED_CELLS = {
    # (extend_segy, water_interface, padding_type): expected_cells
    (True, True, None): 21443,
    (True, True, "rectangular"): 50080,
    (True, True, "hyperelliptical"): 42361,
    (True, False, None): 21881,
    (True, False, "rectangular"): 50649,
    (True, False, "hyperelliptical"): 42954,
    (False, True, None): 21443,
    (False, True, "rectangular"): 34420,
    (False, True, "hyperelliptical"): 30324,
    (False, False, None): 21881,
    (False, False, "rectangular"): 34260,
    (False, False, "hyperelliptical"): 30730,
}


def _base_gmsh3d_parameters(output_filename):
    return {
        "mesh_type": "gmsh_mesh",
        "dimension": 3,
        "length_z": 3000.0,
        "length_x": 5000.0,
        "length_y": 5000.0,
        "output_filename": str(output_filename),
        "segy_velocity_model": "velocity_models/vs_example_3D.bin",
        "segy_nz": 101,
        "segy_nx": 101,
        "segy_ny": 101,
        "segy_dz": 30.0,
        "segy_dx": 50.0,
        "segy_dy": 50.0,
        "segy_byte_order": "big",
        "segy_axes_order": (0, 1, 2),
        "segy_axes_order_sort": "F",
        "segy_dtype": "float32",
        "cells_per_wavelength": 2.0,
        "source_frequency": 3.0,
        "padding_x": 1000.0,
        "padding_y": 1000.0,
        "padding_z": 1000.0,
        "hyper_n": 4.0,
        "hmin_segy": 0.0,
        "grade": 0.05,
        "water_search_value": 0.0,
        "vp_water": 1000.0,
        "min_element_size": 150.0,
        "winslow_iterations": 100,
        "winslow_omega": 0.5,
        "h_padding": 500.0,
        "gmsh_parallel": False,
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
    "apply_winslow",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not HAS_NUMBA, reason="Numba is not installed."
            ),
        ),
    ],
    ids=["no_winslow", "numba"],
)
@pytest.mark.parametrize("padding_type", [None, "rectangular"])
@WATER_INTERFACE_CASES
def test_gmsh3d_structured(
    tmp_path, water_interface, padding_type, apply_winslow
):
    """Structured hexahedra, with and without Winslow smoothing."""
    parameters = _base_gmsh3d_parameters(tmp_path / "mesh.msh")
    parameters.update(
        {
            "padding_type": padding_type,
            "water_interface": water_interface,
            "structured_mesh": True,
            "apply_winslow": apply_winslow,
            # Ignored when apply_winslow=False.
            "winslow_implementation": "numba",
            "extend_segy": False,
        }
    )

    _assert_cell_count(
        _create_mesh_and_count_cells(parameters),
        STRUCTURED_EXPECTED_CELLS[(water_interface, padding_type)],
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "gmsh_parallel", [False, True], ids=["serial_gmsh", "parallel_gmsh"]
)
@pytest.mark.parametrize(
    "padding_type", [None, "rectangular", "hyperelliptical"]
)
@WATER_INTERFACE_CASES
@pytest.mark.parametrize(
    "extend_segy", [True, False], ids=["extend_segy", "no_extend_segy"]
)
def test_gmsh3d_unstructured(
    tmp_path, extend_segy, water_interface, padding_type, gmsh_parallel
):
    """Unstructured tetrahedra, with serial and parallel Gmsh sizing."""
    parameters = _base_gmsh3d_parameters(tmp_path / "mesh.msh")
    parameters.update(
        {
            "padding_type": padding_type,
            "water_interface": water_interface,
            "structured_mesh": False,
            "apply_winslow": False,
            # Ignored because apply_winslow=False.
            "winslow_implementation": "numba",
            "extend_segy": extend_segy,
            "gmsh_parallel": gmsh_parallel,
        }
    )

    _assert_cell_count(
        _create_mesh_and_count_cells(parameters),
        UNSTRUCTURED_EXPECTED_CELLS[
            (extend_segy, water_interface, padding_type)
        ],
    )


@pytest.mark.slow
def test_gmsh3d_structured_winslow_zero_iterations(tmp_path):
    """Exercise the 3-D Winslow path without requiring JIT compilation."""
    print("STARTING 3-D STRUCTURED WINSLOW ZERO-ITERATION COVERAGE TEST")

    output_filename = (
        tmp_path / "example3d_struct_winslow_zero_iterations.msh"
    )

    parameters = _base_gmsh3d_parameters(output_filename)
    parameters.update(
        {
            "padding_type": None,
            "water_interface": False,
            "structured_mesh": True,
            "apply_winslow": True,
            "winslow_implementation": "numba",
            "winslow_iterations": 0,
            "extend_segy": False,
            "gmsh_parallel": False,
        }
    )

    _assert_cell_count(
        _create_mesh_and_count_cells(parameters),
        STRUCTURED_EXPECTED_CELLS[(False, None)],
    )


@pytest.mark.slow
def test_gmsh3d_structured_hyperelliptical_rejected(tmp_path):
    """Check the guard for unsupported structured padding."""
    print("TESTING 3-D STRUCTURED HYPERELLIPTICAL VALIDATION")

    output_filename = (
        tmp_path / "example3d_struct_hyperelliptical_invalid.msh"
    )

    parameters = _base_gmsh3d_parameters(output_filename)
    parameters.update(
        {
            "padding_type": "hyperelliptical",
            "water_interface": False,
            "structured_mesh": True,
            "apply_winslow": False,
            "winslow_implementation": "numba",
            "extend_segy": False,
            "gmsh_parallel": False,
        }
    )

    with pytest.raises(
        ValueError,
        match="Hyperelliptical 3-D padding currently supports only",
    ):
        _create_mesh_and_count_cells(parameters)


@pytest.mark.slow
def test_gmsh3d_water_interface_rejects_noncanonical_axes(tmp_path):
    """Check the real water geometry axis-order validation."""
    print("TESTING 3-D WATER-INTERFACE AXIS-ORDER VALIDATION")

    output_filename = (
        tmp_path / "example3d_water_invalid_axes_order.msh"
    )

    parameters = _base_gmsh3d_parameters(output_filename)
    parameters.update(
        {
            "padding_type": None,
            "water_interface": True,
            "structured_mesh": False,
            "apply_winslow": False,
            "winslow_implementation": "numba",
            "extend_segy": False,
            "gmsh_parallel": False,
            "segy_axes_order": (1, 0, 2),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"axes_order must therefore be \(0, 1, 2\)",
    ):
        _create_mesh_and_count_cells(parameters)


@pytest.mark.slow
def test_gmsh3d_structured_rejects_non_numba_winslow(tmp_path):
    """Exercise the 3-D Winslow implementation validation."""
    print("TESTING 3-D STRUCTURED WINSLOW IMPLEMENTATION VALIDATION")

    output_filename = (
        tmp_path / "example3d_struct_invalid_winslow.msh"
    )

    parameters = _base_gmsh3d_parameters(output_filename)
    parameters.update(
        {
            "padding_type": None,
            "water_interface": False,
            "structured_mesh": True,
            "apply_winslow": True,
            "winslow_implementation": "default",
            "winslow_iterations": 0,
            "extend_segy": False,
            "gmsh_parallel": False,
        }
    )

    with pytest.raises(
        ValueError,
        match="Only winslow_implementation='numba'",
    ):
        _create_mesh_and_count_cells(parameters)
