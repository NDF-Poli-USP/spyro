import importlib.util
import math

import pytest

from spyro.meshing.meshing_functions import AutomaticMesh
from spyro.meshing.meshing_parameters import MeshingParameters


HAS_NUMBA = importlib.util.find_spec("numba") is not None


STRUCTURED_CONFIGURATIONS = [
    # water_interface, padding_type, expected_cells
    (True, None, 26950),
    (True, "rectangular", 69629),
    (False, None, 25725),
    (False, "rectangular", 67228),
]


UNSTRUCTURED_CONFIGURATIONS = [
    # extend_segy, water_interface, padding_type, expected_cells
    (True, True, None, 21443),
    (True, True, "rectangular", 50080),
    (True, True, "hyperelliptical", 42361),
    (True, False, None, 21881),
    (True, False, "rectangular", 50649),
    (True, False, "hyperelliptical", 42954),
    (False, True, None, 21443),
    (False, True, "rectangular", 34420),
    (False, True, "hyperelliptical", 30324),
    (False, False, None, 21881),
    (False, False, "rectangular", 34260),
    (False, False, "hyperelliptical", 30730),
]


STRUCTURED_NO_WINSLOW_CONFIGURATIONS = [
    # water_interface, padding_type, expected_cells
    (True, None, 26950),
    (True, "rectangular", 69629),
    (False, None, 25725),
    (False, "rectangular", 67228),
]


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


def _check_or_collect_baseline(
    *,
    actual_cells,
    expected_cells,
    description,
    baseline_row,
    missing_baselines,
):
    """Assert a known baseline."""
    print(f" Cells actual: {actual_cells} | Expected: {expected_cells}")

    if expected_cells is None:
        missing_baselines.append((*baseline_row, actual_cells))
        return

    assert math.isclose(actual_cells, expected_cells), (
        f"FAILED: {description}. "
        f"Got {actual_cells}, expected {expected_cells}"
    )


def _fail_with_baselines(table_name, missing_baselines):
    """Report all unknown baselines only after all configurations have run."""
    if not missing_baselines:
        return

    pytest.fail(
        f"\nCell-count baselines are not filled for {table_name}.\n",
        pytrace=False,
    )


@pytest.mark.slow
def test_gmsh3d_structured(tmp_path):
    """Structured hexahedra, with/without water and rectangular padding."""
    print("STARTING 3-D STRUCTURED MESH TESTS")

    if not HAS_NUMBA:
        pytest.skip(
            "Skipping the accelerated 3-D Winslow regression test because "
            "Numba is not installed."
        )

    missing_baselines = []

    for water_interface, padding_type, expected_cells in STRUCTURED_CONFIGURATIONS:
        print(
            "\nTesting 3-D Struct"
            " | Winslow: numba"
            f" | Water: {str(water_interface):<5}"
            f" | Padding: {padding_type}"
        )

        output_filename = (
            tmp_path
            / (
                "example3d_struct_numba"
                f"_wat_{water_interface}"
                f"_pad_{padding_type}.msh"
            )
        )

        parameters = _base_gmsh3d_parameters(output_filename)
        parameters.update(
            {
                "padding_type": padding_type,
                "water_interface": water_interface,
                "structured_mesh": True,
                "apply_winslow": True,
                "winslow_implementation": "numba",
                "extend_segy": False,
            }
        )

        actual_cells = _create_mesh_and_count_cells(parameters)

        description = (
            "3-D Struct"
            " | Winslow: numba"
            f" | Wat: {water_interface}"
            f" | Pad: {padding_type}"
        )
        _check_or_collect_baseline(
            actual_cells=actual_cells,
            expected_cells=expected_cells,
            description=description,
            baseline_row=(water_interface, padding_type),
            missing_baselines=missing_baselines,
        )

    _fail_with_baselines(
        "STRUCTURED_CONFIGURATIONS",
        missing_baselines,
    )


@pytest.mark.slow
def test_gmsh3d_unstructured(tmp_path):
    """Unstructured tetrahedra."""
    print("STARTING 3-D UNSTRUCTURED MESH TESTS")

    missing_baselines = []

    for (
        extend_segy,
        water_interface,
        padding_type,
        expected_cells,
    ) in UNSTRUCTURED_CONFIGURATIONS:
        print(
            "\nTesting 3-D Unstruct"
            f" | Ext SEGY: {str(extend_segy):<5}"
            f" | Water: {str(water_interface):<5}"
            f" | Padding: {padding_type}"
        )

        output_filename = (
            tmp_path
            / (
                "example3d_unstruct"
                f"_ext_{extend_segy}"
                f"_wat_{water_interface}"
                f"_pad_{padding_type}.msh"
            )
        )

        parameters = _base_gmsh3d_parameters(output_filename)
        parameters.update(
            {
                "padding_type": padding_type,
                "water_interface": water_interface,
                "structured_mesh": False,
                "apply_winslow": False,
                # Ignored because apply_winslow=False.
                "winslow_implementation": "numba",
                "extend_segy": extend_segy,
            }
        )

        actual_cells = _create_mesh_and_count_cells(parameters)

        description = (
            "3-D Unstruct"
            f" | Ext: {extend_segy}"
            f" | Wat: {water_interface}"
            f" | Pad: {padding_type}"
        )
        _check_or_collect_baseline(
            actual_cells=actual_cells,
            expected_cells=expected_cells,
            description=description,
            baseline_row=(extend_segy, water_interface, padding_type),
            missing_baselines=missing_baselines,
        )

    _fail_with_baselines(
        "UNSTRUCTURED_CONFIGURATIONS",
        missing_baselines,
    )


@pytest.mark.slow
def test_gmsh3d_structured_no_winslow(tmp_path):
    """Structured hexahedra without Winslow smoothing."""
    print("STARTING 3-D STRUCTURED MESH (NO WINSLOW) TESTS")

    missing_baselines = []

    for (
        water_interface,
        padding_type,
        expected_cells,
    ) in STRUCTURED_NO_WINSLOW_CONFIGURATIONS:
        print(
            "\nTesting 3-D Struct (No Winslow)"
            f" | Water: {str(water_interface):<5}"
            f" | Padding: {padding_type}"
        )

        output_filename = (
            tmp_path
            / (
                "example3d_struct_nowinslow"
                f"_wat_{water_interface}"
                f"_pad_{padding_type}.msh"
            )
        )

        parameters = _base_gmsh3d_parameters(output_filename)
        parameters.update(
            {
                "padding_type": padding_type,
                "water_interface": water_interface,
                "structured_mesh": True,
                "apply_winslow": False,
                # Ignored because apply_winslow=False.
                "winslow_implementation": "numba",
                "extend_segy": False,
            }
        )

        actual_cells = _create_mesh_and_count_cells(parameters)

        description = (
            "3-D Struct (No Winslow)"
            f" | Wat: {water_interface}"
            f" | Pad: {padding_type}"
        )
        _check_or_collect_baseline(
            actual_cells=actual_cells,
            expected_cells=expected_cells,
            description=description,
            baseline_row=(water_interface, padding_type),
            missing_baselines=missing_baselines,
        )

    _fail_with_baselines(
        "STRUCTURED_NO_WINSLOW_CONFIGURATIONS",
        missing_baselines,
    )
