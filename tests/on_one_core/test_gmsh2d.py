import importlib.util
import math

import pytest

from spyro.meshing.meshing_functions import AutomaticMesh
from spyro.meshing.meshing_parameters import MeshingParameters


HAS_NUMBA = importlib.util.find_spec("numba") is not None

CELL_COUNT_TOLERANCE = 0.25


STRUCTURED_CONFIGURATIONS = [
    # water_interface, padding_type, expected_cells

    (True, None, 770),
    (True, 'rectangular', 1421),
    (True, 'hyperelliptical', 1323),
    (False, None, 735),
    (False, 'rectangular', 1372),
    (False, 'hyperelliptical', 1225),
]


UNSTRUCTURED_CONFIGURATIONS = [
    # extend_segy, water_interface, padding_type, expected_cells

    (True, True, None, 17196),
    (True, True, 'rectangular', 24932),
    (True, True, 'hyperelliptical', 24526),
    (True, False, None, 17304),
    (True, False, 'rectangular', 25113),
    (True, False, 'hyperelliptical', 24653),
    (False, True, None, 17196),
    (False, True, 'rectangular', 18086),
    (False, True, 'hyperelliptical', 18115),
    (False, False, None, 17304),
    (False, False, 'rectangular', 18218),
    (False, False, 'hyperelliptical', 18215),
]


STRUCTURED_NO_WINSLOW_CONFIGURATIONS = [
    # water_interface, padding_type, expected_cells

    (True, None, 770),
    (True, 'rectangular', 1421),
    (True, 'hyperelliptical', 1323),
    (False, None, 735),
    (False, 'rectangular', 1372),
    (False, 'hyperelliptical', 1225),
]


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


def _check_or_collect_baseline(
    *,
    actual_cells,
    expected_cells,
    description,
    baseline_row,
    missing_baselines,
    relative_tolerance=CELL_COUNT_TOLERANCE,
):
    """Assert a cell-count baseline within an allowed relative interval."""
    if expected_cells is None:
        print(
            f"     Cells actual: {actual_cells}"
            " | Expected: not defined"
            " | Baseline collected"
        )
        missing_baselines.append((*baseline_row, actual_cells))
        return

    lower_bound = math.floor(
        expected_cells * (1.0 - relative_tolerance)
    )
    upper_bound = math.ceil(
        expected_cells * (1.0 + relative_tolerance)
    )

    print(
        f"     Cells actual: {actual_cells}"
        f" | Expected: {expected_cells}"
        f" | Allowed: [{lower_bound}, {upper_bound}]"
    )

    assert lower_bound <= actual_cells <= upper_bound, (
        f"FAILED: {description}. "
        f"Got {actual_cells} cells, expected approximately "
        f"{expected_cells} cells "
        f"(allowed interval: {lower_bound} to {upper_bound}, "
        f"tolerance: +/-{100.0 * relative_tolerance:.1f}%)."
    )


def _fail_with_baselines(table_name, missing_baselines):
    """Report collected baselines after every configuration has run."""
    if not missing_baselines:
        return

    lines = [
        "",
        f"Cell-count baselines are not filled for {table_name}.",
        "Replace the None values with these measured baselines:",
        "",
    ]

    for row in missing_baselines:
        lines.append(f"    {row},")

    lines.append("")

    pytest.fail(
        "\n".join(lines),
        pytrace=False,
    )


@pytest.mark.slow
def test_gmsh2d_structured(tmp_path):
    """Structured quadrilateral meshes"""
    print("STARTING STRUCTURED MESH TESTS")

    winslow_implementations = ["default", "fast"]

    if HAS_NUMBA:
        print("Numba is installed, making numba test")
        winslow_implementations.append("numba")
    else:
        print(
            "\n[INFO] Numba is not installed. "
            "Skipping 'numba' Winslow implementation."
        )

    missing_baselines = []

    for winslow_impl in winslow_implementations:
        for (
            water_interface,
            padding_type,
            expected_cells,
        ) in STRUCTURED_CONFIGURATIONS:
            print(
                "\nTesting Struct"
                f" | Winslow: {winslow_impl:<7}"
                f" | Water: {str(water_interface):<5}"
                f" | Padding: {padding_type}"
            )

            output_filename = (
                tmp_path
                / (
                    "vs_example_2d_struct"
                    f"_{winslow_impl}"
                    f"_wat_{water_interface}"
                    f"_pad_{padding_type}.msh"
                )
            )

            parameters = _base_gmsh2d_parameters(output_filename)
            parameters.update(
                {
                    "padding_type": padding_type,
                    "water_interface": water_interface,
                    "structured_mesh": True,
                    "apply_winslow": True,
                    "winslow_implementation": winslow_impl,
                    "extend_segy": False,
                }
            )

            actual_cells = _create_mesh_and_count_cells(parameters)

            description = (
                "2-D Struct"
                f" | Winslow: {winslow_impl}"
                f" | Wat: {water_interface}"
                f" | Pad: {padding_type}"
            )

            _check_or_collect_baseline(
                actual_cells=actual_cells,
                expected_cells=expected_cells,
                description=description,
                baseline_row=(
                    winslow_impl,
                    water_interface,
                    padding_type,
                ),
                missing_baselines=missing_baselines,
            )

    _fail_with_baselines(
        "STRUCTURED_CONFIGURATIONS",
        missing_baselines,
    )


@pytest.mark.slow
def test_gmsh2d_unstructured(tmp_path):
    """Unstructured triangular meshes"""
    print("STARTING UNSTRUCTURED MESH TESTS")

    missing_baselines = []

    for (
        extend_segy,
        water_interface,
        padding_type,
        expected_cells,
    ) in UNSTRUCTURED_CONFIGURATIONS:
        print(
            "\nTesting Unstruct"
            f" | Ext SEGY: {str(extend_segy):<5}"
            f" | Water: {str(water_interface):<5}"
            f" | Padding: {padding_type}"
        )

        output_filename = (
            tmp_path
            / (
                "vs_example_2d_unstruct"
                f"_ext_{extend_segy}"
                f"_wat_{water_interface}"
                f"_pad_{padding_type}.msh"
            )
        )

        parameters = _base_gmsh2d_parameters(output_filename)
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

        actual_cells = _create_mesh_and_count_cells(parameters)

        description = (
            "2-D Unstruct"
            f" | Ext: {extend_segy}"
            f" | Wat: {water_interface}"
            f" | Pad: {padding_type}"
        )

        _check_or_collect_baseline(
            actual_cells=actual_cells,
            expected_cells=expected_cells,
            description=description,
            baseline_row=(
                extend_segy,
                water_interface,
                padding_type,
            ),
            missing_baselines=missing_baselines,
        )

    _fail_with_baselines(
        "UNSTRUCTURED_CONFIGURATIONS",
        missing_baselines,
    )


@pytest.mark.slow
def test_gmsh2d_structured_no_winslow(tmp_path):
    """Structured quadrilateral meshes without Winslow smoothing."""
    print("STARTING STRUCTURED MESH (NO WINSLOW) TESTS")

    missing_baselines = []

    for (
        water_interface,
        padding_type,
        expected_cells,
    ) in STRUCTURED_NO_WINSLOW_CONFIGURATIONS:
        print(
            "\nTesting Struct (No Winslow)"
            f" | Water: {str(water_interface):<5}"
            f" | Padding: {padding_type}"
        )

        output_filename = (
            tmp_path
            / (
                "vs_example_2d_struct_nowinslow"
                f"_wat_{water_interface}"
                f"_pad_{padding_type}.msh"
            )
        )

        parameters = _base_gmsh2d_parameters(output_filename)
        parameters.update(
            {
                "padding_type": padding_type,
                "water_interface": water_interface,
                "structured_mesh": True,
                "apply_winslow": False,
                "winslow_implementation": "default",
                "extend_segy": False,
            }
        )

        actual_cells = _create_mesh_and_count_cells(parameters)

        description = (
            "2-D Struct (No Winslow)"
            f" | Wat: {water_interface}"
            f" | Pad: {padding_type}"
        )

        _check_or_collect_baseline(
            actual_cells=actual_cells,
            expected_cells=expected_cells,
            description=description,
            baseline_row=(
                water_interface,
                padding_type,
            ),
            missing_baselines=missing_baselines,
        )

    _fail_with_baselines(
        "STRUCTURED_NO_WINSLOW_CONFIGURATIONS",
        missing_baselines,
    )
