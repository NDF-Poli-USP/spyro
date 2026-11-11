import importlib.util

import pytest

from .gmsh_mesh_checks import (
    VELOCITY_MODELS_DIR,
    assert_bounds,
    assert_cell_count,
    assert_cell_regions,
    assert_cell_type,
    assert_measure,
    assert_no_degenerate_cells,
    assert_region_above,
    create_gmsh_mesh,
    hyperellipsoid_volume,
    measure,
    physical_groups,
)


HAS_NUMBA = importlib.util.find_spec("numba") is not None

LENGTH_Z = 3000.0
LENGTH_X = 5000.0
LENGTH_Y = 5000.0
PADDING = 1000.0
HYPER_N = 4.0

# Structured meshes are transfinite, so their cell count is exact. Unstructured
# counts may change slightly between Gmsh versions; the baselines were made
# with Gmsh 4.13.1.
UNSTRUCTURED_CELL_COUNT_RTOL = 0.02
# The hyperellipsoid is a coarse B-spline surface, so its volume is slightly
# smaller than the exact hyperellipsoid.
HYPERELLIPSOID_VOLUME_RTOL = 0.02


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
    # (extend_segy, water_interface, padding_type, gmsh_parallel): expected_cells
    # The parallel sizing path samples the size field on a grid, so its counts
    # differ from the serial callback by up to about 1%.
    (True, True, None, False): 21443,
    (True, True, None, True): 21613,
    (True, True, "rectangular", False): 50080,
    (True, True, "rectangular", True): 50694,
    (True, True, "hyperelliptical", False): 42361,
    (True, True, "hyperelliptical", True): 42369,
    (True, False, None, False): 21881,
    (True, False, None, True): 21887,
    (True, False, "rectangular", False): 50649,
    (True, False, "rectangular", True): 50873,
    (True, False, "hyperelliptical", False): 42954,
    (True, False, "hyperelliptical", True): 43080,
    (False, True, None, False): 21443,
    (False, True, None, True): 21613,
    (False, True, "rectangular", False): 34420,
    (False, True, "rectangular", True): 34546,
    (False, True, "hyperelliptical", False): 30324,
    (False, True, "hyperelliptical", True): 30519,
    (False, False, None, False): 21881,
    (False, False, None, True): 21887,
    (False, False, "rectangular", False): 34260,
    (False, False, "rectangular", True): 34118,
    (False, False, "hyperelliptical", False): 30730,
    (False, False, "hyperelliptical", True): 30822,
}


def _base_gmsh3d_parameters(output_filename):
    return {
        "mesh_type": "gmsh_mesh",
        "dimension": 3,
        "length_z": LENGTH_Z,
        "length_x": LENGTH_X,
        "length_y": LENGTH_Y,
        "output_filename": str(output_filename),
        "segy_velocity_model": str(VELOCITY_MODELS_DIR / "vs_example_3D.bin"),
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
        "padding_x": PADDING,
        "padding_y": PADDING,
        "padding_z": PADDING,
        "hyper_n": HYPER_N,
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


def _check_mesh(mesh, msh_file, water_interface, padding_type) -> None:
    """Check the geometry and regions of a 3D Gmsh mesh.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh returned by ``AutomaticMesh``.
    msh_file : pathlib.Path
        Gmsh file the mesh was read from.
    water_interface : bool
        Whether the water layer is a separate region.
    padding_type : str or None
        Padding geometry.
    """
    box_volume = LENGTH_Z * LENGTH_X * LENGTH_Y
    if padding_type is None:
        lower, upper = (-LENGTH_Z, 0.0, 0.0), (0.0, LENGTH_X, LENGTH_Y)
        total_volume, volume_rtol, bounds_rtol = box_volume, 1e-9, 1e-9
    else:
        lower = (-LENGTH_Z - PADDING, -PADDING, -PADDING)
        upper = (0.0, LENGTH_X + PADDING, LENGTH_Y + PADDING)
        if padding_type == "rectangular":
            total_volume = (
                (LENGTH_Z + PADDING) * (LENGTH_X + 2 * PADDING)
                * (LENGTH_Y + 2 * PADDING)
            )
            volume_rtol, bounds_rtol = 1e-9, 1e-9
        else:
            total_volume = hyperellipsoid_volume(
                LENGTH_Z, LENGTH_X, LENGTH_Y, PADDING, PADDING, PADDING, HYPER_N
            )
            volume_rtol, bounds_rtol = HYPERELLIPSOID_VOLUME_RTOL, 1e-3

    assert_bounds(mesh, lower, upper, rtol=bounds_rtol)
    assert_measure(mesh, total_volume, rtol=volume_rtol)
    assert_no_degenerate_cells(mesh)

    cell_groups = physical_groups(msh_file)[3]
    core = ("Subsurface", "Water") if water_interface else ("SubsurfaceAndWater",)
    expected_measures = {core: box_volume}
    if padding_type is not None:
        # The total was checked above; the box faces are flat, so the split
        # between model and padding is exact.
        expected_measures[("Padding",)] = measure(mesh) - box_volume
    assert_cell_regions(mesh, cell_groups, expected_measures)
    if water_interface:
        assert_region_above(mesh, cell_groups["Water"], cell_groups["Subsurface"])


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

    mesh = create_gmsh_mesh(parameters)

    assert_cell_type(mesh, "hexahedron")
    assert_cell_count(mesh, STRUCTURED_EXPECTED_CELLS[(water_interface, padding_type)])
    _check_mesh(mesh, parameters["output_filename"], water_interface, padding_type)


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

    mesh = create_gmsh_mesh(parameters)

    assert_cell_type(mesh, "tetrahedron")
    assert_cell_count(
        mesh,
        UNSTRUCTURED_EXPECTED_CELLS[
            (extend_segy, water_interface, padding_type, gmsh_parallel)
        ],
        rtol=UNSTRUCTURED_CELL_COUNT_RTOL,
    )
    _check_mesh(mesh, parameters["output_filename"], water_interface, padding_type)


@pytest.mark.slow
def test_gmsh3d_structured_winslow_zero_iterations(tmp_path):
    """Exercise the 3-D Winslow path without requiring JIT compilation."""

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

    mesh = create_gmsh_mesh(parameters)

    assert_cell_type(mesh, "hexahedron")
    assert_cell_count(mesh, STRUCTURED_EXPECTED_CELLS[(False, None)])
    _check_mesh(mesh, parameters["output_filename"], False, None)


@pytest.mark.slow
def test_gmsh3d_structured_hyperelliptical_rejected(tmp_path):
    """Check the guard for unsupported structured padding."""

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
        create_gmsh_mesh(parameters)


@pytest.mark.slow
def test_gmsh3d_water_interface_rejects_noncanonical_axes(tmp_path):
    """Check the real water geometry axis-order validation."""

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
        create_gmsh_mesh(parameters)


@pytest.mark.slow
def test_gmsh3d_structured_rejects_non_numba_winslow(tmp_path):
    """Exercise the 3-D Winslow implementation validation."""

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
        create_gmsh_mesh(parameters)
