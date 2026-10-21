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
    hyperellipse_area,
    measure,
    physical_groups,
)


HAS_NUMBA = importlib.util.find_spec("numba") is not None

LENGTH_Z = 3000.0
LENGTH_X = 5000.0
PADDING = 1000.0
HYPER_N = 4.0

# Structured meshes are transfinite, so their cell count is exact. Unstructured
# counts may change slightly between Gmsh versions; the baselines were made
# with Gmsh 4.13.1.
UNSTRUCTURED_CELL_COUNT_RTOL = 0.02
# The hyperelliptical boundary is a polyline, so its area is slightly smaller
# than the exact hyperellipse.
HYPERELLIPSE_AREA_RTOL = 0.01

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
        "length_z": LENGTH_Z,
        "length_x": LENGTH_X,
        "output_filename": str(output_filename),
        "segy_velocity_model": str(VELOCITY_MODELS_DIR / "vs_example_2D.segy"),
        "cells_per_wavelength": 2.0,
        "source_frequency": 10.0,
        "padding_x": PADDING,
        "padding_z": PADDING,
        "hyper_n": HYPER_N,
        "hmin_segy": 0.0,
        "grade": 0.1,
        "water_search_value": 0.0,
        "vp_water": 500.0,
        "min_element_size": 150.0,
        "winslow_iterations": 100,
        "winslow_omega": 0.5,
        "h_padding": 500.0,
    }


def _check_mesh(mesh, msh_file, water_interface, padding_type) -> None:
    """Check the geometry and regions of a 2D Gmsh mesh.

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
    box_area = LENGTH_Z * LENGTH_X
    if padding_type is None:
        lower, upper = (-LENGTH_Z, 0.0), (0.0, LENGTH_X)
        total_area, area_rtol, bounds_rtol = box_area, 1e-9, 1e-9
    else:
        # The hyperellipse reaches the same extremes as the rectangle.
        lower, upper = (-LENGTH_Z - PADDING, -PADDING), (0.0, LENGTH_X + PADDING)
        if padding_type == "rectangular":
            total_area, area_rtol, bounds_rtol = (
                (LENGTH_Z + PADDING) * (LENGTH_X + 2 * PADDING), 1e-9, 1e-9,
            )
        else:
            total_area = hyperellipse_area(
                LENGTH_Z, LENGTH_X, PADDING, PADDING, HYPER_N
            )
            area_rtol, bounds_rtol = HYPERELLIPSE_AREA_RTOL, 1e-3

    assert_bounds(mesh, lower, upper, rtol=bounds_rtol)
    assert_measure(mesh, total_area, rtol=area_rtol)
    assert_no_degenerate_cells(mesh)

    groups = physical_groups(msh_file)
    core = ("WaterSurface", "SubSurface") if water_interface else ("SubSurface",)
    expected_measures = {core: box_area}
    if padding_type is not None:
        # The total was checked above; the box edges are straight, so the
        # split between model and padding is exact.
        expected_measures[("Padding",)] = measure(mesh) - box_area
    assert_cell_regions(mesh, groups[2], expected_measures)
    if water_interface:
        assert_region_above(
            mesh, groups[2]["WaterSurface"], groups[2]["SubSurface"]
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

    mesh = create_gmsh_mesh(parameters)

    assert_cell_type(mesh, "quadrilateral")
    assert_cell_count(mesh, STRUCTURED_EXPECTED_CELLS[(water_interface, padding_type)])
    _check_mesh(mesh, parameters["output_filename"], water_interface, padding_type)


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

    mesh = create_gmsh_mesh(parameters)

    assert_cell_type(mesh, "triangle")
    assert_cell_count(
        mesh,
        UNSTRUCTURED_EXPECTED_CELLS[(extend_segy, water_interface, padding_type)],
        rtol=UNSTRUCTURED_CELL_COUNT_RTOL,
    )
    _check_mesh(mesh, parameters["output_filename"], water_interface, padding_type)
