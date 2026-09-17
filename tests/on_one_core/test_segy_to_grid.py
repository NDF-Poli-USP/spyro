import numpy as np
import pytest
import firedrake as fire
from spyro.meshing import AutomaticMesh, MeshingParameters
from spyro.io.segy_io import (
    create_grid_dictionary_from_segy,
    create_segy_from_grid,
    read_segy_velocity_model,
)
from spyro.utils.velocity_to_grid import scalar_conditional_to_grid


def make_off_center_conditional(mesh_z, mesh_x):
    """Create an off-center circular Firedrake conditional.

    Parameters
    ----------
    mesh_z : firedrake.ufl.Expr
        Vertical coordinate expression from a Firedrake mesh.
    mesh_x : firedrake.ufl.Expr
        Horizontal coordinate expression from a Firedrake mesh.

    Returns
    -------
    firedrake.ufl.Expr
        Conditional expression with a circular inclusion shifted away from
        the mesh center.
    """
    outside_vp = 1.0
    circle_vp = 2.0
    r_c = 0.45
    center_z = -1.25
    center_x = 1.35
    return fire.conditional(
        (mesh_z - center_z) ** 2 + (mesh_x - center_x) ** 2 < r_c**2,
        circle_vp,
        outside_vp,
    )


def test_create_grid_dictionary_from_segy(tmp_path):
    """Verify that a SEG-Y file is converted back into a grid dictionary.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for test artifacts.

    Returns
    -------
    None
    """
    velocity = np.arange(12, dtype=np.float32).reshape(3, 4)
    segy_file = tmp_path / "model.segy"

    create_segy_from_grid(velocity, str(segy_file))

    grid_velocity_data = create_grid_dictionary_from_segy(
        str(segy_file),
        length_z=2.0,
        length_x=3.0,
    )
    expected_vp_values, nz, nx = read_segy_velocity_model(str(segy_file))

    assert np.array_equal(grid_velocity_data["vp_values"], expected_vp_values)
    assert grid_velocity_data["length_z"] == 2.0
    assert grid_velocity_data["length_x"] == 3.0
    assert grid_velocity_data["abc_pad_length"] == 0.0
    assert grid_velocity_data["grid_spacing_z"] == pytest.approx(2.0 / (nz - 1))
    assert grid_velocity_data["grid_spacing_x"] == pytest.approx(3.0 / (nx - 1))
    if np.isclose(grid_velocity_data["grid_spacing_z"], grid_velocity_data["grid_spacing_x"]):
        assert grid_velocity_data["grid_spacing"] == pytest.approx(grid_velocity_data["grid_spacing_z"])
    else:
        assert grid_velocity_data["grid_spacing"] is None


def test_create_grid_dictionary_from_segy_from_conditional(tmp_path):
    """Verify SEG-Y round-tripping from an off-center Firedrake conditional.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for test artifacts.

    Returns
    -------
    None
    """
    grid_spacing = 0.02
    dictionary = {
        "length_z": 2.0,
        "length_x": 2.0,
        "length_y": 0.0,
        "mesh_type": "firedrake_mesh",
        "edge_length": grid_spacing,
        "dimension": 2,
    }

    mesh_params = MeshingParameters(input_mesh_dictionary=dictionary)
    mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)
    mesh = mesh_generator.create_mesh()

    mesh_z, mesh_x = fire.SpatialCoordinate(mesh)
    conditional = make_off_center_conditional(mesh_z, mesh_x)

    grid_velocity_data = scalar_conditional_to_grid(
        conditional=conditional,
        domain_dimensions=(2.0, 2.0),
        grid_spacing=grid_spacing,
    )

    vp_values = grid_velocity_data["vp_values"]
    assert not np.array_equal(vp_values, np.flipud(vp_values))
    assert not np.array_equal(vp_values, np.fliplr(vp_values))

    segy_file = tmp_path / "conditional.segy"
    create_segy_from_grid(vp_values, str(segy_file), rotate=True)

    segy_grid_data = create_grid_dictionary_from_segy(
        str(segy_file),
        length_z=2.0,
        length_x=2.0,
    )
    expected_vp_values, nz, nx = read_segy_velocity_model(str(segy_file))

    assert np.array_equal(segy_grid_data["vp_values"], expected_vp_values)
    assert np.allclose(segy_grid_data["vp_values"], vp_values)
    assert segy_grid_data["length_z"] == 2.0
    assert segy_grid_data["length_x"] == 2.0
    assert segy_grid_data["abc_pad_length"] == 0.0
    assert segy_grid_data["grid_spacing_z"] == pytest.approx(2.0 / (nz - 1))
    assert segy_grid_data["grid_spacing_x"] == pytest.approx(2.0 / (nx - 1))
    if np.isclose(segy_grid_data["grid_spacing_z"], segy_grid_data["grid_spacing_x"]):
        assert segy_grid_data["grid_spacing"] == pytest.approx(segy_grid_data["grid_spacing_z"])
    else:
        assert segy_grid_data["grid_spacing"] is None
