from types import SimpleNamespace
import pytest
from spyro.tools import input_models


def _base_variables(method="mass_lumped_triangle"):
    return {
        "method": method,
        "degree": 3,
        "dimension": 2,
        "length_z": 5.0,
        "length_x": 7.5,
        "length_y": 0.0,
        "cells_per_wavelength": 2.4,
        "pad": 0.8,
        "source_locations": [(-1.0, 2.0)],
        "frequency": 4.0,
        "receiver_locations": [(-1.0, 2.5), (-1.0, 3.0)],
        "final_time": 2.0,
        "dt": 0.001,
    }


def _base_meshing_obj(**overrides):
    defaults = {
        "dimension": 2,
        "velocity_profile_type": "homogeneous",
        "minimum_velocity": 1.5,
        "source_frequency": 5.0,
        "cpw_initial": 2.2,
        "FEM_method_to_evaluate": "mass_lumped_triangle",
        "desired_degree": 4,
        "reduced_obj_for_testing": False,
        "parameters_dictionary": {"length_z": 10.0, "length_x": 20.0},
        "velocity_model_file_name": "velocity.segy",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    "method, expected_mesh_type",
    [
        ("mass_lumped_triangle", "SeismicMesh"),
        ("spectral_quadrilateral", "firedrake_mesh"),
    ],
)
def test_set_mesh_type(method, expected_mesh_type):
    assert input_models.set_mesh_type(method) == expected_mesh_type


def test_set_mesh_type_raises_for_invalid_method():
    with pytest.raises(ValueError, match="Method is not"):
        input_models.set_mesh_type("unsupported_method")


def test_build_on_top_of_base_dictionary_populates_expected_sections():
    model = input_models.build_on_top_of_base_dictionary(_base_variables())

    assert model["options"]["automatic_adjoint"] is False
    assert model["mesh"]["mesh_type"] == "SeismicMesh"
    assert model["mesh"]["length_x"] == 7.5
    assert model["absorving_boundary_conditions"]["pad_length"] == 0.8
    assert model["acquisition"]["source_type"] == "ricker"
    assert model["time_axis"]["dt"] == 0.001
    assert model["visualization"]["forward_output"] is True


def test_create_initial_model_for_meshing_parameter_dispatches_2d(monkeypatch):
    expected = {"kind": "2d", "dimension": 2}

    def fake_create_2d(obj):
        return {"kind": "2d", "dimension": obj.dimension}

    monkeypatch.setattr(
        input_models,
        "create_initial_model_for_meshing_parameter_2D",
        fake_create_2d,
    )

    obj = _base_meshing_obj(dimension=2)
    got = input_models.create_initial_model_for_meshing_parameter(obj)

    assert got == expected


def test_create_initial_model_for_meshing_parameter_dispatches_3d(monkeypatch):
    expected = {"kind": "3d"}

    monkeypatch.setattr(
        input_models,
        "create_initial_model_for_meshing_parameter_3D",
        lambda _: expected,
    )

    obj = _base_meshing_obj(dimension=3)
    assert input_models.create_initial_model_for_meshing_parameter(obj) == expected


def test_create_initial_model_for_meshing_parameter_rejects_invalid_dimension():
    obj = _base_meshing_obj(dimension=4)
    with pytest.raises(ValueError, match="Dimension is not 2 or 3"):
        input_models.create_initial_model_for_meshing_parameter(obj)


def test_create_initial_model_for_meshing_parameter_2d_dispatches_homogeneous(
    monkeypatch,
):
    expected = {"profile": "homogeneous"}
    monkeypatch.setattr(
        input_models,
        "create_initial_model_for_meshing_parameter_2D_homogeneous",
        lambda _: expected,
    )

    obj = _base_meshing_obj(velocity_profile_type="homogeneous")
    assert input_models.create_initial_model_for_meshing_parameter_2D(obj) == expected


def test_create_initial_model_for_meshing_parameter_2d_dispatches_heterogeneous(
    monkeypatch,
):
    expected = {"profile": "heterogeneous"}
    monkeypatch.setattr(
        input_models,
        "create_initial_model_for_meshing_parameter_2D_heterogeneous",
        lambda _: expected,
    )

    obj = _base_meshing_obj(velocity_profile_type="heterogeneous")
    assert input_models.create_initial_model_for_meshing_parameter_2D(obj) == expected


def test_create_initial_model_for_meshing_parameter_2d_rejects_invalid_profile():
    obj = _base_meshing_obj(velocity_profile_type="layered")
    with pytest.raises(ValueError, match="Velocity profile type is not"):
        input_models.create_initial_model_for_meshing_parameter_2D(obj)


def test_create_initial_model_for_meshing_parameter_2d_heterogeneous_sets_core_fields():
    obj = _base_meshing_obj(
        velocity_profile_type="heterogeneous",
        minimum_velocity=1.6,
        source_frequency=8.0,
        cpw_initial=2.8,
        FEM_method_to_evaluate="spectral_quadrilateral",
        desired_degree=5,
        parameters_dictionary={"length_z": 12.0, "length_x": 30.0},
        velocity_model_file_name="my_model.segy",
    )

    model = input_models.create_initial_model_for_meshing_parameter_2D_heterogeneous(
        obj
    )

    assert model["options"]["method"] == "spectral_quadrilateral"
    assert model["options"]["degree"] == 5
    assert model["mesh"]["length_z"] == 12.0
    assert model["mesh"]["length_x"] == 30.0
    assert model["mesh"]["cells_per_wavelength"] == 2.8
    assert model["absorving_boundary_conditions"]["pad_length"] == pytest.approx(0.2)
    assert model["synthetic_data"]["real_velocity_file"] == "my_model.segy"

    receivers = model["acquisition"]["receiver_locations"]
    assert len(receivers) == 500
    assert tuple(receivers[0]) == pytest.approx((-0.3, 5.0))
    assert tuple(receivers[-1]) == pytest.approx((-0.3, 13.0))


@pytest.mark.parametrize("reduced, expected_receivers", [(True, 4), (False, 36)])
def test_create_initial_model_for_meshing_parameter_2d_homogeneous_reduced_flag(
    reduced, expected_receivers
):
    obj = _base_meshing_obj(
        velocity_profile_type="homogeneous",
        minimum_velocity=2.0,
        source_frequency=4.0,
        cpw_initial=3.1,
        FEM_method_to_evaluate="mass_lumped_triangle",
        desired_degree=3,
        reduced_obj_for_testing=reduced,
    )

    model = input_models.create_initial_model_for_meshing_parameter_2D_homogeneous(obj)

    lbda = obj.minimum_velocity / obj.source_frequency
    assert model["mesh"]["length_z"] == pytest.approx(40 * lbda)
    assert model["mesh"]["length_x"] == pytest.approx(30 * lbda)
    assert model["absorving_boundary_conditions"]["pad_length"] == pytest.approx(lbda)
    assert model["time_axis"]["final_time"] == pytest.approx(
        20 * (1.0 / obj.source_frequency)
    )
    assert len(model["acquisition"]["receiver_locations"]) == expected_receivers


def test_create_initial_model_for_meshing_parameter_2d_homogeneous_warns_for_large_velocity():
    obj = _base_meshing_obj(
        velocity_profile_type="homogeneous",
        minimum_velocity=600.0,
        source_frequency=6.0,
    )

    with pytest.warns(UserWarning, match="Velocity in meters per second"):
        input_models.create_initial_model_for_meshing_parameter_2D_homogeneous(obj)


@pytest.mark.parametrize("profile", ["homogeneous", "heterogeneous"])
def test_create_initial_model_for_meshing_parameter_3d_not_implemented(profile):
    obj = _base_meshing_obj(dimension=3, velocity_profile_type=profile)
    with pytest.raises(NotImplementedError, match="Not yet implemented"):
        input_models.create_initial_model_for_meshing_parameter_3D(obj)


def test_create_initial_model_for_meshing_parameter_3d_rejects_invalid_profile():
    obj = _base_meshing_obj(dimension=3, velocity_profile_type="layered")
    with pytest.raises(ValueError, match="Velocity profile type is not"):
        input_models.create_initial_model_for_meshing_parameter_3D(obj)
