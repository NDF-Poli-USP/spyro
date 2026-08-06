import firedrake as fire
import numpy as np
import pytest

from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave
from spyro.utils.typing import ElasticMaterialParameterization

dummy_dict = {
    "options": {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 3,
        "dimension": 3,
    },
    "time_axis": {
        "final_time": 1,
        "dt": 0.001,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    },
    "mesh": {
        "length_z": 1.0,
        "length_x": 1.0,
        "length_y": 1.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    },
    "acquisition": {
        "receiver_locations": [],
        "source_type": "ricker",
        "source_locations": [(0, 0, 0)],
        "frequency": 5.0,
    },
}


def test_initialize_model_parameters_from_object_missing_parameters():
    synthetic_dict = {
        "type": "object",
    }
    wave = IsotropicWave(dummy_dict)
    with pytest.raises(Exception) as e:  # noqa: F841
        wave.declare_model_parameters(synthetic_dict)


def test_initialize_model_parameters_from_object_first_option():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lambda": 2,
        "mu": 3,
    }
    wave = IsotropicWave(dummy_dict)
    # Phase A only: validating a declaration needs no mesh or function space.
    wave.declare_model_parameters(synthetic_dict)

    assert wave._control_parameterization is ElasticMaterialParameterization.LAME


def test_initialize_model_parameters_from_object_second_option():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "p_wave_velocity": 2,
        "s_wave_velocity": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.declare_model_parameters(synthetic_dict)

    assert wave._control_parameterization is \
        ElasticMaterialParameterization.VELOCITY


def test_declare_model_parameters_accepts_zero_valued_parameter():
    """A zero-valued parameter is declared, even though bool(0) is False."""
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lambda": 0.0,
        "mu": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.declare_model_parameters(synthetic_dict)

    assert wave._control_parameterization is ElasticMaterialParameterization.LAME


def test_initialize_model_parameters_from_object_redundant():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lmbda": 2,
        "mu": 3,
        "p_wave_velocity": 2,
        "s_wave_velocity": 3,
    }
    wave = IsotropicWave(dummy_dict)
    with pytest.raises(Exception) as e:  # noqa: F841
        wave.declare_model_parameters(synthetic_dict)


def test_parse_boundary_conditions():
    d = dummy_dict.copy()
    d["mesh"] = {
        "length_z": 1.0,
        "length_x": 1.0,
        "length_y": 1.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    d["boundary_conditions"] = [
        ("u", 1, fire.Constant((1, 1, 1))),  # x == 0:  1 (z in spyro)
        ("uz", 2, fire.Constant(2)),         # x == Lx: 2 (z in spyro)
        ("ux", 3, fire.Constant(3)),         # y == 0:  3 (x in spyro)
        ("uy", 4, fire.Constant(4)),         # y == Ly: 4 (x in spyro)
    ]
    wave = IsotropicWave(d)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2, "periodic": True})
    wave.parse_boundary_conditions()
    u = fire.Function(wave.function_space)
    for bc in wave.bcs:
        bc.apply(u)

    assert np.allclose([1, 1, 1], u.at(0.0, 0.5, 0.5))
    assert np.allclose([2, 0, 0], u.at(-1.0, 0.5, 0.5))
    assert np.allclose([0, 3, 0], u.at(-0.5, 0.0, 0.5))
    assert np.allclose([0, 0, 4], u.at(-0.5, 1.0, 0.5))


def test_parse_boundary_conditions_exception():
    d = dummy_dict.copy()
    d["mesh"] = {
        "length_z": 1.0,
        "length_x": 1.0,
        "length_y": 1.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    d["boundary_conditions"] = [
        ("?", 2, fire.Constant(2)),
    ]
    wave = IsotropicWave(d)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2, "periodic": True})
    with pytest.raises(Exception) as e:  # noqa: F841
        wave.parse_boundary_conditions()


# NOTE: the former test_initialize_model_parameters_from_file_notimplemented
# covered initialize_model_parameters_from_file, which no longer exists: a file
# is just one more material-parameter source resolved by set_material_property.
# That path is now covered by test_matprop.py::test_fromfile_mat_prop.


if __name__ == "__main__":
    test_initialize_model_parameters_from_object_missing_parameters()
    test_initialize_model_parameters_from_object_first_option()
    test_initialize_model_parameters_from_object_second_option()
    test_declare_model_parameters_accepts_zero_valued_parameter()
    test_initialize_model_parameters_from_object_redundant()
    test_parse_boundary_conditions()
    test_parse_boundary_conditions_exception()
