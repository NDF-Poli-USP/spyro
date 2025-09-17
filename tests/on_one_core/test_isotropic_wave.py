import firedrake as fire
import numpy as np
import pytest

from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave

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
    "mesh": {},
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
        wave.initialize_model_parameters_from_object(synthetic_dict)


def test_initialize_model_parameters_from_object_first_option():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lambda": 2,
        "mu": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.initialize_model_parameters_from_object(synthetic_dict)


def test_initialize_model_parameters_from_object_second_option():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "p_wave_velocity": 2,
        "s_wave_velocity": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.initialize_model_parameters_from_object(synthetic_dict)


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
        wave.initialize_model_parameters_from_object(synthetic_dict)


def test_parse_boundary_conditions():
    d = dummy_dict.copy()
    d["mesh"] = {
        "Lz": 1.0,
        "Lx": 1.0,
        "Ly": 1.0,
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
        "Lz": 1.0,
        "Lx": 1.0,
        "Ly": 1.0,
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


def test_initialize_model_parameters_from_file_notimplemented():
    synthetic_dict = {
        "type": "file",
    }
    wave = IsotropicWave(dummy_dict)
    with pytest.raises(NotImplementedError) as e:  # noqa: F841
        wave.initialize_model_parameters_from_file(synthetic_dict)


if __name__ == "__main__":
    test_initialize_model_parameters_from_object_missing_parameters()
    test_initialize_model_parameters_from_object_first_option()
    test_initialize_model_parameters_from_object_second_option()
    test_initialize_model_parameters_from_object_redundant()
    test_parse_boundary_conditions()
    test_parse_boundary_conditions_exception()
    test_initialize_model_parameters_from_file_notimplemented()
