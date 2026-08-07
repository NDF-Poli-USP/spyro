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


def test_initialize_model_parameters_missing_parameters():
    synthetic_dict = {
        "type": "object",
    }
    wave = IsotropicWave(dummy_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    with pytest.raises(ValueError):
        wave.initialize_model_parameters(synthetic_data=synthetic_dict)


def test_initialize_model_parameters_lame_parameterization(monkeypatch):
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lambda": 2,
        "mu": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    calls = []
    set_material_property = wave.set_material_property

    def record_material_property(*args, **kwargs):
        calls.append(args[0])
        return set_material_property(*args, **kwargs)

    monkeypatch.setattr(wave, "set_material_property", record_material_property)
    wave.initialize_model_parameters(synthetic_data=synthetic_dict)

    assert calls == ["density", "lambda", "mu"]


def test_initialize_model_parameters_velocity_parameterization(monkeypatch):
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "p_wave_velocity": 2,
        "s_wave_velocity": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    calls = []
    set_material_property = wave.set_material_property

    def record_material_property(*args, **kwargs):
        calls.append(args[0])
        return set_material_property(*args, **kwargs)

    monkeypatch.setattr(wave, "set_material_property", record_material_property)
    wave.initialize_model_parameters(synthetic_data=synthetic_dict)

    assert calls == ["density", "p_wave_velocity", "s_wave_velocity"]


def test_initialize_model_parameters_redundant():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lmbda": 2,
        "mu": 3,
        "p_wave_velocity": 2,
        "s_wave_velocity": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    with pytest.raises(ValueError):
        wave.initialize_model_parameters(synthetic_data=synthetic_dict)


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
    with pytest.raises(Exception):
        wave.parse_boundary_conditions()


def test_elastic_file_material_parameters_notimplemented():
    synthetic_dict = {
        "type": "file",
    }
    wave = IsotropicWave(dummy_dict)
    with pytest.raises(NotImplementedError):
        wave.initialize_model_parameters(synthetic_data=synthetic_dict)


def test_initialize_model_parameters_preserves_material_properties():
    wave = IsotropicWave(dummy_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.rho = wave.set_material_property(
        "density", "scalar", constant=1.0,
    )
    wave.lmbda = wave.set_material_property(
        "lambda", "scalar", constant=2.0,
    )
    wave.mu = wave.set_material_property(
        "mu", "scalar", constant=3.0,
    )

    wave.initialize_model_parameters()
    parameters = (wave.rho, wave.lmbda, wave.mu, wave.c, wave.c_s)

    wave.initialize_model_parameters()

    assert all(
        before is after
        for before, after in zip(
            parameters,
            (wave.rho, wave.lmbda, wave.mu, wave.c, wave.c_s),
        )
    )


def test_initialize_model_parameters_recomputes_after_change():
    wave = IsotropicWave(dummy_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.rho = wave.set_material_property("density", "scalar", constant=1.0)
    wave.lmbda = wave.set_material_property("lambda", "scalar", constant=2.0)
    wave.mu = wave.set_material_property("mu", "scalar", constant=3.0)

    wave.initialize_model_parameters()
    c_before, c_s_before = wave.c, wave.c_s

    wave.mu = wave.set_material_property("mu", "scalar", constant=5.0)
    wave.initialize_model_parameters()

    assert wave.c is not c_before
    assert wave.c_s is not c_s_before
