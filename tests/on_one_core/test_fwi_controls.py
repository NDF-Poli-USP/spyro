import numpy as np
import firedrake as fire
import pytest

import spyro


def build_acoustic_dictionary():
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5)],
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.02,
            "dt": 0.002,
            "amplitude": 1.0,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def build_elastic_dictionary():
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5)],
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "synthetic_data": {
            "type": "object",
            "density": 1.0,
            "p_wave_velocity": 2.5,
            "s_wave_velocity": 1.0,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.02,
            "dt": 0.002,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def make_elastic_controls(wave):
    V = wave.get_control_parameter_function_space()
    rho = fire.Function(V, name="density").assign(1.0)
    lmbda = fire.Function(V, name="lambda")
    mu = fire.Function(V, name="mu").assign(1.0)
    z = wave.mesh_z
    x = wave.mesh_x
    lmbda.interpolate(
        fire.conditional(
            (z + 0.5) ** 2 + (x - 0.5) ** 2 < 0.12 ** 2,
            5.0,
            4.0,
        ),
    )
    return {
        spyro.ElasticMaterialParameter.DENSITY: rho,
        spyro.ElasticMaterialParameter.LAMBDA: lmbda,
        spyro.ElasticMaterialParameter.MU: mu,
    }


def test_full_waveform_inversion_uses_composition():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())

    assert not isinstance(fwi, spyro.Wave)
    assert isinstance(fwi.wave, spyro.AcousticWave)
    assert fwi.wave_type is spyro.WaveType.ISOTROPIC_ACOUSTIC


def test_full_waveform_inversion_rejects_non_wave_instance():
    with pytest.raises(TypeError, match="wave must be an instance of Wave"):
        spyro.FullWaveformInversion(wave=object())


def test_full_waveform_inversion_rejects_non_wave_class():
    with pytest.raises(TypeError, match="wave_class must be a Wave subclass"):
        spyro.FullWaveformInversion(
            dictionary=build_acoustic_dictionary(),
            wave_class=object,
        )


def test_full_waveform_inversion_rejects_non_acoustic_wave_class():
    with pytest.raises(NotImplementedError, match="supports only acoustic"):
        spyro.FullWaveformInversion(
            dictionary=build_elastic_dictionary(),
            wave_class=spyro.IsotropicWave,
        )


def test_full_waveform_inversion_rejects_non_acoustic_wave_instance():
    wave = spyro.IsotropicWave(dictionary=build_elastic_dictionary())

    with pytest.raises(NotImplementedError, match="supports only acoustic"):
        spyro.FullWaveformInversion(wave=wave)


def test_acoustic_constant_control_is_converted_to_function():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_control(fire.Constant(2.0))

    assert isinstance(fwi.guess_control, fire.Function)
    assert isinstance(fwi.wave.get_control_parameters(), fire.Function)
    assert np.allclose(fwi.guess_control.dat.data_ro, 2.0)
    assert fwi.wave.initial_velocity_model is None
    fwi.wave._initialize_model_parameters()
    assert fwi.wave.initial_velocity_model is None


def test_acoustic_velocity_model_is_canonical_and_c_is_compatible():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.set_control_parameters(fire.Constant(2.0))

    assert wave.get_control_parameters() is wave.velocity_model
    assert wave.get_cfl_wave_speed() is wave.velocity_model
    assert "c" not in wave.__dict__
    with pytest.warns(DeprecationWarning, match="velocity_model"):
        assert wave.c is wave.velocity_model


def test_acoustic_initial_velocity_is_an_immutable_snapshot_during_fwi():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    provided_velocity = fire.Function(wave.function_space).assign(2.0)
    wave.set_initial_velocity_model(
        velocity_model_function=provided_velocity,
    )

    assert wave.velocity_model is not wave.initial_velocity_model
    assert wave.initial_velocity_model is not provided_velocity
    assert np.allclose(wave.initial_velocity_model.dat.data_ro, 2.0)

    provided_velocity.assign(5.0)
    assert np.allclose(wave.initial_velocity_model.dat.data_ro, 2.0)

    wave.velocity_model.assign(3.0)
    wave._initialize_model_parameters()

    assert np.allclose(wave.velocity_model.dat.data_ro, 3.0)
    assert np.allclose(wave.initial_velocity_model.dat.data_ro, 2.0)

    wave.set_control_parameters(fire.Constant(4.0))

    assert np.allclose(wave.velocity_model.dat.data_ro, 4.0)
    assert np.allclose(wave.initial_velocity_model.dat.data_ro, 2.0)


def test_acoustic_c_alias_updates_the_canonical_velocity_model():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())
    marker = object()

    with pytest.warns(DeprecationWarning, match="velocity_model"):
        wave.c = marker

    assert wave.velocity_model is marker


def test_elastic_controls_roundtrip_on_isotropic_wave():
    wave = spyro.IsotropicWave(dictionary=build_elastic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    controls = make_elastic_controls(wave)

    wave.set_control_parameters(controls)

    assert set(wave.get_control_parameters()) == {
        spyro.ElasticMaterialParameter.DENSITY,
        spyro.ElasticMaterialParameter.LAMBDA,
        spyro.ElasticMaterialParameter.MU,
    }
    assert wave.get_control_parameters() is not controls
    assert wave.get_cfl_wave_speed() is wave.p_wave_velocity
    assert "c" not in wave.__dict__
    assert "c_s" not in wave.__dict__
    with pytest.warns(DeprecationWarning, match="p_wave_velocity"):
        assert wave.c is wave.p_wave_velocity
    with pytest.warns(DeprecationWarning, match="s_wave_velocity"):
        assert wave.c_s is wave.s_wave_velocity


def test_elastic_constant_controls_are_converted_to_functions():
    wave = spyro.IsotropicWave(dictionary=build_elastic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})

    wave.set_control_parameters(
        {
            spyro.ElasticMaterialParameter.DENSITY: fire.Constant(1.0),
            spyro.ElasticMaterialParameter.LAMBDA: fire.Constant(4.0),
            spyro.ElasticMaterialParameter.MU: fire.Constant(1.0),
        },
    )

    controls = wave.get_control_parameters()
    assert all(isinstance(control, fire.Function) for control in controls.values())
    assert all(
        not isinstance(control, fire.Constant)
        for control in controls.values()
    )


def test_elastic_controls_reject_string_keys():
    wave = spyro.IsotropicWave(dictionary=build_elastic_dictionary())

    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    controls = make_elastic_controls(wave)

    with pytest.raises(TypeError):
        wave.set_control_parameters(
            {
                parameter.value: control
                for parameter, control in controls.items()
            },
        )
