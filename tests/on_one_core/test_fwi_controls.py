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
