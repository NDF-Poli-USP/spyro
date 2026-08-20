import numpy as np
import firedrake as fire
import pytest

import spyro
from spyro.utils.typing import (AcousticMaterialParameter,
                                ElasticMaterialParameter)


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


VELOCITY = AcousticMaterialParameter.P_WAVE_VELOCITY


def build_elastic_wave():
    wave = spyro.IsotropicWave(dictionary=build_elastic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.initialize_physical_parameters()
    return wave


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


# Physical parameters belong to the wave equation.


def test_acoustic_wave_declares_its_physical_parameters():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.set_initial_velocity_model(constant=2.0)

    parameters = wave.initialize_physical_parameters()

    assert set(parameters) == {VELOCITY}
    assert parameters[VELOCITY] is wave.c


def test_isotropic_wave_declares_its_physical_parameters():
    wave = build_elastic_wave()

    assert set(wave.physical_parameters) == set(ElasticMaterialParameter)


def test_physical_parameters_compare_as_a_set_of_names():
    wave = build_elastic_wave()

    assert {
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.MU,
    } <= wave.physical_parameters
    assert wave.physical_parameters.issuperset(
        {ElasticMaterialParameter.S_WAVE_VELOCITY},
    )
    # An acoustic medium names its wave speed with its own enum, so the
    # elastic solver does not model that parameter.
    assert VELOCITY not in wave.physical_parameters


def test_physical_parameters_raise_before_initialization():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())

    with pytest.raises(ValueError, match="Physical parameters have not been set"):
        wave.physical_parameters


def test_updating_a_physical_parameter_writes_into_the_field():
    wave = build_elastic_wave()
    density = wave.physical_parameters[ElasticMaterialParameter.DENSITY]

    wave.physical_parameters.update(
        ElasticMaterialParameter.DENSITY, fire.Constant(2.0),
    )

    # Writing into the field is what keeps the assembled forms, and the
    # parameters computed from this one, valid without being rebuilt.
    assert wave.physical_parameters[ElasticMaterialParameter.DENSITY] is density
    assert wave.rho is density
    assert np.allclose(density.dat.data_ro, 2.0)


def test_unknown_physical_parameter_is_rejected():
    wave = build_elastic_wave()

    with pytest.raises(KeyError, match="p_wave_velocity"):
        wave.physical_parameters.update(VELOCITY, 1.0)


def test_physical_parameters_are_keyed_by_material_parameter_enums():
    wave = build_elastic_wave()

    assert (
        wave.physical_parameters[spyro.ElasticMaterialParameter.DENSITY]
        is wave.rho
    )


# Control parameters belong to the inversion.


def test_control_parameters_are_a_subset_of_the_physical_ones():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))

    # Which parameters are inverted for is recorded by the keys of the control
    # dictionary, and nowhere else.
    assert set(fwi._control_parameters) <= fwi.wave.physical_parameters


def test_acoustic_constant_control_is_converted_to_function():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_control(fire.Constant(2.0))

    control = fwi.control_parameters
    assert isinstance(control, fire.Function)
    assert np.allclose(control.dat.data_ro, 2.0)


def test_control_parameters_update_the_wave_physical_parameter():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))
    velocity = fwi.wave.physical_parameters[VELOCITY]

    fwi.set_guess_control(fire.Constant(3.0))

    assert fwi.wave.physical_parameters[VELOCITY] is velocity
    assert np.allclose(velocity.dat.data_ro, 3.0)


def test_control_parameters_are_independent_of_the_wave_field():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))

    fwi.wave.physical_parameters[VELOCITY].assign(9.0)

    assert np.allclose(fwi.control_parameters.dat.data_ro, 2.0)


def test_control_parameters_accept_a_mapping_keyed_by_parameter_name():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_control({VELOCITY: fire.Constant(2.5)})

    assert np.allclose(
        fwi.wave.physical_parameters[VELOCITY].dat.data_ro,
        2.5,
    )


def test_control_parameters_reject_uncontrolled_parameters():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    with pytest.raises(ValueError, match="not controlled by this inversion"):
        fwi.set_guess_control(
            {ElasticMaterialParameter.DENSITY: fire.Constant(1.0)},
        )


def test_velocity_model_setters_capture_the_control():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_velocity_model(constant=2.0)

    assert isinstance(fwi.control_parameters, fire.Function)
    assert np.allclose(fwi.control_parameters.dat.data_ro, 2.0)


def test_misfit_without_a_configured_control_raises():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    with pytest.raises(ValueError, match="No guess control parameter"):
        fwi.calculate_misfit()
