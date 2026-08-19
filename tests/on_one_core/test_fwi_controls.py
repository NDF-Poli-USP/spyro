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


ELASTIC_PARAMETERS = {
    "density",
    "lambda",
    "mu",
    "p_wave_velocity",
    "s_wave_velocity",
}


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


# Physical parameters belong to the wave equation.


def test_acoustic_wave_declares_its_physical_parameters():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.set_initial_velocity_model(constant=2.0)

    parameters = wave.initialize_physical_parameters()

    assert set(parameters) == {"p_wave_velocity"}
    assert parameters["p_wave_velocity"] is wave.c


def test_isotropic_wave_declares_its_physical_parameters():
    wave = build_elastic_wave()

    assert set(wave.physical_parameters) == ELASTIC_PARAMETERS


def test_physical_parameters_compare_as_a_set_of_names():
    wave = build_elastic_wave()

    assert {"density", "mu"} <= wave.physical_parameters
    assert wave.physical_parameters.issuperset({"p_wave_velocity"})
    assert "porosity" not in wave.physical_parameters


def test_physical_parameters_raise_before_initialization():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())

    with pytest.raises(ValueError, match="Physical parameters have not been set"):
        wave.physical_parameters


def test_wave_does_not_know_about_control_parameters():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())

    assert not hasattr(wave, "get_control_parameters")
    assert not hasattr(wave, "set_control_parameters")
    assert not hasattr(wave, "control_parameters")


def test_updating_a_physical_parameter_writes_into_the_field():
    wave = build_elastic_wave()
    density = wave.physical_parameters["density"]

    wave.physical_parameters.update("density", fire.Constant(2.0))

    # Writing into the field is what keeps the assembled forms, and the
    # parameters computed from this one, valid without being rebuilt.
    assert wave.physical_parameters["density"] is density
    assert wave.rho is density
    assert np.allclose(density.dat.data_ro, 2.0)


def test_dependent_physical_parameters_cannot_be_set_on_their_own():
    # The medium is declared with density and the two wave speeds, so the Lame
    # parameters are expressions of those.
    wave = build_elastic_wave()

    with pytest.raises(TypeError, match="computed from"):
        wave.physical_parameters.update("lambda", fire.Constant(4.0))


def test_unknown_physical_parameter_is_rejected():
    wave = build_elastic_wave()

    with pytest.raises(KeyError, match="porosity"):
        wave.physical_parameters.update("porosity", 1.0)


def test_physical_parameters_accept_enum_names():
    wave = build_elastic_wave()

    assert (
        wave.physical_parameters[spyro.ElasticMaterialParameter.DENSITY]
        is wave.rho
    )


# Control parameters belong to the inversion.


def test_control_parameters_default_to_the_acoustic_velocity():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())

    assert fwi.control_parameters == {"p_wave_velocity"}


def test_control_parameters_are_a_subset_of_the_physical_ones():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))

    assert fwi.control_parameters <= fwi.wave.physical_parameters


def test_control_parameters_reject_parameters_the_wave_does_not_model():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())

    with pytest.raises(ValueError, match="subset of the physical parameters"):
        fwi.control_parameters = ["density"]


def test_control_parameters_reject_an_empty_set():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())

    with pytest.raises(ValueError, match="at least one physical parameter"):
        fwi.control_parameters = []


def test_acoustic_constant_control_is_converted_to_function():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_control(fire.Constant(2.0))

    control = fwi.guess_control["p_wave_velocity"]
    assert isinstance(control, fire.Function)
    assert np.allclose(control.dat.data_ro, 2.0)


def test_guess_control_updates_the_wave_physical_parameter():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))
    velocity = fwi.wave.physical_parameters["p_wave_velocity"]

    fwi.set_guess_control(fire.Constant(3.0))

    assert fwi.wave.physical_parameters["p_wave_velocity"] is velocity
    assert np.allclose(velocity.dat.data_ro, 3.0)


def test_guess_control_is_independent_of_the_wave_field():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))

    fwi.wave.physical_parameters["p_wave_velocity"].assign(9.0)

    assert np.allclose(fwi.guess_control["p_wave_velocity"].dat.data_ro, 2.0)


def test_guess_control_accepts_a_mapping_keyed_by_parameter_name():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_control({"p_wave_velocity": fire.Constant(2.5)})

    assert np.allclose(
        fwi.wave.physical_parameters["p_wave_velocity"].dat.data_ro,
        2.5,
    )


def test_guess_control_rejects_uncontrolled_parameters():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    with pytest.raises(ValueError, match="not controlled by this inversion"):
        fwi.set_guess_control({"density": fire.Constant(1.0)})


def test_velocity_model_setters_capture_the_control():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_velocity_model(constant=2.0)

    assert set(fwi.guess_control) == {"p_wave_velocity"}
    assert np.allclose(fwi.guess_control["p_wave_velocity"].dat.data_ro, 2.0)


def test_misfit_without_a_configured_control_raises():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    with pytest.raises(ValueError, match="No guess control parameter"):
        fwi.calculate_misfit()
