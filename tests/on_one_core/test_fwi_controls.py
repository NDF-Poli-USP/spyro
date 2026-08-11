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


def initialize_elastic_wave(material_parameters, control_parameters=None):
    """Build and initialize an elastic wave with optional control selection."""
    dictionary = build_elastic_dictionary()
    dictionary["synthetic_data"] = {
        "type": "object",
        **material_parameters,
        "real_velocity_file": None,
    }
    if control_parameters is not None:
        dictionary["synthetic_data"]["control_parameters"] = control_parameters

    wave = spyro.IsotropicWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave._initialize_model_parameters()
    return wave


def interpolate_material_parameter(wave, parameter):
    """Interpolate a material expression for value comparisons."""
    return fire.Function(
        wave.get_control_parameter_function_space(),
    ).interpolate(parameter)


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


@pytest.mark.parametrize(
    (
        "material_parameters",
        "control_parameters",
        "expected_parameters",
        "expected_values",
        "expected_parameterization",
    ),
    [
        pytest.param(
            {"density": 1.0, "p_wave_velocity": 2.5, "s_wave_velocity": 1.0},
            None,
            (
                spyro.ElasticMaterialParameter.DENSITY,
                spyro.ElasticMaterialParameter.P_WAVE_VELOCITY,
                spyro.ElasticMaterialParameter.S_WAVE_VELOCITY,
            ),
            (1.0, 2.5, 1.0),
            spyro.ElasticMaterialParameterization.VELOCITY,
            id="default-equation-parameters",
        ),
        pytest.param(
            {"density": 1.0, "p_wave_velocity": 2.5, "s_wave_velocity": 1.0},
            ["c_s"],
            (spyro.ElasticMaterialParameter.S_WAVE_VELOCITY,),
            (1.0,),
            spyro.ElasticMaterialParameterization.VELOCITY,
            id="only-cs",
        ),
        pytest.param(
            {"density": 1.0, "lambda": 4.25, "mu": 1.0},
            ["c"],
            (spyro.ElasticMaterialParameter.P_WAVE_VELOCITY,),
            (2.5,),
            spyro.ElasticMaterialParameterization.VELOCITY,
            id="only-c-from-lame-equation",
        ),
        pytest.param(
            {"density": 1.0, "p_wave_velocity": 2.5, "s_wave_velocity": 1.0},
            ["lambda", "mu"],
            (
                spyro.ElasticMaterialParameter.LAMBDA,
                spyro.ElasticMaterialParameter.MU,
            ),
            (4.25, 1.0),
            spyro.ElasticMaterialParameterization.LAME,
            id="lambda-mu-from-velocity-equation",
        ),
        pytest.param(
            {"density": 1.0, "lambda": 4.25, "mu": 1.0},
            ["c", "c_s"],
            (
                spyro.ElasticMaterialParameter.P_WAVE_VELOCITY,
                spyro.ElasticMaterialParameter.S_WAVE_VELOCITY,
            ),
            (2.5, 1.0),
            spyro.ElasticMaterialParameterization.VELOCITY,
            id="c-cs-from-lame-equation",
        ),
    ],
)
def test_elastic_control_selection_is_independent_from_equation_parameters(
    material_parameters,
    control_parameters,
    expected_parameters,
    expected_values,
    expected_parameterization,
):
    wave = initialize_elastic_wave(material_parameters, control_parameters)

    controls = wave.get_control_parameters()
    assert tuple(controls) == expected_parameters
    assert wave._control_parameterization is expected_parameterization
    assert all(isinstance(control, fire.Function) for control in controls.values())
    for control, expected_value in zip(controls.values(), expected_values):
        assert np.allclose(control.dat.data_ro, expected_value)


@pytest.mark.parametrize(
    ("control_parameters", "exception", "message"),
    [
        pytest.param("c_s", TypeError, "list or tuple", id="not-a-list"),
        pytest.param([], ValueError, "At least one", id="empty"),
        pytest.param(["unknown"], ValueError, "Unknown", id="unknown"),
        pytest.param(
            ["lambda", "c_s"],
            ValueError,
            "cannot mix",
            id="mixed-parameterizations",
        ),
        pytest.param(
            ["c", "p_wave_velocity"],
            ValueError,
            "duplicates",
            id="aliases-for-the-same-parameter",
        ),
    ],
)
def test_elastic_control_selection_rejects_invalid_input(
    control_parameters,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        initialize_elastic_wave(
            {
                "density": 1.0,
                "p_wave_velocity": 2.5,
                "s_wave_velocity": 1.0,
            },
            control_parameters,
        )


def test_set_elastic_control_parameters_updates_only_selected_subset():
    wave = initialize_elastic_wave(
        {"density": 1.0, "p_wave_velocity": 2.5, "s_wave_velocity": 1.0},
    )

    wave.set_control_parameters({
        spyro.ElasticMaterialParameter.S_WAVE_VELOCITY: fire.Constant(1.2),
    })

    controls = wave.get_control_parameters()
    assert tuple(controls) == (
        spyro.ElasticMaterialParameter.S_WAVE_VELOCITY,
    )
    assert np.allclose(wave.rho.dat.data_ro, 1.0)
    assert np.allclose(wave.c.dat.data_ro, 2.5)
    assert np.allclose(wave.c_s.dat.data_ro, 1.2)
    assert np.allclose(
        interpolate_material_parameter(wave, wave.mu).dat.data_ro,
        1.2 ** 2,
    )
    assert np.allclose(
        interpolate_material_parameter(wave, wave.lmbda).dat.data_ro,
        2.5 ** 2 - 2 * 1.2 ** 2,
    )


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
