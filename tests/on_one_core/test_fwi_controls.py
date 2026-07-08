from copy import deepcopy

import numpy as np
import firedrake as fire
import pytest

import spyro
from spyro.utils.typing import AdjointType


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


def build_elastic_lame_gradient_dictionary():
    dictionary = build_elastic_dictionary()
    dictionary["synthetic_data"] = {
        "type": "object",
        "density": 1.0,
        "lambda": 4.0,
        "mu": 1.0,
        "real_velocity_file": None,
    }
    dictionary["time_axis"]["final_time"] = 0.5
    dictionary["time_axis"]["dt"] = 0.002
    dictionary["time_axis"]["output_frequency"] = 100
    return dictionary


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


def solve_elastic_gradient_model(
    model, edge_length=0.5, implemented_adjoint=False,
    real_shot_record=None, use_vertex_only_mesh=False,
):
    wave = spyro.IsotropicWave(dictionary=deepcopy(model))
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
    if real_shot_record is not None:
        wave.real_shot_record = real_shot_record
    if implemented_adjoint:
        wave.enable_implemented_adjoint(
            adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
        )
    elif use_vertex_only_mesh:
        wave.use_vertex_only_mesh = True
    wave.forward_solve()
    return wave


def normalized_scalar_direction(function_space, seed):
    values = np.random.default_rng(seed).random(function_space.dim()) - 0.5
    direction = fire.Function(function_space, val=values)
    direction.assign(direction / fire.norm(direction))
    return direction


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


def test_elastic_ufl_derived_adjoint_uses_forward_residual_form():
    model = build_elastic_lame_gradient_dictionary()
    exact_model = deepcopy(model)
    exact_model["synthetic_data"]["density"] = 1.2

    exact = solve_elastic_gradient_model(
        exact_model, use_vertex_only_mesh=True,
    )
    guess = solve_elastic_gradient_model(
        model,
        implemented_adjoint=True,
        real_shot_record=exact.forward_solution_receivers,
    )

    functional = guess.functional_value
    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )

    expected_controls = {
        spyro.ElasticMaterialParameter.DENSITY,
        spyro.ElasticMaterialParameter.LAMBDA,
        spyro.ElasticMaterialParameter.MU,
    }
    assert guess.forward_residual_form is not None
    assert guess.forward_residual_states is not None
    assert guess.get_adjoint_source().function_space() == (
        guess.source_function.function_space()
    )
    assert np.isfinite(functional)
    assert set(gradient) == expected_controls
    assert all(isinstance(value, fire.Function) for value in gradient.values())
    assert all(np.isfinite(fire.norm(value)) for value in gradient.values())


@pytest.mark.parametrize(
    "parameter",
    [
        spyro.ElasticMaterialParameter.DENSITY,
        spyro.ElasticMaterialParameter.LAMBDA,
        spyro.ElasticMaterialParameter.MU,
    ],
)
def test_elastic_ufl_derived_adjoint_taylor_remainder(parameter):
    model = build_elastic_lame_gradient_dictionary()
    exact_model = deepcopy(model)
    exact_model["synthetic_data"][parameter.value] = (
        1.2 * exact_model["synthetic_data"][parameter.value]
    )

    exact = solve_elastic_gradient_model(
        exact_model, use_vertex_only_mesh=True,
    )
    real_record = exact.forward_solution_receivers

    guess = solve_elastic_gradient_model(
        model,
        implemented_adjoint=True,
        real_shot_record=real_record,
    )
    base_controls = {
        parameter: fire.Function(control.function_space(), name=control.name())
        for parameter, control in guess.get_control_parameters().items()
    }
    for parameter, control in guess.get_control_parameters().items():
        base_controls[parameter].assign(control)

    base_functional = guess.functional_value
    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )

    direction = normalized_scalar_direction(
        base_controls[parameter].function_space(), 5,
    )
    directional_derivative = fire.assemble(
        gradient[parameter] * direction * fire.dx(**guess.quadrature_rule)
    )

    steps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
    remainders = []
    directional_errors = []
    for step in steps:
        controls = {
            control_parameter: fire.Function(
                control.function_space(), name=control.name(),
            )
            for control_parameter, control in base_controls.items()
        }
        for control_parameter, control in base_controls.items():
            controls[control_parameter].assign(control)
        controls[parameter].assign(base_controls[parameter] + step * direction)
        guess.set_control_parameters(controls)
        guess.forward_solve()
        perturbed_functional = guess.functional_value
        finite_difference = (
            perturbed_functional - base_functional
        ) / step
        remainders.append(abs(
            perturbed_functional
            - base_functional
            - step * directional_derivative
        ))
        directional_errors.append(abs(
            (finite_difference - directional_derivative)
            / directional_derivative
        ))

    remainders = np.array(remainders)
    directional_errors = np.array(directional_errors)
    convergence_rates = np.log(remainders[:-1] / remainders[1:]) / np.log(
        steps[:-1] / steps[1:]
    )

    assert np.all(np.isfinite(remainders))
    assert np.all(convergence_rates > 1.8), convergence_rates
    assert np.all(directional_errors < 0.1)
