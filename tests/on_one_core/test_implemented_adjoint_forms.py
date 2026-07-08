from copy import deepcopy

import firedrake as fire
import numpy as np
import pytest
import spyro

from spyro.utils.typing import AdjointType, ImplementedAdjointDerivation


def _small_acoustic_model():
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
            "source_locations": [(-0.2, 0.5)],
            "frequency": 5.0,
            "delay": 1.0,
            "receiver_locations": [(-0.2, 0.6)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.5,
            "dt": 0.001,
            "amplitude": 1.0,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def _taylor_acoustic_model():
    model = _small_acoustic_model()
    model["time_axis"]["final_time"] = 0.5
    model["time_axis"]["dt"] = 0.005
    model["time_axis"]["output_frequency"] = 100
    return model


def _small_acoustic_pml_model():
    model = _small_acoustic_model()
    model["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }
    return model


def _small_elastic_model():
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
            "lambda": 4.0,
            "mu": 1.0,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.5,
            "dt": 0.002,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def _solve_acoustic(
    model, velocity, edge_length=0.5, implemented_adjoint=False,
    real_shot_record=None,
):
    wave = spyro.AcousticWave(dictionary=deepcopy(model))
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
    if isinstance(velocity, (float, int)):
        wave.set_initial_velocity_model(constant=velocity)
    else:
        wave.set_control_parameters(velocity)
    if real_shot_record is not None:
        wave.real_shot_record = real_shot_record
    if implemented_adjoint:
        wave.enable_implemented_adjoint()
    wave.forward_solve()
    return wave


def _solve_elastic(
    model, edge_length=0.5, implemented_adjoint=False,
    real_shot_record=None, use_vertex_only_mesh=False,
):
    wave = spyro.IsotropicWave(dictionary=deepcopy(model))
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
    if real_shot_record is not None:
        wave.real_shot_record = real_shot_record
    if implemented_adjoint:
        wave.enable_implemented_adjoint()
    elif use_vertex_only_mesh:
        wave.use_vertex_only_mesh = True
    wave.forward_solve()
    return wave


def test_implemented_adjoint_derivation_is_separate_from_adjoint_type():
    assert AdjointType.IMPLEMENTED_ADJOINT is AdjointType.IMPLEMENTED_ADJOINT
    assert (
        ImplementedAdjointDerivation.UFL_DIFFERENTIATION
        is not ImplementedAdjointDerivation.HAND_DERIVED
    )


def test_acoustic_implemented_adjoint_uses_forward_residual_form():
    model = _small_acoustic_model()

    exact = spyro.AcousticWave(dictionary=deepcopy(model))
    exact.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    exact.set_initial_velocity_model(constant=1.5)
    exact.forward_solve()

    guess = spyro.AcousticWave(dictionary=deepcopy(model))
    guess.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    guess.set_initial_velocity_model(constant=2.0)
    guess.real_shot_record = exact.forward_solution_receivers
    guess.enable_implemented_adjoint()
    guess.forward_solve()

    assert guess.use_vertex_only_mesh

    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
    )

    assert guess.forward_residual_form is not None
    assert isinstance(gradient, fire.Function)
    assert np.isfinite(fire.norm(gradient))


def test_elastic_implemented_adjoint_uses_forward_residual_form():
    model = _small_elastic_model()
    exact_model = deepcopy(model)
    exact_model["synthetic_data"]["density"] = 1.2

    exact = _solve_elastic(exact_model, use_vertex_only_mesh=True)
    guess = _solve_elastic(
        model, implemented_adjoint=True,
        real_shot_record=exact.forward_solution_receivers,
    )

    assert guess.use_vertex_only_mesh

    functional = guess.functional_value
    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
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
def test_elastic_implemented_adjoint_taylor_remainder(parameter):
    model = _small_elastic_model()
    exact_model = deepcopy(model)
    exact_model["synthetic_data"][parameter.value] = (
        1.2 * exact_model["synthetic_data"][parameter.value]
    )

    exact = _solve_elastic(exact_model, use_vertex_only_mesh=True)
    real_record = exact.forward_solution_receivers

    guess = _solve_elastic(
        model, implemented_adjoint=True,
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
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
    )

    direction = fire.Function(base_controls[parameter].function_space())
    direction_shape = direction.dat.data_ro.shape
    direction.dat.data_wo[:] = (
        np.random.default_rng(5).random(direction_shape) - 0.5
    )
    direction.assign(direction / fire.norm(direction))
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


def test_acoustic_implemented_adjoint_taylor_remainder():
    model = _taylor_acoustic_model()

    exact = _solve_acoustic(model, 1.5)
    real_record = exact.forward_solution_receivers

    guess = _solve_acoustic(
        model, 2.0, implemented_adjoint=True,
        real_shot_record=real_record,
    )
    base_control = fire.Function(guess.get_control_parameters())
    base_control.assign(guess.get_control_parameters())
    base_functional = guess.functional_value
    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
    )

    direction = fire.Function(guess.get_control_parameter_function_space())
    direction_shape = direction.dat.data_ro.shape
    direction.dat.data_wo[:] = (
        np.random.default_rng(7).random(direction_shape) - 0.5
    )
    direction.assign(direction / fire.norm(direction))
    directional_derivative = fire.assemble(
        gradient * direction * fire.dx(**guess.quadrature_rule)
    )

    steps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
    remainders = []
    directional_errors = []
    for step in steps:
        guess.reset_pressure()
        guess.initial_velocity_model = base_control + step * direction
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


def test_acoustic_pml_implemented_adjoint_uses_mixed_residual_form():
    model = _small_acoustic_pml_model()

    exact = spyro.AcousticWave(dictionary=deepcopy(model))
    exact.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    exact.set_initial_velocity_model(constant=1.5)
    exact.forward_solve()

    guess = spyro.AcousticWave(dictionary=deepcopy(model))
    guess.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    guess.set_initial_velocity_model(constant=2.0)
    guess.real_shot_record = exact.forward_solution_receivers
    guess.enable_implemented_adjoint()
    guess.forward_solve()

    assert guess.forward_solution[0].function_space() == guess.mixed_function_space

    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
    )

    residual_np1, _, _ = guess.forward_residual_states
    assert guess.forward_residual_form is not None
    assert residual_np1.function_space() == guess.mixed_function_space
    assert (
        guess.get_adjoint_source().function_space()
        == guess.source_function.function_space()
    )
    assert guess.get_adjoint_source() is not guess.source_function
    assert isinstance(gradient, fire.Function)
    assert np.isfinite(fire.norm(gradient))
