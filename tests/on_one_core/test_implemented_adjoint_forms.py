from copy import deepcopy

import firedrake as fire
import numpy as np
import spyro

from spyro.utils.typing import AdjointType


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
            "final_time": 0.002,
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
    model["time_axis"]["final_time"] = 0.05
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


def _solve_acoustic(model, velocity, edge_length=0.5, implemented_adjoint=False):
    wave = spyro.AcousticWave(dictionary=deepcopy(model))
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
    if isinstance(velocity, (float, int)):
        wave.set_initial_velocity_model(constant=velocity)
    else:
        wave.set_control_parameters(velocity)
    if implemented_adjoint:
        wave.enable_implemented_adjoint()
    wave.forward_solve()
    return wave


def test_acoustic_implemented_adjoint_uses_forward_residual_form():
    model = _small_acoustic_model()

    exact = spyro.AcousticWave(dictionary=deepcopy(model))
    exact.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    exact.set_initial_velocity_model(constant=1.5)
    exact.forward_solve()

    guess = spyro.AcousticWave(dictionary=deepcopy(model))
    guess.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    guess.set_initial_velocity_model(constant=2.0)
    guess.enable_implemented_adjoint()
    guess.forward_solve()

    misfit = exact.forward_solution_receivers - guess.forward_solution_receivers
    gradient = guess.gradient_solve(
        misfit=misfit,
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
    )

    assert guess.forward_residual_form is not None
    assert isinstance(gradient, fire.Function)
    assert np.isfinite(fire.norm(gradient))


def test_acoustic_implemented_adjoint_taylor_remainder():
    model = _taylor_acoustic_model()

    exact = _solve_acoustic(model, 1.5)
    real_record = exact.forward_solution_receivers

    guess = _solve_acoustic(model, 2.0, implemented_adjoint=True)
    base_control = fire.Function(guess.get_control_parameters())
    base_control.assign(guess.get_control_parameters())
    misfit = real_record - guess.forward_solution_receivers
    base_functional = spyro.utils.compute_functional(guess, misfit)
    gradient = guess.gradient_solve(
        misfit=misfit,
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
    )

    direction = fire.Function(guess.get_control_parameter_function_space())
    direction.dat.data_wo[:] = (
        np.random.default_rng(7).random(direction.dat.data_wo.shape) - 0.5
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
        perturbed_misfit = real_record - guess.forward_solution_receivers
        perturbed_functional = spyro.utils.compute_functional(
            guess, perturbed_misfit,
        )
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
    guess.enable_implemented_adjoint()
    guess.forward_solve()

    assert guess.forward_solution[0].function_space() == guess.mixed_function_space

    misfit = exact.forward_solution_receivers - guess.forward_solution_receivers
    gradient = guess.gradient_solve(
        misfit=misfit,
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
