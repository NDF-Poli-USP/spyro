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

    assert hasattr(guess, "forward_residual_form")
    assert guess.misfit_form is not None
    assert isinstance(gradient, fire.Function)
    assert np.isfinite(fire.norm(gradient))


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
    assert hasattr(guess, "forward_residual_form")
    assert residual_np1.function_space() == guess.mixed_function_space
    assert isinstance(gradient, fire.Function)
    assert np.isfinite(fire.norm(gradient))
