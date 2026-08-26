"""Checkpointing parity for the three-dimensional elastic adjoint."""

import numpy as np
import pytest

import spyro
from spyro.utils.typing import AdjointType, ElasticMaterialParameter


pytestmark = [pytest.mark.newer_firedrake, pytest.mark.slow]

PARAMETERS = (
    ElasticMaterialParameter.P_WAVE_VELOCITY,
    ElasticMaterialParameter.S_WAVE_VELOCITY,
)


def _dictionary(material: dict) -> dict:
    """Return a tiny 3D elastic setup suitable for gradient parity tests."""
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 2,
            "dimension": 3,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 1.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.2, 0.5, 0.5)],
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "amplitude": np.array([1.0, 0.0, 0.0]),
            "receiver_locations": [
                (-0.8, 0.3, 0.3),
                (-0.8, 0.5, 0.5),
                (-0.8, 0.7, 0.7),
            ],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.5,
            "dt": 0.002,
            "output_frequency": 100000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
        "synthetic_data": {
            "type": "object",
            **material,
            "real_velocity_file": None,
        },
    }


@pytest.fixture(scope="module")
def observed_record():
    """Generate one record from a homogeneous model distinct from the guess."""
    wave = spyro.IsotropicWave(_dictionary({
        "density": 0.12,
        "p_wave_velocity": 1.75,
        "s_wave_velocity": 0.82,
    }))
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave.forward_solve()
    return wave.forward_solution_receivers


def _gradient(observed_record, *, checkpointing: bool):
    wave = spyro.IsotropicWave(_dictionary({
        "density": 0.12,
        "p_wave_velocity": 1.60,
        "s_wave_velocity": 0.72,
    }))
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave.real_shot_record = observed_record
    wave.enable_automated_adjoint(
        control_parameters=PARAMETERS,
        checkpointing=checkpointing,
        snapshots=8 if checkpointing else None,
        gc_timestep_frequency=25 if checkpointing else None,
    )
    try:
        wave.forward_solve()
        gradient = wave.gradient_solve(
            adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        )
        result = {
            parameter: np.array(gradient[parameter].dat.data_ro, copy=True)
            for parameter in PARAMETERS
        }
        if checkpointing:
            # Optimizers repeatedly execute objective-gradient pairs on one
            # ReducedFunctional. A finite schedule must reset on the forward
            # replay before the next reverse sweep.
            wave.automated_adjoint.recompute_functional(
                wave.automated_adjoint.controls
            )
            repeated = wave.automated_adjoint.compute_gradient()
            for parameter, repeated_gradient in zip(PARAMETERS, repeated):
                repeated_values = np.asarray(repeated_gradient.dat.data_ro)
                error = np.linalg.norm(repeated_values - result[parameter])
                scale = np.linalg.norm(result[parameter])
                assert error / scale < 1.0e-12
        return result
    finally:
        wave.automated_adjoint.clear_tape()


def test_elastic_3d_gradient_matches_mixed_checkpointing(observed_record):
    """Mixed checkpointing must preserve both elastic control gradients."""
    reference = _gradient(observed_record, checkpointing=False)
    checkpointed = _gradient(observed_record, checkpointing=True)

    relative_errors = {}
    for parameter in PARAMETERS:
        denominator = np.linalg.norm(reference[parameter])
        assert denominator > 0.0, f"zero reference gradient for {parameter}"
        relative_errors[parameter] = (
            np.linalg.norm(checkpointed[parameter] - reference[parameter])
            / denominator
        )

    assert all(error < 1.0e-12 for error in relative_errors.values()), (
        f"3D elastic gradients differ with checkpointing: {relative_errors}"
    )
