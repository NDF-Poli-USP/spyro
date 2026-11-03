"""Time-receiver window of the misfit functional.

``wave.misfit_window`` weights each sample of the misfit. With the automated
adjoint the functional is accumulated during the forward solve, so the
window has to enter there; this checks it against the same functional
computed from the recorded traces.
"""

import numpy as np
import pytest

import spyro

FINAL_TIME = 0.4
DT = 0.002


def build_wave(velocity: float) -> spyro.AcousticWave:
    """Small constant-velocity acoustic problem.

    Parameters
    ----------
    velocity : float
        Velocity (km/s).

    Returns
    -------
    spyro.AcousticWave
        Wave with mesh and velocity set.
    """
    dictionary = {
        "options": {"cell_type": "T", "variant": "lumped", "degree": 2,
                    "dimension": 2},
        "parallelism": {"type": "automatic"},
        "mesh": {"length_z": 1.0, "length_x": 1.0, "length_y": 0.0,
                 "mesh_file": None, "mesh_type": "firedrake_mesh"},
        "acquisition": {"source_type": "ricker",
                        "source_locations": [(-0.1, 0.5)],
                        "frequency": 5.0, "delay": 1.5,
                        "delay_type": "multiples_of_minimum",
                        "receiver_locations": spyro.create_transect(
                            (-0.6, 0.2), (-0.6, 0.8), 4)},
        "time_axis": {"initial_time": 0.0, "final_time": FINAL_TIME, "dt": DT,
                      "amplitude": 1, "output_frequency": 10**9,
                      "gradient_sampling_frequency": 1},
        "visualization": {"forward_output": False, "forward_output_filename": None,
                          "fwi_velocity_model_output": False,
                          "velocity_model_filename": None, "gradient_output": False,
                          "gradient_filename": None, "adjoint_output": False,
                          "adjoint_filename": None, "debug_output": False},
    }
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=velocity)
    return wave


def test_windowed_functional_matches_traces() -> None:
    """The windowed per-step functional equals the windowed trapezoidal sum."""
    observed = build_wave(2.0)
    observed.forward_solve()
    record = observed.forward_solution_receivers

    wave = build_wave(1.8)
    wave.real_shot_record = record
    steps = record.shape[0]
    rng = np.random.default_rng(3)
    window = (rng.random(record.shape) > 0.4).astype(float)
    wave.misfit_window = window
    wave.enable_automated_adjoint()
    wave.forward_solve()

    residual = window * (record - wave.forward_solution_receivers)
    weights = np.ones(steps)
    weights[[0, -1]] = 0.5
    expected = 0.5 * DT * float((weights[:, None] * residual**2).sum())
    assert float(wave.functional_value) == pytest.approx(expected, rel=1e-10)
    wave.automated_adjoint.clear_tape()
