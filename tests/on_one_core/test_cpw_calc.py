"""Acoustic cells-per-wavelength search on the homogeneous 2D model."""

import numpy as np
import pytest

import spyro

FREQUENCY = 5.0
VELOCITY = 1.5

#: Default Ricker delay, 1.5 times the distance between its minima, in s.
RICKER_DELAY = 1.5 * np.sqrt(6.0) / (np.pi * FREQUENCY)


def make_parameters(method: str, degree: int) -> dict:
    """Build calculator parameters small enough to run in under 2 minutes.

    The reference uses the reduced receiver grid, the solves use a fixed,
    coarse time step, and the search starts from its upper bound.

    Parameters
    ----------
    method : str
        Finite element method to evaluate.
    degree : int
        Polynomial degree to evaluate.

    Returns
    -------
    dict
        Parameters for :class:`spyro.tools.MeshingParameterCalculator`.
    """
    return {
        "source_frequency": FREQUENCY,
        "minimum_velocity_in_the_domain": VELOCITY,
        "velocity_profile_type": "homogeneous",
        "velocity_model_file_name": None,
        "FEM_method_to_evaluate": method,
        "dimension": 2,
        "receiver_setup": "near",
        "testing": True,
        "time-step_calculation": "float",
        "time-step": 0.002,
        "reference_degree": None,
        "C_reference": None,
        "desired_degree": degree,
        "C_initial": 2.5,
        "C_max": 2.5,
        "accepted_error_threshold": 0.01,
        "C_accuracy": 0.1,
    }


@pytest.mark.slow
def test_acoustic_cpw_search_quads() -> None:
    """The analytical reference is causal and the search finds the minimum."""
    calculator = spyro.tools.MeshingParameterCalculator(
        make_parameters("spectral_quadrilateral", 4)
    )
    wave = calculator.initial_guess_object
    reference = calculator.reference_solution
    assert reference.shape == (round(wave.final_time / wave.dt) + 1, 4)

    # Each receiver peaks just after the delayed direct arrival
    source = np.asarray(wave.source_locations[0])
    for i, receiver in enumerate(wave.receiver_locations):
        arrival = np.linalg.norm(np.asarray(receiver) - source) / VELOCITY
        peak_time = np.argmax(reference[:, i]) * wave.dt
        assert 0.0 <= peak_time - (arrival + RICKER_DELAY) < 0.25 / FREQUENCY

    cpw = calculator.find_minimum()

    errors = {e.cpw: e.error for e in calculator.search_history}
    assert np.isclose(cpw, 2.1)
    assert errors[cpw] <= 0.01
    assert errors[round(cpw - 0.1, 10)] > 0.01
