"""Replaying a checkpointed tape before its first adjoint.

With a checkpoint schedule, pyadjoint gets the first adjoint of a tape wrong
when the tape was replayed before it. Spyro runs that first adjoint itself
before any replay (``AutomatedAdjoint.ensure_first_adjoint``), so replays,
Taylor tests and TAO inversions over a checkpointed tape must give what the
same computation gives on a tape without checkpointing.
"""

import numpy as np
import pytest

import firedrake as fire
import spyro
from spyro.utils.typing import AdjointType

from .test_fwi_automated_adjoint import (
    ACOUSTIC_GUESS, ACOUSTIC_REAL, build_dictionary as fwi_dictionary,
)
from .test_gradient_auto_adjoint_nrbc import (
    ABSORBING_SIDES, guess_wave, observed_record,
)


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    """Relative error of ``value`` with respect to ``reference``.

    Parameters
    ----------
    value : numpy.ndarray
        Computed array.
    reference : numpy.ndarray
        Reference array, assumed nonzero.

    Returns
    -------
    float
        ``||value - reference|| / ||reference||``.
    """
    return np.linalg.norm(value - reference) / np.linalg.norm(reference)


def recorded_wave(record: np.ndarray, velocity: np.ndarray | None,
                  **schedule) -> spyro.AcousticWave:
    """Record a forward solve and create its reduced functional.

    Parameters
    ----------
    record : numpy.ndarray
        Observed receiver data.
    velocity : numpy.ndarray or None
        Nodal values of the velocity to record at, or ``None`` for the guess.
    **schedule
        Keyword arguments of
        :meth:`~spyro.solvers.wave.Wave.enable_automated_adjoint`.

    Returns
    -------
    spyro.AcousticWave
        The wave object, with its reduced functional created.
    """
    wave = guess_wave(ABSORBING_SIDES["all_sides"])
    wave.real_shot_record = record
    if velocity is not None:
        # Same mesh construction, so the same degree-of-freedom ordering.
        wave.initial_velocity_model.dat.data[:] = velocity
    wave.enable_automated_adjoint(**schedule)
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    wave.automated_adjoint.create_reduced_functional(wave.functional_value)
    return wave


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_replay_before_first_gradient() -> None:
    """A gradient taken after a replay matches a fresh recording there.

    The tape is replayed at a new velocity before it was ever differentiated,
    which is what an optimizer does.
    """
    record = observed_record(ABSORBING_SIDES["all_sides"])

    wave = recorded_wave(record, None, checkpointing=True, snapshots=5)
    adjoint = wave.automated_adjoint
    new_velocity = fire.Function(wave.c.function_space()).interpolate(
        wave.c + 0.1 * fire.sin(np.pi * wave.mesh_x)
    )
    J_replay = float(adjoint.recompute_functional(new_velocity))
    grad_replay = np.array(adjoint.compute_gradient().dat.data_ro, copy=True)
    values = np.array(new_velocity.dat.data_ro, copy=True)
    adjoint.clear_tape()

    fresh = recorded_wave(record, values)
    J_fresh = float(fresh.functional_value)
    grad_fresh = np.array(
        fresh.automated_adjoint.compute_gradient().dat.data_ro, copy=True
    )
    fresh.automated_adjoint.clear_tape()

    assert abs(J_replay - J_fresh) <= 1e-12 * abs(J_fresh)
    assert relative_error(grad_replay, grad_fresh) < 1e-10


def inversion(tmp_path, adjoint_options: dict | None) -> tuple:
    """Run a short acoustic TAO inversion.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the inversion writes its output to.
    adjoint_options : dict or None
        ``adjoint_options`` of :meth:`run_fwi`.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The functional history and the velocity model the run stopped at.
    """
    fwi = spyro.FullWaveformInversion(dictionary=fwi_dictionary())
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(ACOUSTIC_REAL)
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=ACOUSTIC_GUESS)
    kwargs = {}
    if adjoint_options is not None:
        kwargs["adjoint_options"] = adjoint_options
    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=2.0, vmax=3.5, maxiter=3, **kwargs,
    )
    history = np.array(fwi.functional_history, dtype=float)
    fwi.wave.automated_adjoint.clear_tape()
    return history, np.array(result.dat.data_ro, copy=True)


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_tao_inversion_with_checkpointing(tmp_path, monkeypatch) -> None:
    """A TAO inversion follows the same path with and without checkpointing.

    TAO evaluates the functional at its starting point before it asks for the
    first gradient, so a wrong first adjoint would send the checkpointed run
    down a different path from the first step.
    """
    monkeypatch.chdir(tmp_path)
    history_ref, model_ref = inversion(tmp_path, None)
    history, model = inversion(
        tmp_path, {"checkpointing": True, "snapshots": 3},
    )

    assert len(history) == len(history_ref) > 1
    assert relative_error(history, history_ref) < 1e-10
    assert relative_error(model, model_ref) < 1e-10
