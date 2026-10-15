"""Automated adjoint of the acoustic solver under recomputation.

A checkpoint schedule that keeps few checkpoints recomputes most of the
forward solve from them, replaying the taped time steps, including the point
source amplitude. The gradient must not depend on the schedule, and replaying
the tape at a different velocity model must match a fresh recording there.
"""

import numpy as np
import pytest

import firedrake as fire
from checkpoint_schedules import Forward
import spyro
from spyro.utils.typing import AdjointType

from .test_gradient_auto_adjoint_nrbc import (
    ABSORBING_SIDES, EDGE_LENGTH, build_dictionary, observed_record,
)


def guess_wave(absorbing_sides: dict | None,
               use_vertex_only_mesh: bool = False) -> spyro.AcousticWave:
    """Build a guess model with a heterogeneous velocity.

    Parameters
    ----------
    absorbing_sides : dict or None
        ``absorb_<side>`` flags of the non-reflecting boundary condition, or
        ``None`` for reflecting boundaries.
    use_vertex_only_mesh : bool, optional
        Whether sources and receivers use vertex-only meshes. The automated
        adjoint turns this on regardless.

    Returns
    -------
    spyro.AcousticWave
        The wave object, with mesh and velocity model set, not yet solved.
    """
    dictionary = build_dictionary(absorbing_sides)
    dictionary["acquisition"]["use_vertex_only_mesh"] = use_vertex_only_mesh
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(
        conditional=2.0 - 0.5 * wave.mesh_z
        + 0.2 * fire.exp(-((wave.mesh_x - 0.5) ** 2) / 0.02),
        dg_velocity_model=False,
    )
    return wave


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


def taped_gradient(wave: spyro.AcousticWave, **schedule) -> tuple:
    """Record a forward solve and return its functional and gradient.

    Parameters
    ----------
    wave : spyro.AcousticWave
        Wave object with mesh, velocity model and real shot record set.
    **schedule
        Keyword arguments of
        :meth:`~spyro.solvers.wave.Wave.enable_automated_adjoint`.

    Returns
    -------
    tuple of (float, numpy.ndarray)
        The functional value and a copy of the gradient's values.
    """
    wave.enable_automated_adjoint(**schedule)
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    dJ = wave.gradient_solve(adjoint_type=AdjointType.AUTOMATED_ADJOINT)
    return float(wave.functional_value), np.array(dJ.dat.data_ro, copy=True)


@pytest.mark.slow
@pytest.mark.newer_firedrake
@pytest.mark.parametrize("sides", [None, "all_sides"])
def test_gradient_with_few_checkpoints(sides: str | None) -> None:
    """Few checkpoints force recomputation and leave the gradient unchanged.

    Parameters
    ----------
    sides : str or None
        Key into :data:`ABSORBING_SIDES`, or ``None`` for reflecting
        boundaries.
    """
    absorbing_sides = None if sides is None else ABSORBING_SIDES[sides]
    record = observed_record(absorbing_sides)

    results = {}
    for name, schedule in {
        "none": {},
        "mixed": {"checkpointing": True, "snapshots": 3},
    }.items():
        wave = guess_wave(absorbing_sides)
        wave.real_shot_record = record
        results[name] = taped_gradient(wave, **schedule)
        if name == "mixed":
            manager = wave.automated_adjoint._tape._checkpoint_manager
            nt = manager.total_timesteps
            recomputed = sum(
                min(action.n1, nt) - action.n0
                for action in manager.reverse_schedule
                if isinstance(action, Forward)
            )
            assert recomputed > nt, "the schedule did not recompute"
        wave.automated_adjoint.clear_tape()

    J_ref, grad_ref = results["none"]
    J, grad = results["mixed"]
    assert abs(J - J_ref) <= 1e-12 * abs(J_ref)
    assert relative_error(grad, grad_ref) < 1e-12


@pytest.mark.slow
@pytest.mark.newer_firedrake
@pytest.mark.xfail(
    strict=True,
    reason="pyadjoint: replaying a checkpointed tape before its first "
           "adjoint makes that adjoint wrong (about 2e-3 here); see "
           "benchmarks/adjoint_acceleration/repro_replay_before_first_adjoint.py",
)
def test_reevaluation_after_velocity_change() -> None:
    """Replaying a checkpointed tape at a new velocity matches a new solve.

    The functional and gradient obtained from the reduced functional at a
    perturbed model must match those of a fresh recording at that model.
    """
    absorbing_sides = ABSORBING_SIDES["all_sides"]
    record = observed_record(absorbing_sides)

    wave = guess_wave(absorbing_sides)
    wave.real_shot_record = record
    wave.enable_automated_adjoint(checkpointing=True, snapshots=5)
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    functional = wave.automated_adjoint.create_reduced_functional(
        wave.functional_value
    )
    c_new = fire.Function(wave.c.function_space()).interpolate(
        wave.c + 0.1 * fire.sin(np.pi * wave.mesh_x)
    )
    J_replay = float(functional(c_new))
    grad_replay = np.array(
        functional.derivative(apply_riesz=True).dat.data_ro, copy=True
    )
    new_values = np.array(c_new.dat.data_ro, copy=True)
    wave.automated_adjoint.clear_tape()

    fresh = guess_wave(absorbing_sides)
    fresh.real_shot_record = record
    # Same mesh construction, so the same degree-of-freedom ordering.
    fresh.initial_velocity_model.dat.data[:] = new_values
    J_fresh, grad_fresh = taped_gradient(fresh)
    fresh.automated_adjoint.clear_tape()

    assert abs(J_replay - J_fresh) <= 1e-12 * abs(J_fresh)
    assert relative_error(grad_replay, grad_fresh) < 1e-10


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_taylor_test_with_nrbc_and_recomputation() -> None:
    """Second-order Taylor convergence with NRBC and few checkpoints."""
    absorbing_sides = ABSORBING_SIDES["all_sides"]
    wave = guess_wave(absorbing_sides)
    wave.real_shot_record = observed_record(absorbing_sides)
    wave.enable_automated_adjoint(checkpointing=True, snapshots=5)
    try:
        wave.forward_solve()
        wave.automated_adjoint.stop_recording()
        wave.automated_adjoint.create_reduced_functional(wave.functional_value)
        size, = np.shape(wave.c.dat.data_ro)
        direction = fire.Function(
            wave.c.function_space(),
            val=np.random.default_rng(0).random(size),
        )
        rate = wave.automated_adjoint.verify_gradient(
            wave.c, direction=direction
        )
        assert rate > 1.9, f"Taylor convergence rate {rate}"
    finally:
        wave.automated_adjoint.clear_tape()
