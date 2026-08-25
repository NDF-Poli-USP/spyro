"""Serial tests for checkpointed automated-adjoint gradients.

Two things are checked here.

The first is the architectural property that motivates the design: the
checkpoint schedule cannot be built when the automated adjoint is enabled,
because the number of forward time steps is not known yet. ``dt`` may still be
replaced by :meth:`~spyro.solvers.wave.Wave.get_and_set_maximum_dt` and
``final_time`` may still change, and the mixed schedule needs the exact step
count. The tests below assert that spyro really does defer construction to the
forward solve, that the schedule it builds matches the ``nt`` the integrator
ran, and that a fresh schedule is built per solve (a
:mod:`checkpoint_schedules` schedule is consumed by the run that executes it).

The second is that checkpointing does not change the gradient. The reference is
the same automated-adjoint gradient the existing serial gradient tests
verify, computed with checkpointing off; the checkpointed run has to reproduce
it. That comparison is much sharper than a Taylor test - it catches an
adjoint that is wrong by a few percent, which a convergence-rate check
would not.
"""

import numpy as np
import pytest

import firedrake as fire
import spyro
from spyro.solvers.adjoint_checkpointing import (
    CheckpointingConfig,
    CheckpointingMode,
)
from spyro.utils.typing import AdjointType

from checkpoint_schedules import (
    MixedCheckpointSchedule,
    SingleMemoryStorageSchedule,
)


# ---------------------------------------------------------------------------
# Configuration: deliberately small. The gradient comparison below is exact,
# so it does not need a well-resolved model to be meaningful - it only needs
# the source to reach the receivers, which sets ``final_time``.
# ---------------------------------------------------------------------------
FINAL_TIME = 0.8
EDGE_LENGTH = 0.2
DT = 0.004


def build_dictionary():
    """Return the model dictionary shared by every test in this module."""
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 2,
        "dimension": 2,
    }
    dictionary["parallelism"] = {"type": "automatic"}
    dictionary["mesh"] = {
        "length_z": 1.0,
        "length_x": 1.0,
        "length_y": 0.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.1, 0.5)],
        "frequency": 5.0,
        "delay": 1.5,
        "delay_type": "multiples_of_minimum",
        "receiver_locations": spyro.create_transect((-0.8, 0.1), (-0.8, 0.9), 5),
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,
        "final_time": FINAL_TIME,
        "dt": DT,
        "amplitude": 1,
        "output_frequency": 100000,
        "gradient_sampling_frequency": 1,
    }
    dictionary["visualization"] = {
        "forward_output": False,
        "forward_output_filename": None,
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "adjoint_output": False,
        "adjoint_filename": None,
        "debug_output": False,
    }
    return dictionary


def expected_nt(wave):
    """Number of steps the central-difference integrator runs for ``wave``."""
    return int(wave.final_time / wave.dt) + 1


@pytest.fixture(scope="module")
def observed_record():
    """Receiver data from a two-layer 'exact' model, used as the shot record."""
    wave = spyro.AcousticWave(dictionary=build_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(
        conditional=fire.conditional(wave.mesh_z > -0.5, 1.5, 3.5),
        dg_velocity_model=False,
    )
    wave.forward_solve()
    return wave.forward_solution_receivers


def build_guess(observed_record, checkpointing=False, snapshots=None):
    """Return a guess-model wave object with the automated adjoint enabled.

    Parameters
    ----------
    observed_record : array_like
        Receiver data used as the real shot record.
    checkpointing : bool, optional
        Whether to checkpoint the tape.
    snapshots : int, optional
        Snapshot budget, forwarded to
        :meth:`~spyro.solvers.wave.Wave.enable_automated_adjoint`.

    Returns
    -------
    spyro.AcousticWave
        The wave object, before any forward solve.
    """
    wave = spyro.AcousticWave(dictionary=build_dictionary())
    wave.real_shot_record = observed_record
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(constant=2.0)
    wave.enable_automated_adjoint(
        checkpointing=checkpointing, snapshots=snapshots
    )
    return wave


def gradient_of(wave):
    """Run the forward solve and return the automated-adjoint gradient array."""
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    dJ = wave.gradient_solve(adjoint_type=AdjointType.AUTOMATED_ADJOINT)
    return np.array(dJ.dat.data_ro, copy=True)


# ---------------------------------------------------------------------------
# Deferred schedule construction. These do not solve anything.
# ---------------------------------------------------------------------------
def test_disabled_config_builds_no_schedule():
    """A disabled configuration never produces a schedule."""
    config = CheckpointingConfig()
    assert config.enabled is False
    assert config.mode is CheckpointingMode.NONE
    assert config.build_schedule(100) is None


def test_no_snapshot_budget_selects_single_memory():
    """Without a snapshot budget spyro keeps every step in memory."""
    config = CheckpointingConfig(enabled=True)
    assert config.mode is CheckpointingMode.SINGLE_MEMORY
    assert config.recomputes_forward is False
    assert config.permits_repeated_adjoints is True
    assert isinstance(config.build_schedule(100), SingleMemoryStorageSchedule)


def test_snapshot_budget_selects_mixed_schedule_sized_to_the_run():
    """A snapshot budget selects the mixed schedule, built for exactly nt steps."""
    config = CheckpointingConfig(enabled=True, snapshots=7)
    assert config.mode is CheckpointingMode.MIXED
    assert config.recomputes_forward is True
    assert config.permits_repeated_adjoints is False

    schedule = config.build_schedule(250)
    assert isinstance(schedule, MixedCheckpointSchedule)
    # ``max_n`` is the whole reason the schedule cannot be built earlier.
    assert schedule.max_n == 250
    # A schedule is consumed as it runs, so each call must return a new one.
    assert config.build_schedule(250) is not schedule


def test_snapshot_budget_is_clamped_to_the_number_of_steps():
    """More snapshots than steps is clamped rather than passed through."""
    config = CheckpointingConfig(enabled=True, snapshots=1000)
    assert config.resolve_snapshots(10) == 9
    assert config.build_schedule(10).max_n == 10


def test_default_snapshots_follows_the_square_root_heuristic():
    """The automatic budget is ceil(sqrt(nt)), clamped to at least one."""
    assert CheckpointingConfig.default_snapshots(2001) == 45
    assert CheckpointingConfig.default_snapshots(100) == 10
    assert CheckpointingConfig.default_snapshots(2) == 1


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"snapshots": 4}, ValueError),           # budget without enabling
        ({"enabled": True, "snapshots": 0}, ValueError),
        ({"enabled": True, "snapshots": 2.5}, TypeError),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, error):
    """Nonsensical configurations fail at construction, not mid-solve."""
    with pytest.raises(error):
        CheckpointingConfig(**kwargs)


@pytest.mark.parametrize(
    "total_steps, error",
    [(0, ValueError), (1, ValueError), (10.0, TypeError)],
)
def test_mixed_schedule_rejects_bad_step_counts(total_steps, error):
    """A mixed schedule needs a sane, integral number of steps."""
    with pytest.raises(error):
        CheckpointingConfig(enabled=True, snapshots=3).build_schedule(total_steps)


# ---------------------------------------------------------------------------
# Deferral, end to end through the solver.
# ---------------------------------------------------------------------------
@pytest.mark.newer_firedrake
def test_enabling_the_adjoint_does_not_build_a_schedule(observed_record):
    """Enabling records intent only; no schedule exists until the forward runs."""
    wave = build_guess(observed_record, checkpointing=True, snapshots=5)
    adjoint = wave.automated_adjoint

    assert adjoint.checkpointing.enabled is True
    assert adjoint.checkpoint_schedule is None
    assert adjoint.checkpointing_enabled is False

    wave.forward_solve()
    wave.automated_adjoint.stop_recording()

    assert adjoint.checkpointing_enabled is True
    assert isinstance(adjoint.checkpoint_schedule, MixedCheckpointSchedule)
    assert adjoint.checkpoint_schedule.max_n == expected_nt(wave)


@pytest.mark.newer_firedrake
def test_schedule_follows_a_timestep_size_chosen_after_enabling(observed_record):
    """A ``dt`` set after enabling still gives a correctly sized schedule.

    This is the case that makes eager construction impossible: ``dt`` is often
    only fixed by :meth:`get_and_set_maximum_dt` after the adjoint has been
    enabled, and a schedule built for the old ``dt`` would have the wrong
    ``max_n``.
    """
    wave = build_guess(observed_record, checkpointing=True, snapshots=5)
    nt_at_enable = expected_nt(wave)

    wave.dt = 2 * DT
    nt_after_change = expected_nt(wave)
    assert nt_after_change != nt_at_enable

    wave.forward_solve()
    wave.automated_adjoint.stop_recording()

    assert wave.automated_adjoint.checkpoint_schedule.max_n == nt_after_change


@pytest.mark.newer_firedrake
def test_each_forward_solve_gets_a_fresh_schedule_and_tape(observed_record):
    """Consecutive forward solves must not share a consumed schedule or tape."""
    wave = build_guess(observed_record, checkpointing=True, snapshots=5)

    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    first_schedule = wave.automated_adjoint.checkpoint_schedule
    first_tape = wave.automated_adjoint._tape

    wave.forward_solve()
    wave.automated_adjoint.stop_recording()

    assert wave.automated_adjoint.checkpoint_schedule is not first_schedule
    assert wave.automated_adjoint._tape is not first_tape
    assert wave.automated_adjoint.checkpoint_schedule.max_n == expected_nt(wave)


@pytest.mark.newer_firedrake
def test_start_recording_requires_the_step_count_when_checkpointing(observed_record):
    """Driving the tape by hand without ``nt`` fails loudly, not silently."""
    wave = build_guess(observed_record, checkpointing=True, snapshots=5)
    with pytest.raises(ValueError, match="total_steps"):
        wave.automated_adjoint.start_recording()


# ---------------------------------------------------------------------------
# Gradient equivalence.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_checkpointed_gradient_matches_uncheckpointed(observed_record):
    """Keeping every step in memory reproduces the reference gradient exactly.

    The schedule changes how the tape is stored, not what it computes, so the
    two gradients agree to round-off rather than merely to a tolerance.
    """
    reference = gradient_of(build_guess(observed_record))
    assert np.linalg.norm(reference) > 0.0, "reference gradient is identically zero"

    checkpointed_wave = build_guess(observed_record, checkpointing=True)
    checkpointed = gradient_of(checkpointed_wave)

    assert isinstance(
        checkpointed_wave.automated_adjoint.checkpoint_schedule,
        SingleMemoryStorageSchedule,
    )
    relative_error = np.linalg.norm(checkpointed - reference) / np.linalg.norm(reference)
    assert relative_error < 1e-12, (
        "Checkpointed gradient differs from the uncheckpointed one by "
        f"{relative_error:.3e}"
    )


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_checkpointed_taylor_test(observed_record):
    """The checkpointed gradient passes the same Taylor test as the plain one.

    ``dJdm`` is deliberately not supplied, so ``verify_gradient`` takes the
    derivative itself rather than letting pyadjoint's ``taylor_test`` reach
    past the single-sweep guard into the reduced functional.
    """
    wave = build_guess(observed_record, checkpointing=True)
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()
    wave.automated_adjoint.create_reduced_functional(wave.functional_value)

    size, = np.shape(wave.c.dat.data[:])
    direction = fire.Function(
        wave.c.function_space(), val=np.random.default_rng(0).random(size)
    )
    rate = wave.automated_adjoint.verify_gradient(wave.c, direction=direction)
    assert rate > 1.9, (
        f"Checkpointed automated adjoint failed the Taylor test: rate {rate}"
    )


@pytest.mark.slow
@pytest.mark.newer_firedrake
@pytest.mark.xfail(
    reason=(
        "A recomputation-based checkpoint schedule returns a wrong gradient for "
        "spyro's second-order central-difference stepping, whose restart state "
        "spans two time levels. The functional replays exactly; only the adjoint "
        "is wrong. Reproducible in plain Firedrake with no spyro code, affects "
        "Revolve as well as MixedCheckpointSchedule, and disappears once the "
        "snapshot budget is large enough that nothing is recomputed. Tracked as "
        "an upstream pyadjoint/checkpoint_schedules issue."
    ),
    strict=False,
)
def test_snapshot_budget_gradient_matches_uncheckpointed(observed_record):
    """A bounded snapshot budget should also reproduce the reference gradient."""
    reference = gradient_of(build_guess(observed_record))
    budgeted = gradient_of(
        build_guess(observed_record, checkpointing=True, snapshots=10)
    )
    relative_error = np.linalg.norm(budgeted - reference) / np.linalg.norm(reference)
    assert relative_error < 1e-12, (
        f"Snapshot-budgeted gradient differs by {relative_error:.3e}"
    )


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_snapshot_budget_replays_the_functional_exactly(observed_record):
    """The forward replay under a snapshot budget is exact.

    This is the half of the recomputing schedule that does work, and pinning it
    down localises the upstream defect to the adjoint sweep: re-evaluating the
    reduced functional at the recorded control reproduces the taped value to
    round-off even though the gradient does not.
    """
    wave = build_guess(observed_record, checkpointing=True, snapshots=10)
    wave.forward_solve()
    wave.automated_adjoint.stop_recording()

    taped = float(wave.functional_value)
    reduced_functional = wave.automated_adjoint.create_reduced_functional(
        wave.functional_value
    )
    replayed = float(reduced_functional(wave.c.copy(deepcopy=True)))

    assert abs(replayed - taped) <= 1e-12 * abs(taped), (
        f"Forward replay under checkpointing changed the functional: "
        f"{taped!r} -> {replayed!r}"
    )


@pytest.mark.slow
@pytest.mark.newer_firedrake
def test_second_gradient_from_one_tape_is_rejected(observed_record):
    """An offline schedule permits one adjoint sweep; a second must raise.

    pyadjoint does not detect the second sweep - it returns a gradient built
    from consumed checkpoints. Turning that into an exception is what keeps a
    silently wrong gradient out of an inversion.
    """
    wave = build_guess(observed_record, checkpointing=True, snapshots=10)
    gradient_of(wave)

    with pytest.raises(RuntimeError, match="single adjoint sweep"):
        wave.automated_adjoint.compute_gradient()

    # Re-running the forward records a fresh tape, which is allowed again.
    gradient_of(wave)
