"""Checkpointing configuration for spyro's automated adjoint.

Reverse-mode differentiation of a time-dependent wave propagation has to make
the whole forward trajectory available to the adjoint sweep. Storing every
time step is the fastest option but its memory cost grows linearly with the
number of steps, which is what makes long 3D runs unaffordable.
*Checkpointing* trades that memory for recomputation: only a bounded number of
forward states is kept and the missing ones are recomputed from the nearest
stored state while the adjoint sweeps backwards.

pyadjoint delegates that decision to a :mod:`checkpoint_schedules` *schedule*.
Spyro uses exactly two of them, and picks between them itself - callers choose
a memory budget, not a schedule:

no snapshot budget
    :class:`~checkpoint_schedules.SingleMemoryStorageSchedule` - every adjoint
    dependency is held in RAM and nothing is recomputed. Memory is
    :math:`O(n_t)`, runtime is one forward plus one adjoint sweep.

a snapshot budget
    :class:`~checkpoint_schedules.MixedCheckpointSchedule` - only ``snapshots``
    checkpointing units are kept, mixing forward-restart data and non-linear
    dependency data in the same budget (Maddison, 2024). Memory is
    :math:`O(\\text{snapshots})` instead of :math:`O(n_t)`, at the cost of
    recomputing the forward between checkpoints.

Snapshots are always kept in RAM. Disk storage would additionally require
:func:`firedrake.adjoint.enable_disk_checkpointing`, which spyro does not set
up.

Why the schedule cannot be built when the adjoint is enabled
-----------------------------------------------------------
:class:`~checkpoint_schedules.MixedCheckpointSchedule` needs ``max_n``, the
total number of forward steps, at construction time. When
:meth:`~spyro.solvers.wave.Wave.enable_automated_adjoint` runs, that number is
not known yet: ``dt`` may still be replaced by
:meth:`~spyro.solvers.wave.Wave.get_and_set_maximum_dt`, and ``final_time`` may
still change. The step count is only settled inside the time integrator, where
``nt = int(final_time / dt) + 1`` is formed.

This module therefore stores the *intent* in a :class:`CheckpointingConfig` and
defers schedule construction to :meth:`CheckpointingConfig.build_schedule`,
which the automated adjoint calls once the forward loop knows ``nt``. Deferring
is not merely convenient: a schedule built with the wrong ``max_n`` either
raises ``CheckpointError: Not enough timesteps in schedule`` (when
``max_n < nt``) or silently wastes recomputation (when ``max_n > nt``).

A schedule instance is also single-use - pyadjoint's ``CheckpointManager``
consumes it as the forward is taped - so a *fresh* schedule must be built for
every forward solve. That is why the configuration is the persistent object and
the schedule is not.

.. warning::

    With the currently installed pyadjoint and ``checkpoint_schedules``, a
    recomputation-based schedule returns an **incorrect gradient** for
    spyro's second-order central-difference time stepping, whose restart state
    spans two time levels (``u_n`` and ``u_nm1``). The functional replays
    exactly; only the adjoint is wrong, and it is wrong silently. The
    behaviour reproduces in a few dozen lines of plain Firedrake with no spyro
    code involved, affects ``Revolve`` as well as ``MixedCheckpointSchedule``,
    and disappears as soon as the snapshot budget is large enough that nothing
    has to be recomputed. Until that is resolved upstream, only the
    no-snapshot-budget configuration
    (:class:`~checkpoint_schedules.SingleMemoryStorageSchedule`) is verified
    against the reference gradient; see
    ``tests/on_one_core/test_gradient_checkpointing.py``.
"""

import math
from dataclasses import dataclass
from enum import Enum

from checkpoint_schedules import (
    MixedCheckpointSchedule,
    SingleMemoryStorageSchedule,
    StorageType,
)

__all__ = [
    "CheckpointingMode",
    "CheckpointingConfig",
]


class CheckpointingMode(Enum):
    """Which schedule spyro selected for a given configuration.

    This is derived from the configuration rather than chosen by the caller;
    it exists so tests and diagnostics can name the backend in play.

    NONE:
        No checkpoint manager on the tape. pyadjoint keeps every block output,
        which is spyro's historical behaviour.
    SINGLE_MEMORY:
        :class:`~checkpoint_schedules.SingleMemoryStorageSchedule`, selected
        when checkpointing is on and no snapshot budget was given.
    MIXED:
        :class:`~checkpoint_schedules.MixedCheckpointSchedule`, selected when a
        snapshot budget was given.
    """

    NONE = "none"
    SINGLE_MEMORY = "single_memory"
    MIXED = "mixed"


@dataclass(frozen=True)
class CheckpointingConfig:
    """Checkpointing intent, resolved into a schedule once ``nt`` is known.

    The configuration is immutable and carries no schedule: schedules are built
    on demand by :meth:`build_schedule`, one per forward solve, because a
    :mod:`checkpoint_schedules` schedule is consumed as it is executed and
    because the mixed schedule needs the total step count.

    Callers express a memory budget; spyro maps it onto a schedule. Passing a
    ``snapshots`` budget selects the mixed schedule, leaving it ``None``
    selects single-memory storage.

    Parameters
    ----------
    enabled : bool, optional
        Whether to install a checkpoint manager at all. Defaults to ``False``,
        which leaves the tape exactly as it was before checkpointing existed.
    snapshots : int, optional
        Number of checkpointing units to keep. ``None`` (the default) keeps
        every step in RAM. An explicit budget is clamped to
        ``total_steps - 1`` when the schedule is built.

    Examples
    --------
    .. code-block:: python

        CheckpointingConfig(enabled=True)                 # single-memory
        CheckpointingConfig(enabled=True, snapshots=20)   # mixed, 20 units
    """

    enabled: bool = False
    snapshots: int = None

    def __post_init__(self):
        # ``frozen=True`` forbids plain assignment, so normalisation has to go
        # through ``object.__setattr__``.
        object.__setattr__(self, "enabled", bool(self.enabled))
        if self.snapshots is not None:
            if not isinstance(self.snapshots, int) or isinstance(self.snapshots, bool):
                raise TypeError(
                    f"snapshots must be an int, got {type(self.snapshots).__name__}."
                )
            if self.snapshots < 1:
                raise ValueError(
                    f"snapshots must be at least 1, got {self.snapshots}."
                )
            if not self.enabled:
                raise ValueError(
                    "snapshots was given but checkpointing is disabled. Enable "
                    "checkpointing to use a snapshot budget."
                )

    @property
    def mode(self):
        """The :class:`CheckpointingMode` this configuration resolves to."""
        if not self.enabled:
            return CheckpointingMode.NONE
        if self.snapshots is None:
            return CheckpointingMode.SINGLE_MEMORY
        return CheckpointingMode.MIXED

    @property
    def recomputes_forward(self):
        """``True`` when the schedule recomputes forward steps for the adjoint.

        Single-memory storage keeps everything, so it never recomputes; the
        mixed schedule does whenever its budget is smaller than the run.
        """
        return self.mode is CheckpointingMode.MIXED

    @property
    def permits_repeated_adjoints(self):
        """``True`` when more than one adjoint sweep per tape is well defined.

        :class:`~checkpoint_schedules.MixedCheckpointSchedule` is an *offline*
        schedule: one reverse sweep exhausts it, and a second gradient taken
        from the same tape is silently wrong rather than an error.
        :class:`~checkpoint_schedules.SingleMemoryStorageSchedule` is *online*
        and permits unlimited adjoint calculations.
        """
        return self.mode is not CheckpointingMode.MIXED

    @staticmethod
    def default_snapshots(total_steps):
        """Snapshot budget used when a budget is wanted but none was given.

        Uses the square-root heuristic, ``ceil(sqrt(n_t))``, which balances the
        two costs the schedule trades against each other: peak memory grows
        with the number of snapshots while recomputation grows with the number
        of steps per snapshot interval, and both are :math:`O(\\sqrt{n_t})`
        here. Clamped to ``[1, total_steps - 1]``.

        Parameters
        ----------
        total_steps : int
            Number of forward time steps.

        Returns
        -------
        int
            The snapshot budget.
        """
        snapshots = math.isqrt(max(total_steps - 1, 0)) + 1
        return max(1, min(snapshots, max(total_steps - 1, 1)))

    def resolve_snapshots(self, total_steps):
        """Return the snapshot budget actually used for ``total_steps`` steps.

        An explicit request is clamped to ``total_steps - 1``: a mixed
        schedule cannot use more checkpointing units than it has steps to place
        them in.

        Parameters
        ----------
        total_steps : int
            Number of forward time steps.

        Returns
        -------
        int or None
            The snapshot budget, or ``None`` when this configuration does not
            use one.
        """
        if self.snapshots is None:
            return None
        return max(1, min(self.snapshots, max(total_steps - 1, 1)))

    def build_schedule(self, total_steps):
        """Build a fresh schedule for a forward run of ``total_steps`` steps.

        Call once per forward solve: a schedule is consumed as pyadjoint
        executes it, so it cannot be shared between tapes.

        Parameters
        ----------
        total_steps : int
            Number of forward time steps that will be taped. Must match what
            the forward loop actually runs.

        Returns
        -------
        checkpoint_schedules.CheckpointSchedule or None
            The schedule, or ``None`` when checkpointing is disabled.

        Raises
        ------
        TypeError
            If ``total_steps`` is not an integer.
        ValueError
            If ``total_steps`` is not positive, or is too short for a mixed
            schedule.
        """
        if not self.enabled:
            return None
        if not isinstance(total_steps, int) or isinstance(total_steps, bool):
            raise TypeError(
                f"total_steps must be an int, got {type(total_steps).__name__}."
            )
        if total_steps < 1:
            raise ValueError(f"total_steps must be at least 1, got {total_steps}.")

        if self.mode is CheckpointingMode.SINGLE_MEMORY:
            # Online schedule: it discovers the step count while taping, so
            # ``total_steps`` is not needed to construct it.
            return SingleMemoryStorageSchedule()

        if total_steps < 2:
            raise ValueError(
                "A snapshot budget needs at least 2 time steps, got "
                f"{total_steps}. Drop the budget to keep every step in memory."
            )
        return MixedCheckpointSchedule(
            total_steps,
            self.resolve_snapshots(total_steps),
            storage=StorageType.RAM,
        )

    def describe(self, total_steps=None):
        """Return a short human-readable description, suitable for logging.

        Parameters
        ----------
        total_steps : int, optional
            When given, resolves the snapshot budget for that step count.

        Returns
        -------
        str
            One-line description.
        """
        if not self.enabled:
            return "checkpointing disabled"
        if self.mode is CheckpointingMode.SINGLE_MEMORY:
            return "checkpointing: all time steps in memory"
        snapshots = (
            self.resolve_snapshots(total_steps)
            if total_steps is not None
            else self.snapshots
        )
        return f"checkpointing: {snapshots} snapshots in RAM, forward recomputed"
