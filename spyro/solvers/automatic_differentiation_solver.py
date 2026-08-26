from contextlib import contextmanager

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad

from checkpoint_schedules import (
    CheckpointSchedule,
    MixedCheckpointSchedule,
    SingleMemoryStorageSchedule,
    StorageType,
)


class AutomatedAdjoint:
    """Automated adjoint driver for spyro using firedrake.adjoint.

    Ensemble (shot) parallelism
    ---------------------------
    Full-waveform inversion sums a per-shot misfit functional

    .. math::

        J(m) = \\sum_{i=1}^{N} J_i(m),

    where :math:`m` is the control (velocity model) and each :math:`J_i` is the
    functional of a single source. Because differentiation is linear over the
    sum,

    .. math::

        \\frac{dJ}{dm} = \\sum_{i=1}^{N} \\frac{dJ_i}{dm}.

    Under spyro's ensemble parallelism every ensemble member runs the forward
    solve for its own subset of sources and therefore records only its local
    functional :math:`J_i` on its own tape. To turn those local functionals
    into the global :math:`J` (and its gradient) the reduced functional is built
    as a :class:`firedrake.adjoint.EnsembleReducedFunctional`, which evaluates
    each :math:`J_i` and ``dJ_i/dm`` simultaneously and then performs an
    ``allreduce`` over the ensemble communicator to sum them. That ensemble
    communicator is supplied by the owning wave solver as ``wave.comm`` and is
    passed here through the ``ensemble`` argument.

    When no ensemble is provided (``ensemble=None``) the class falls back to a
    plain :class:`pyadjoint.ReducedFunctional`, which is equivalent to the
    single-ensemble-member case.

    Typical workflow
    ----------------
    .. code-block:: python

        wave.enable_automated_adjoint()   # builds AutomatedAdjoint(wave.comm)
        with wave.automated_adjoint.fresh_tape():
            wave.forward_solve()          # forward run recorded on the tape
        wave.automated_adjoint.create_reduced_functional(wave.functional_value)
        dJ = wave.automated_adjoint.compute_gradient()
        rate = wave.automated_adjoint.verify_gradient(wave.c)  # Taylor test

    Checkpointing
    -------------
    Storing every forward step on the tape costs memory linear in the number
    of steps. Passing ``checkpointing=True`` hands the tape to a
    :mod:`checkpoint_schedules` schedule instead. Spyro uses two of them and
    picks between them from the snapshot budget:

    ``snapshots=None``
        :class:`~checkpoint_schedules.SingleMemoryStorageSchedule` - every
        adjoint dependency stays in RAM, nothing is recomputed.
    ``snapshots=N``
        :class:`~checkpoint_schedules.MixedCheckpointSchedule` - only ``N``
        checkpoints are kept in RAM and the forward is recomputed in between,
        turning :math:`O(n_t)` memory into :math:`O(N)`.

    The schedule is built by :meth:`start_recording`, not here.

    Parameters
    ----------
    controls : firedrake.Function, optional
        The control with respect to which the functional is differentiated.
        It is wrapped in a :class:`pyadjoint.Control` when the reduced functional is
        created.
    ensemble : firedrake.ensemble.Ensemble, optional
        The Firedrake ensemble communicator used to sum the per-shot
        functionals and gradients across ensemble members. In practice this is
        ``wave.comm``. If ``None``, a non-ensemble
        :class:`pyadjoint.ReducedFunctional` is used instead.
    checkpointing : bool, optional
        Whether to manage the tape with a checkpoint schedule. Defaults to
        ``False``, which keeps the tape exactly as it was before checkpointing
        existed.
    snapshots : int, optional
        How many checkpoints to keep in RAM, which is also what selects the
        schedule. ``None`` (the default) keeps every step.

    Attributes
    ----------
    controls : firedrake.Function
        The control passed at construction time.
    ensemble : firedrake.ensemble.Ensemble or None
        The ensemble communicator used by the reduced functional.
    reduced_functional : firedrake.adjoint.EnsembleReducedFunctional or \
pyadjoint.ReducedFunctional or None
        The reduced functional, created lazily by
        :meth:`create_reduced_functional`.
    """

    def __init__(self, ensemble: fire.Ensemble | None,
                 controls: fire.Function | None = None,
                 checkpointing: bool = False, snapshots: int | None = None):
        self.controls = controls
        self.ensemble = ensemble
        self.reduced_functional = None
        self._tape = None
        self._checkpointing = bool(checkpointing)
        self._snapshots = snapshots
        # Schedule of the tape being recorded; ``None`` when not checkpointed.
        self._checkpointing_schedule = None

    @property
    def checkpointing_schedule(self) -> CheckpointSchedule | None:
        """Schedule of the current tape, or ``None`` if it is not checkpointed.

        Returns
        -------
        checkpoint_schedules.CheckpointSchedule or None
            The schedule installed by the most recent :meth:`start_recording`.
        """
        return self._checkpointing_schedule

    @property
    def checkpointing_enabled(self) -> bool:
        """Whether the current tape is managed by a checkpoint schedule.

        Returns
        -------
        bool
            ``True`` once :meth:`start_recording` has installed a schedule on
            the tape, ``False`` when checkpointing is off.
        """
        return self._checkpointing_schedule is not None

    def _build_schedule(self, total_steps: int) -> CheckpointSchedule:
        """Build a schedule for a forward run of ``total_steps`` steps.

        Parameters
        ----------
        total_steps : int
            The total number of time steps used in the forward solver.

        Returns
        -------
        checkpoint_schedules.CheckpointSchedule
            A single-use schedule.
        """
        if self._snapshots is None:
            return SingleMemoryStorageSchedule()
        # A mixed schedule cannot hold more checkpoints than it has steps to
        # place them in.
        snapshots = max(1, min(self._snapshots, total_steps - 1))
        return MixedCheckpointSchedule(
            total_steps, snapshots, storage=StorageType.RAM
        )

    @contextmanager
    def fresh_tape(self):
        """Context manager that records the forward solve on a brand new tape.

        Clears any previous tape, installs a fresh :class:`pyadjoint.Tape` as
        the working tape and turns annotation on for the duration of the
        ``with`` block. Annotation is always paused again on exit, even if an
        exception is raised, so the caller cannot accidentally leave taping
        enabled.

        Yields
        ------
        pyadjoint.Tape
            The freshly created working tape.
        """
        self.clear_tape()
        self._tape = Tape()
        fire_ad.set_working_tape(self._tape)
        continue_annotation()
        try:
            yield self._tape
        finally:
            pause_annotation()

    def start_recording(self, total_steps: int | None = None) -> Tape:
        """Install a fresh tape and start recording operations on it.

        Parameters
        ----------
        total_steps : int, optional
            The total number of time steps used in the forward solver.
            Required when checkpointing is enabled, ignored otherwise.

        Returns
        -------
        pyadjoint.Tape
            The active working tape.

        Raises
        ------
        ValueError
            If checkpointing is enabled and ``total_steps`` was not supplied.
        """
        if self._checkpointing and total_steps is None:
            raise ValueError(
                "start_recording() needs total_steps when checkpointing "
                "is enabled: the schedule is built for a specific number "
                "of forward time steps. Spyro's time integrator passes it "
                "automatically."
            )

        self._tape = Tape()
        fire_ad.set_working_tape(self._tape)
        self.reduced_functional = None
        self._checkpointing_schedule = None

        if self._checkpointing:
            self._checkpointing_schedule = self._build_schedule(total_steps)
            self._tape.enable_checkpointing(self._checkpointing_schedule)

        continue_annotation()
        return self._tape

    def end_timestep(self) -> None:
        """Mark the end of one forward time step on the tape.

        A no-op when the tape is not checkpointed, since only a schedule cares
        where the time steps are.

        Returns
        -------
        None
            The tape is advanced in place.
        """
        if self._checkpointing_schedule is not None:
            self._tape.end_timestep()

    def stop_recording(self):
        """Pause annotation, stopping further operations from being taped."""
        pause_annotation()

    def clear_tape(self):
        """Reset the adjoint state.

        Drops the cached reduced functional, tape and checkpoint schedule,
        installs a clean working tape and pauses annotation. Call this between
        independent gradient computations to make sure no stale operations leak
        from one tape onto the next. The checkpointing *settings* survive, so
        the next forward solve is checkpointed the same way.
        """
        self.reduced_functional = None
        self._tape = None
        self._checkpointing_schedule = None
        fire_ad.set_working_tape(Tape())
        pause_annotation()

    def create_reduced_functional(self, functional, ensemble=None):
        """Build the reduced functional for the recorded forward problem.

        The reduced functional ties the (local) functional value to the control
        and the recorded tape. When an ensemble communicator is available the
        functional is wrapped in a
        :class:`firedrake.adjoint.EnsembleReducedFunctional`, so that calling or
        differentiating it transparently sums the per-shot functionals and
        gradients across the ensemble. Otherwise a plain
        :class:`pyadjoint.ReducedFunctional` is used.

        Parameters
        ----------
        functional : pyadjoint.AdjFloat
            The (per-ensemble-member) functional value recorded on the tape,
            e.g. ``wave.functional_value``.
        ensemble : firedrake.ensemble.Ensemble, optional
            Ensemble communicator to use. Defaults to the ensemble supplied at
            construction time (``self.ensemble``, i.e. ``wave.comm``).

        Returns
        -------
        firedrake.adjoint.EnsembleReducedFunctional or pyadjoint.ReducedFunctional
            The reduced functional, also stored on
            :attr:`reduced_functional`.
        """
        control = fire_ad.Control(self.controls)

        self.reduced_functional = fire_ad.EnsembleReducedFunctional(
            functional,
            control,
            self.ensemble,
            scatter_control=True,
            tape=self._tape,
        )
        return self.reduced_functional

    def recompute_functional(self, control_value):
        """Re-evaluate the reduced functional at a new control value.

        Parameters
        ----------
        control_value : firedrake.Function
            The control at which to evaluate the functional.

        Returns
        -------
        pyadjoint.AdjFloat
            The functional value. With an ensemble reduced functional this is
            the sum over all ensemble members.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional(control_value)

    def compute_gradient(self):
        """Return the gradient of the functional.

        Computes the gradient via reverse-mode differentiation of the tape and maps
        it back to the primal space (``apply_riesz=True``), yielding a
        :class:`firedrake.Function`. With an ensemble reduced functional the
        gradient is summed across the ensemble.

        Returns
        -------
        firedrake.Function
            The gradient of the functional with respect to the control.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional.derivative(apply_riesz=True)

    def compute_derivative(self):
        """Return the raw derivative of the functional.

        Similar to :meth:`compute_gradient` but without the Riesz map
        (``apply_riesz=False``), so the result lives in the dual space as a
        :class:`firedrake.Cofunction`. The derivative is useful when the
        Full-Waveform Inversion employs scipy optimization routines that require
        derivatives. As with :meth:`compute_gradient`, ``apply_riesz`` requires
        Firedrake ``>= 2026.4``.

        Returns
        -------
        firedrake.Cofunction
            The derivative of the functional with respect to the control.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional.derivative(apply_riesz=False)

    def verify_gradient(self, control_var, direction=None, dJdm=None):
        """Run a Taylor test to validate the automated-adjoint gradient.

        Performs pyadjoint's :func:`~pyadjoint.taylor_test`, which perturbs the
        control by ``h * direction`` for a sequence of decreasing ``h`` and
        checks that the first-order Taylor remainder converges at second order.
        A returned rate close to ``2`` indicates a correct gradient. When the
        reduced functional is an
        :class:`firedrake.adjoint.EnsembleReducedFunctional`, the Taylor test
        transparently uses the ensemble-summed functional and gradient.

        Parameters
        ----------
        control_var : firedrake.Function
            The control about which the gradient is verified.
        direction : firedrake.Function, optional
            Perturbation direction. Defaults to a constant ``0.01`` field in the
            control's function space.
        dJdm : float, firedrake.Function, or firedrake.Cofunction, optional
            The directional derivative ``J'(m)(direction)``. pyadjoint expects a
            scalar here, so if a gradient ``Function`` (Riesz representer) or a
            ``Cofunction`` (raw derivative) is supplied it is first paired with
            ``direction`` to reduce it to a scalar. If left as ``None`` (the
            recommended choice under ensemble parallelism) pyadjoint computes
            the directional derivative itself from the reduced functional, which
            keeps the ensemble reduction consistent.

        Returns
        -------
        float
            The estimated Taylor convergence rate (≈ 2 for a correct gradient).

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        if direction is None:
            direction = fire.Function(control_var.function_space())
            direction.interpolate(0.01)
        # pyadjoint's ``taylor_test`` expects ``dJdm`` to be the scalar
        # directional derivative ``J'(m)(h)``, not the gradient itself. When a
        # Firedrake ``Function`` (Riesz representer of the gradient) or a
        # ``Cofunction`` (raw derivative) is supplied, reduce it to a scalar by
        # pairing it with the perturbation ``direction``. Otherwise ``eps *
        # dJdm`` inside pyadjoint becomes a UFL expression and the comparison
        # ``min(residuals) < 1E-15`` raises ``UFL conditions cannot be
        # evaluated as bool in a Python context``.
        if dJdm is not None and not isinstance(dJdm, (int, float)):
            if isinstance(dJdm, fire.Function):
                dJdm = fire.assemble(
                    fire.inner(dJdm, direction) * fire.dx
                )
            elif isinstance(dJdm, fire.Cofunction):
                # Apply the cofunction to the direction (duality pairing).
                dJdm = fire.assemble(fire.action(dJdm, direction))
            else:
                # Unknown type, fall back to pyadjoint's internal computation.
                dJdm = None
        return taylor_test(self.reduced_functional, control_var, direction, dJdm=dJdm)
