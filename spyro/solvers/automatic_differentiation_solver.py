from contextlib import contextmanager

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad

from .adjoint_checkpointing import CheckpointingConfig


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
    Storing every forward time step on the tape costs memory linear in the
    number of steps. A :class:`~spyro.solvers.adjoint_checkpointing.\
CheckpointingConfig` replaces that with a
    :mod:`checkpoint_schedules` schedule, trading memory for recomputation.
    The caller gives a snapshot budget, not a schedule name; spyro picks
    between the two schedules it supports:

    .. code-block:: python

        wave.enable_automated_adjoint(checkpointing=True, snapshots=20)

    The schedule itself is *not* built here. The mixed schedule needs
    the total number of forward steps, which is only known once ``dt`` and
    ``final_time`` are final - that is, inside the time integrator. The
    configuration is therefore kept and :meth:`start_recording` builds a fresh
    schedule for each forward solve, passing the step count the integrator
    computed. See :mod:`spyro.solvers.adjoint_checkpointing` for details.

    A checkpointed tape is valid for differentiation **only at the control
    value it was recorded at**. Re-evaluating the reduced functional at a
    different control and then differentiating returns a stale gradient
    without raising - pyadjoint's checkpoint managers have already consumed
    the recorded checkpoints. Spyro's supported pattern is one forward solve
    per gradient: :meth:`start_recording` installs a fresh tape and a fresh
    schedule every time the forward loop runs, so an FWI iteration that calls
    ``wave.forward_solve()`` is always correct. Taylor tests are also safe,
    because they only ever re-evaluate the functional *after* the derivative
    has been taken.

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
    checkpointing : CheckpointingConfig or bool, optional
        Checkpointing strategy. A bare ``True`` becomes a
        :class:`CheckpointingConfig` that keeps every step in memory; pass a
        configuration to set a snapshot budget. Defaults to no checkpointing.

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

    def __init__(self, ensemble, controls=None, checkpointing=None):
        self.controls = controls
        self.ensemble = ensemble
        self.reduced_functional = None
        self._tape = None
        self._checkpointing = self._coerce_checkpointing(checkpointing)
        # Schedule built for the tape currently being recorded, kept for
        # introspection; ``None`` whenever the tape carries no checkpointing.
        self._checkpoint_schedule = None
        # Number of adjoint sweeps taken from the current tape, used to reject
        # a second sweep on a schedule that only permits one.
        self._adjoint_sweeps = 0

    @staticmethod
    def _coerce_checkpointing(checkpointing):
        """Normalise the ``checkpointing`` argument into a config object.

        Parameters
        ----------
        checkpointing : CheckpointingConfig, bool or None
            The checkpointing specification.

        Returns
        -------
        CheckpointingConfig
            A configuration; the disabled one when ``checkpointing`` is falsy.
        """
        if isinstance(checkpointing, CheckpointingConfig):
            return checkpointing
        return CheckpointingConfig(enabled=bool(checkpointing))

    @property
    def checkpointing(self):
        """The active :class:`CheckpointingConfig`."""
        return self._checkpointing

    @property
    def checkpoint_schedule(self):
        """Schedule built for the current tape, or ``None`` if not checkpointed.

        A new schedule is created by every :meth:`start_recording` call, so
        this reflects the most recent forward solve rather than a persistent
        object.
        """
        return self._checkpoint_schedule

    @property
    def checkpointing_enabled(self):
        """``True`` when the current tape is managed by a checkpoint schedule."""
        return self._checkpoint_schedule is not None

    def enable_checkpointing(self, snapshots=None):
        """Ask for checkpointing on subsequent forward solves.

        Only the *intent* is recorded here. The schedule is built by
        :meth:`start_recording`, once the forward loop knows how many time
        steps it will run - the mixed schedule cannot be constructed before
        then, and a schedule is consumed by the run that executes it, so one
        has to be built per forward solve anyway.

        Parameters
        ----------
        snapshots : int, optional
            How many checkpointing units to keep in RAM. ``None`` (the
            default) keeps every time step. Spyro picks the backing schedule
            from this budget; see
            :mod:`spyro.solvers.adjoint_checkpointing`.

        Returns
        -------
        CheckpointingConfig
            The configuration that will be used, also stored on
            :attr:`checkpointing`.
        """
        self._checkpointing = CheckpointingConfig(
            enabled=True, snapshots=snapshots
        )
        return self._checkpointing

    def disable_checkpointing(self):
        """Turn checkpointing off for subsequent forward solves.

        The tape currently in hand keeps whatever manager it was given; the
        change takes effect the next time :meth:`start_recording` runs.
        """
        self._checkpointing = CheckpointingConfig()
        return self._checkpointing

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

    def start_recording(self, total_steps=None):
        """Start recording operations on the tape.

        Creates a tape and registers it as the working tape if one does not
        already exist, then enables annotation. Unlike :meth:`fresh_tape`, an
        existing *empty* tape is reused rather than discarded.

        When checkpointing is enabled this method also installs the checkpoint
        manager, which is why ``total_steps`` becomes mandatory: a
        ``MixedCheckpointSchedule`` is built for exactly that many steps. It
        is also why a tape that already holds blocks is replaced by a fresh
        one - pyadjoint refuses to enable checkpointing on a non-empty tape,
        and reusing a consumed tape would silently produce a stale gradient.

        Parameters
        ----------
        total_steps : int, optional
            Number of forward time steps about to be taped. Required when
            checkpointing is enabled, ignored otherwise.

        Returns
        -------
        pyadjoint.Tape
            The active working tape.

        Raises
        ------
        ValueError
            If checkpointing is enabled and ``total_steps`` was not supplied.
        """
        if not self._checkpointing.enabled:
            if self._tape is None:
                self._tape = Tape()
                fire_ad.set_working_tape(self._tape)
            self._checkpoint_schedule = None
            self._adjoint_sweeps = 0
            continue_annotation()
            return self._tape

        if total_steps is None:
            raise ValueError(
                "start_recording() needs total_steps when checkpointing is "
                "enabled: the checkpoint schedule is built for a specific "
                "number of forward time steps. spyro's time integrator passes "
                "it automatically; pass nt explicitly if you drive the tape "
                "yourself."
            )

        # ``Tape.enable_checkpointing`` rejects a tape that already holds
        # blocks, and a schedule is consumed by the run that executes it, so
        # every checkpointed forward solve starts from a clean tape.
        if self._tape is None or len(self._tape.get_blocks()) > 0:
            self._tape = Tape()
            fire_ad.set_working_tape(self._tape)
            self.reduced_functional = None

        self._checkpoint_schedule = self._checkpointing.build_schedule(total_steps)
        self._tape.enable_checkpointing(self._checkpoint_schedule)
        self._adjoint_sweeps = 0
        continue_annotation()
        return self._tape

    def timestep_iterator(self, iterable):
        """Wrap the forward time loop so the tape records step boundaries.

        A checkpoint schedule addresses the tape in time steps, so the forward
        loop has to tell pyadjoint where each step ends. :meth:`Tape.timestepper`
        does that by calling ``Tape.end_timestep`` between iterations.

        Without checkpointing the loop is returned unchanged, keeping the
        non-checkpointed path exactly as it was.

        Parameters
        ----------
        iterable : iterable
            The sequence of time step indices, e.g. ``range(nt)``.

        Returns
        -------
        iterable
            ``iterable`` itself, or a ``TapeTimeStepper`` wrapping it.
        """
        if not self.checkpointing_enabled:
            return iterable
        # ``TapeTimeStepper`` calls ``next()`` on what it is given, so a plain
        # sequence such as ``range(nt)`` has to be turned into an iterator.
        return self._tape.timestepper(iter(iterable))

    def stop_recording(self):
        """Pause annotation, stopping further operations from being taped."""
        pause_annotation()

    def clear_tape(self):
        """Reset the adjoint state.

        Drops the cached reduced functional, tape and checkpoint schedule,
        installs a clean working tape and pauses annotation. Call this between
        independent gradient computations to make sure no stale operations leak
        from one tape onto the next. The checkpointing *configuration* survives,
        so the next forward solve is checkpointed the same way.
        """
        self.reduced_functional = None
        self._tape = None
        self._checkpoint_schedule = None
        self._adjoint_sweeps = 0
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
        RuntimeError
            If the active checkpoint schedule permits a single adjoint sweep
            per tape and one has already been taken.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        self._check_adjoint_sweep_allowed()
        gradient = self.reduced_functional.derivative(apply_riesz=True)
        self._adjoint_sweeps += 1
        return gradient

    def _check_adjoint_sweep_allowed(self):
        """Reject an adjoint sweep the active schedule cannot serve correctly.

        ``MixedCheckpointSchedule`` is an offline schedule: it is exhausted by
        one reverse sweep. pyadjoint does not detect a second one - it returns
        a gradient built from consumed checkpoints, which is wrong by a margin
        large enough to matter but small enough to look plausible. Turning that
        into an exception is the whole point of this check.

        Raises
        ------
        RuntimeError
            If a second adjoint sweep is requested from a tape whose schedule
            permits only one.
        """
        if not self.checkpointing_enabled:
            return
        if self._checkpointing.permits_repeated_adjoints:
            return
        if self._adjoint_sweeps == 0:
            return
        raise RuntimeError(
            "The checkpoint schedule used for a bounded snapshot budget is "
            "offline and permits a single adjoint sweep per tape; a second "
            "gradient taken from this tape would be silently incorrect. Re-run "
            "the forward solve (wave.forward_solve()) to record a fresh tape, "
            "or drop the snapshot budget, which permits repeated adjoint "
            "calculations."
        )

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
        RuntimeError
            If the active checkpoint schedule permits a single adjoint sweep
            per tape and one has already been taken.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        self._check_adjoint_sweep_allowed()
        derivative = self.reduced_functional.derivative(apply_riesz=False)
        self._adjoint_sweeps += 1
        return derivative

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
        if dJdm is None and self.checkpointing_enabled:
            # ``taylor_test`` would call ``reduced_functional.derivative()``
            # itself, bypassing the single-sweep guard on offline schedules.
            # Take the derivative here instead, so the guard applies and the
            # Taylor test only ever re-evaluates the functional afterwards.
            dJdm = self.compute_gradient()
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
