from contextlib import contextmanager
from collections.abc import Mapping

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad

from checkpoint_schedules import (
    CheckpointSchedule,
    MixedCheckpointSchedule,
    SingleMemoryStorageSchedule,
    StorageType,
)
from ..utils.physical_parameters import PhysicalParameters


def _as_list(value: object) -> list:
    """Return one value or collection as a list.

    Parameters
    ----------
    value : object, mapping, PhysicalParameters, list, tuple, or None
        Value to normalize. Anything keyed by name contributes its values,
        and ``None`` produces an empty list.

    Returns
    -------
    list
        Normalized values.
    """
    if value is None:
        return []
    if isinstance(value, (Mapping, PhysicalParameters)):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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

        wave.enable_automated_adjoint(control_parameters=parameters)
        wave.forward_solve()          # the time integrator starts recording
        dJ = wave.gradient_solve()    # one derivative per selected parameter
        wave.automated_adjoint.clear_tape()

    ``gradient_solve`` builds the reduced functional from the recorded tape
    on its first call. Building it here is only needed to hold on to it, or
    to run a Taylor test against it:

    .. code-block:: python

        wave.automated_adjoint.create_reduced_functional(wave.functional_value)
        rate = wave.automated_adjoint.verify_gradient(wave.c)

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
    controls : object, mapping, or iterable, optional
        Fields with respect to which the functional is differentiated. A
        mapping keyed by material parameters labels its controls, so the
        derivatives can be handed back under the same names; anything else is
        taken as unlabeled fields. The wave equation resolves parameter names
        to these fields before constructing the adjoint solver.
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
    gc_timestep_frequency : int, optional
        Run a garbage collection every this many time steps. Reference cycles
        can keep checkpoints alive past the point the schedule intended, so
        collecting periodically lowers the peak. ``None`` (the default)
        disables it.

    Attributes
    ----------
    controls : list
        Controls passed at construction time.
    control_parameter_names : list
        Labels supplied by a control container, or ``None`` for unlabeled
        controls.
    ensemble : firedrake.ensemble.Ensemble or None
        The ensemble communicator used by the reduced functional.
    reduced_functional : firedrake.adjoint.EnsembleReducedFunctional or \
pyadjoint.ReducedFunctional or None
        The reduced functional, created lazily by
        :meth:`create_reduced_functional`.
    """

    def __init__(self, ensemble: object, controls: object = None,
                 checkpointing: bool = False, snapshots: int | None = None,
                 gc_timestep_frequency: int | None = None) -> None:
        self.ensemble = ensemble
        self.reduced_functional = None
        self._tape = None
        if isinstance(controls, (Mapping, PhysicalParameters)):
            # Keyed by material parameter: keep the labels, so the
            # derivatives can be handed back under the same names.
            self.control_parameter_names = list(controls)
            self.controls = list(controls.values())
        else:
            self.controls = _as_list(controls)
            self.control_parameter_names = [None] * len(self.controls)
        self._checkpointing = bool(checkpointing)
        self._snapshots = snapshots
        self._gc_timestep_frequency = gc_timestep_frequency
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
            self._tape.enable_checkpointing(
                self._checkpointing_schedule,
                gc_timestep_frequency=self._gc_timestep_frequency,
            )

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

    def create_reduced_functional(
        self, functional: object, ensemble: object = None,
    ) -> object:
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
        if not self.controls:
            raise ValueError("At least one control is required.")
        controls = [fire_ad.Control(value) for value in self.controls]
        # pyadjoint mirrors the shape it is given: a bare control comes back
        # as a bare derivative, a list as a list. Handing it a one-item list
        # would make every single-control caller unwrap by hand.
        control = controls[0] if len(controls) == 1 else controls

        self.reduced_functional = fire_ad.EnsembleReducedFunctional(
            functional,
            control,
            self.ensemble,
            scatter_control=True,
            tape=self._tape,
        )
        return self.reduced_functional

    def recompute_functional(self, control_value: object) -> object:
        """Re-evaluate the reduced functional at a new control value.

        Parameters
        ----------
        control_value : firedrake.Function or iterable of firedrake.Function
            Controls at which to evaluate the functional.

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
        return self.reduced_functional(_as_list(control_value))

    def compute_gradient(self) -> object:
        """Return the gradient of the functional.

        Computes the gradient via reverse-mode differentiation of the tape and maps
        it back to the primal space (``apply_riesz=True``), yielding a
        :class:`firedrake.Function`. With an ensemble reduced functional the
        gradient is summed across the ensemble.

        Returns
        -------
        firedrake.Function or list of firedrake.Function
            The gradient, or one for each control when there is more than
            one.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional.derivative(apply_riesz=True)

    def compute_derivative(self) -> object:
        """Return the raw derivative of the functional.

        Similar to :meth:`compute_gradient` but without the Riesz map
        (``apply_riesz=False``), so the result lives in the dual space as a
        :class:`firedrake.Cofunction`. The derivative is useful when the
        Full-Waveform Inversion employs scipy optimization routines that require
        derivatives. As with :meth:`compute_gradient`, ``apply_riesz`` requires
        Firedrake ``>= 2026.4``.

        Returns
        -------
        firedrake.Cofunction or list of firedrake.Cofunction
            The derivative, or one for each control when there is more than
            one.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return self.reduced_functional.derivative(apply_riesz=False)

    def label_derivatives(self, derivatives: object) -> PhysicalParameters:
        """Associate computed derivatives with selected physical parameters.

        Parameters
        ----------
        derivatives : object or iterable
            Derivatives positionally matching :attr:`controls`.

        Returns
        -------
        PhysicalParameters
            Derivatives keyed by their selected physical parameter enums.

        Raises
        ------
        ValueError
            If the controls were supplied without physical parameter labels.
        """
        if any(name is None for name in self.control_parameter_names):
            raise ValueError(
                "Physical parameter labels are required for labeled "
                "derivatives.",
            )
        return PhysicalParameters(zip(
            self.control_parameter_names,
            _as_list(derivatives),
        ))

    def verify_gradient(
        self, control_var: object, direction: object = None,
        dJdm: object = None,
    ) -> float:
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
        control_var : firedrake.Function or iterable of firedrake.Function
            Controls about which the gradient is verified.
        direction : firedrake.Function or iterable of firedrake.Function, optional
            Perturbation directions. Each defaults to a constant ``0.01``
            field in the corresponding control's function space.
        dJdm : float or iterable, optional
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
        control_var = _as_list(control_var)
        if direction is None:
            direction = []
            for control in control_var:
                perturbation = fire.Function(control.function_space())
                perturbation.interpolate(0.01)
                direction.append(perturbation)
        else:
            direction = _as_list(direction)
        if len(control_var) != len(direction):
            raise ValueError(
                "Each control requires exactly one perturbation direction.",
            )
        # pyadjoint's ``taylor_test`` expects ``dJdm`` to be the scalar
        # directional derivative ``J'(m)(h)``, not the gradient itself. When a
        # Firedrake ``Function`` (Riesz representer of the gradient) or a
        # ``Cofunction`` (raw derivative) is supplied, reduce it to a scalar by
        # pairing it with the perturbation ``direction``. Otherwise ``eps *
        # dJdm`` inside pyadjoint becomes a UFL expression and the comparison
        # ``min(residuals) < 1E-15`` raises ``UFL conditions cannot be
        # evaluated as bool in a Python context``.
        if dJdm is not None and not isinstance(dJdm, (int, float)):
            derivatives = _as_list(dJdm)
            if len(derivatives) != len(direction):
                raise ValueError(
                    "Each control requires exactly one derivative.",
                )
            directional_derivatives = []
            for derivative, perturbation in zip(derivatives, direction):
                if isinstance(derivative, fire.Function):
                    directional_derivatives.append(
                        fire.assemble(
                            fire.inner(derivative, perturbation) * fire.dx,
                        ),
                    )
                elif isinstance(derivative, fire.Cofunction):
                    directional_derivatives.append(
                        fire.assemble(fire.action(derivative, perturbation)),
                    )
                else:
                    dJdm = None
                    break
            else:
                dJdm = sum(directional_derivatives)
        return taylor_test(self.reduced_functional, control_var, direction, dJdm=dJdm)
