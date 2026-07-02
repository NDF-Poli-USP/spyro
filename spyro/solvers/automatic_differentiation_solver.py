from contextlib import contextmanager

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad
from ..tools.checkpointing import SpyroCheckpointManager


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
        If ``True``, enable Spyro checkpointing on each fresh tape before
        any forward blocks are recorded.
    checkpoint_schedule : checkpoint_schedules.schedule, optional
        Checkpoint schedule instance to pass to
        :class:`spyro.tools.checkpointing.SpyroCheckpointManager`.
    checkpoint_form : object, optional
        Metadata reserved for future form-aware checkpoint validation.
    checkpoint_recompute_strategy : spyro.tools.checkpointing.RecomputeStrategy, optional
        Strategy used by :class:`spyro.tools.checkpointing.SpyroCheckpointManager`
        during checkpoint replay.

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

    def __init__(
        self,
        controls=None,
        ensemble=None,
        checkpointing=False,
        checkpoint_schedule=None,
        checkpoint_form=None,
        checkpoint_recompute_strategy=None,
    ):
        if checkpointing and checkpoint_schedule is None:
            raise ValueError(
                "checkpoint_schedule must be provided when checkpointing=True."
            )
        self.controls = controls
        self.ensemble = ensemble
        self.reduced_functional = None
        self._tape = None
        self.checkpointing = checkpointing
        self.checkpoint_schedule = checkpoint_schedule
        self.checkpoint_form = checkpoint_form
        self.checkpoint_recompute_strategy = checkpoint_recompute_strategy

    def _new_tape(self):
        """Create a tape and install checkpointing before recording blocks."""
        if self.checkpointing and self.checkpoint_schedule is None:
            raise ValueError(
                "checkpoint_schedule must be provided when checkpointing=True."
            )
        tape = Tape()
        if self.checkpointing:
            tape._checkpoint_manager = SpyroCheckpointManager(
                self.checkpoint_schedule,
                tape,
                recompute_strategy=self.checkpoint_recompute_strategy,
            )
        fire_ad.set_working_tape(tape)
        return tape

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
        self._tape = self._new_tape()
        continue_annotation()
        try:
            yield self._tape
        finally:
            pause_annotation()

    def start_recording(self):
        """Start recording operations on the tape.

        Creates a tape and registers it as the working tape if one does not
        already exist, then enables annotation. Unlike :meth:`fresh_tape`, an
        existing tape is reused rather than discarded.

        Returns
        -------
        pyadjoint.Tape
            The active working tape.
        """
        if self._tape is None:
            self._tape = self._new_tape()
        continue_annotation()
        return self._tape

    def stop_recording(self):
        """Pause annotation, stopping further operations from being taped."""
        pause_annotation()

    def clear_tape(self):
        """Reset the adjoint state.

        Drops the cached reduced functional and tape, installs a clean working
        tape and pauses annotation. Call this between independent gradient
        computations to make sure no stale operations leak from one tape onto
        the next.
        """
        self.reduced_functional = None
        self._tape = None
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
        if ensemble is not None:
            self.reduced_functional = fire_ad.EnsembleReducedFunctional(
                functional,
                control,
                ensemble,
                scatter_control=True,
                tape=self._tape,
            )
        else:
            self.reduced_functional = fire_ad.ReducedFunctional(
                functional,
                control,
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
