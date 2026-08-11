from contextlib import contextmanager

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad


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
    controls : sequence of firedrake.Function, optional
        The controls with respect to which the functional is differentiated.
        Each item is wrapped in a :class:`pyadjoint.Control` when the reduced
        functional is created. A single value is accepted for backward
        compatibility and normalized to a one-item list.
    ensemble : firedrake.ensemble.Ensemble, optional
        The Firedrake ensemble communicator used to sum the per-shot
        functionals and gradients across ensemble members. In practice this is
        ``wave.comm``. If ``None``, a non-ensemble
        :class:`pyadjoint.ReducedFunctional` is used instead.

    Attributes
    ----------
    controls : list of firedrake.Function
        The controls passed at construction time.
    ensemble : firedrake.ensemble.Ensemble or None
        The ensemble communicator used by the reduced functional.
    reduced_functional : firedrake.adjoint.EnsembleReducedFunctional or \
pyadjoint.ReducedFunctional or None
        The reduced functional, created lazily by
        :meth:`create_reduced_functional`.
    """

    def __init__(self, ensemble, controls=None):
        if controls is None:
            controls = []
        elif not isinstance(controls, (list, tuple)):
            controls = [controls]
        self.controls = list(controls)
        self.ensemble = ensemble
        self.reduced_functional = None
        self._tape = None

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
            self._tape = Tape()
            fire_ad.set_working_tape(self._tape)
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
        if not self.controls:
            raise ValueError("At least one control is required.")
        control = [fire_ad.Control(value) for value in self.controls]

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
        control_value : sequence of firedrake.Function
            The controls at which to evaluate the functional.

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
        list of firedrake.Function
            The gradients of the functional with respect to the controls.

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
        list of firedrake.Cofunction
            The derivatives of the functional with respect to the controls.

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
        control_var : firedrake.Function or sequence of firedrake.Function
            Control or controls about which the gradient is verified. A single
            control is normalized to a one-item list.
        direction : firedrake.Function or sequence of firedrake.Function, optional
            Perturbation direction or directions. A single direction is
            normalized to a one-item list. Defaults to constant ``0.01`` fields
            in the controls' function spaces.
        dJdm : float, firedrake.Function, firedrake.Cofunction, or sequence, optional
            The directional derivative ``J'(m)(direction)``. pyadjoint expects a
            scalar here, so if a gradient ``Function`` (Riesz representer) or a
            ``Cofunction`` (raw derivative) is supplied it is first paired with
            the corresponding ``direction`` to reduce it to a scalar. A single
            object is accepted for a one-control problem. If left as ``None``
            (the recommended choice under ensemble parallelism) pyadjoint
            computes the directional derivative itself from the reduced
            functional, which keeps the ensemble reduction consistent.

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

        if isinstance(control_var, (list, tuple)):
            control_var = list(control_var)
        else:
            control_var = [control_var]
        if direction is not None:
            if isinstance(direction, (list, tuple)):
                direction = list(direction)
            else:
                direction = [direction]

        if direction is None:
            direction = [
                fire.Function(control.function_space()).interpolate(0.01)
                for control in control_var
            ]
        # pyadjoint's ``taylor_test`` expects ``dJdm`` to be the scalar
        # directional derivative ``J'(m)(h)``, not the gradient itself. When a
        # Firedrake ``Function`` (Riesz representer of the gradient) or a
        # ``Cofunction`` (raw derivative) is supplied, reduce it to a scalar by
        # pairing it with the perturbation ``direction``. Otherwise ``eps *
        # dJdm`` inside pyadjoint becomes a UFL expression and the comparison
        # ``min(residuals) < 1E-15`` raises ``UFL conditions cannot be
        # evaluated as bool in a Python context``.
        if isinstance(dJdm, (fire.Function, fire.Cofunction)):
            if len(direction) != 1:
                raise ValueError(
                    "A single derivative requires exactly one direction."
                )
            dJdm = [dJdm]

        if isinstance(dJdm, (list, tuple)):
            if len(dJdm) != len(direction):
                raise ValueError(
                    "The derivative and direction must have the same length."
                )
            directional_derivative = 0.0
            for derivative, perturbation in zip(dJdm, direction):
                if isinstance(derivative, fire.Function):
                    directional_derivative += fire.assemble(
                        fire.inner(derivative, perturbation) * fire.dx
                    )
                elif isinstance(derivative, fire.Cofunction):
                    directional_derivative += fire.assemble(
                        fire.action(derivative, perturbation)
                    )
                else:
                    directional_derivative = None
                    break
            dJdm = directional_derivative
        elif dJdm is not None and not isinstance(dJdm, (int, float)):
            # Unknown type, fall back to pyadjoint's internal computation.
            dJdm = None
        return taylor_test(self.reduced_functional, control_var, direction, dJdm=dJdm)
