from collections.abc import Mapping
from contextlib import contextmanager

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad


def _as_list(value):
    """Return ``value`` as a list, one entry per control.

    Controls, perturbation directions and derivatives are handled uniformly as
    per-control sequences, but the single-control APIs (the acoustic velocity
    model, in practice) pass bare objects, and the labeled collections used
    here are mappings. This keeps every spelling valid without duplicating the
    branch at each call site.

    Parameters
    ----------
    value : object, mapping, list, tuple, or None
        Value to normalize. Mappings contribute their values, ``None`` yields
        an empty list.

    Returns
    -------
    list
        ``value`` as a list, or a one-item list wrapping it.

    Examples
    --------
    ``_as_list(control)``, ``_as_list([control])`` and
    ``_as_list({"velocity": control})`` all return ``[control]``.
    """
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_controls(controls):
    """Return a control collection as an ordered ``{label: control}`` mapping.

    Parameters
    ----------
    controls : mapping, object, or None
        Labeled controls, a single unlabeled control, or ``None`` for an empty
        selection.

    Returns
    -------
    dict
        Mapping preserving the caller's order. A bare control is labeled
        ``"control"``.
    """
    if controls is None:
        return {}
    if isinstance(controls, Mapping):
        return dict(controls)
    return {"control": controls}


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
    controls : mapping or firedrake.Function, optional
        The controls with respect to which the functional is differentiated,
        as an ordered ``{label: Function}`` mapping. Each value is wrapped in a
        :class:`pyadjoint.Control` when the reduced functional is created, and
        the labels are carried through to the computed derivatives. Labels are
        opaque here: the owning solver chooses them, typically a material
        parameter enum. A bare control is accepted and labeled ``"control"``.
    ensemble : firedrake.ensemble.Ensemble, optional
        The Firedrake ensemble communicator used to sum the per-shot
        functionals and gradients across ensemble members. In practice this is
        ``wave.comm``. If ``None``, a non-ensemble
        :class:`pyadjoint.ReducedFunctional` is used instead.

    Attributes
    ----------
    controls : dict
        The labeled controls being differentiated. This is the single place
        where the inversion's control selection is stored; the solver owns the
        fields themselves, not the choice of which are inverted for.
    ensemble : firedrake.ensemble.Ensemble or None
        The ensemble communicator used by the reduced functional.
    reduced_functional : firedrake.adjoint.EnsembleReducedFunctional or \
pyadjoint.ReducedFunctional or None
        The reduced functional, created lazily by
        :meth:`create_reduced_functional`.
    """

    def __init__(self, ensemble, controls=None):
        self.controls = _as_controls(controls)
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
        control = [
            fire_ad.Control(value) for value in self.controls.values()
        ]

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
        dict
            The gradients of the functional, labeled like :attr:`controls`.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        return self._labeled_derivative(apply_riesz=True)

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
        dict
            The derivatives of the functional, labeled like :attr:`controls`.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        return self._labeled_derivative(apply_riesz=False)

    def _labeled_derivative(self, apply_riesz):
        """Differentiate the reduced functional and label the result.

        pyadjoint returns derivatives as a list positionally matching the
        controls it was built with, so zipping with :attr:`controls` restores
        the labels the owning solver assigned.

        Parameters
        ----------
        apply_riesz : bool
            Whether to map the derivative back to the primal space.

        Returns
        -------
        dict
            One derivative per control, labeled like :attr:`controls`.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        derivatives = self.reduced_functional.derivative(
            apply_riesz=apply_riesz,
        )
        return dict(zip(self.controls, _as_list(derivatives)))

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
        control_var : firedrake.Function, sequence, or mapping
            Control or controls about which the gradient is verified. A bare
            control is normalized to a one-item list, and a labeled mapping —
            such as :attr:`controls` itself — contributes its values in order.
        direction : firedrake.Function or sequence of firedrake.Function, optional
            Perturbation direction or directions. A single direction is
            normalized to a one-item list. Defaults to constant ``0.01`` fields
            in the controls' function spaces.
        dJdm : float, firedrake.Function, firedrake.Cofunction, sequence, or \
mapping, optional
            The directional derivative ``J'(m)(direction)``. pyadjoint expects a
            scalar here, so gradient ``Function`` objects (Riesz representers)
            and ``Cofunction`` objects (raw derivatives) are first paired with
            the corresponding ``direction`` and summed. The labeled mapping
            returned by :meth:`compute_gradient` is accepted directly. If left
            as ``None`` (the recommended choice under ensemble parallelism)
            pyadjoint computes the directional derivative itself from the
            reduced functional, which keeps the ensemble reduction consistent.

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
            direction = [
                fire.Function(control.function_space()).interpolate(0.01)
                for control in control_var
            ]
        else:
            direction = _as_list(direction)

        if not isinstance(dJdm, (int, float, type(None))):
            dJdm = self._directional_derivative(_as_list(dJdm), direction)
        return taylor_test(self.reduced_functional, control_var, direction, dJdm=dJdm)

    @staticmethod
    def _directional_derivative(derivatives, direction):
        """Contract per-control derivatives with the perturbation directions.

        pyadjoint's ``taylor_test`` expects ``dJdm`` to be the scalar
        directional derivative :math:`J'(m)(h) = \\sum_i \\langle dJ/dm_i,
        h_i \\rangle`, not the derivative objects themselves. Passing a
        ``Function`` or ``Cofunction`` straight through would make ``eps *
        dJdm`` a UFL expression inside pyadjoint, and the subsequent
        ``min(residuals) < 1E-15`` comparison would raise "UFL conditions
        cannot be evaluated as bool in a Python context".

        Parameters
        ----------
        derivatives : list
            One derivative per control. ``Function`` entries are the Riesz
            representers of the gradient and are paired with the direction in
            :math:`L^2`; ``Cofunction`` entries are dual objects and are
            applied to the direction through the duality pairing.
        direction : list of firedrake.Function
            Perturbation direction of each control.

        Returns
        -------
        float or None
            The directional derivative, or ``None`` when an entry has an
            unsupported type, which tells pyadjoint to compute it itself.

        Raises
        ------
        ValueError
            If the two sequences have different lengths.
        """
        if len(derivatives) != len(direction):
            raise ValueError(
                "The derivative and direction must have the same length."
            )

        directional_derivative = 0.0
        for derivative, perturbation in zip(derivatives, direction):
            if isinstance(derivative, fire.Function):
                directional_derivative += fire.assemble(
                    fire.inner(derivative, perturbation) * fire.dx
                )
            elif isinstance(derivative, fire.Cofunction):
                directional_derivative += fire.assemble(
                    fire.action(derivative, perturbation)
                )
            else:
                # Unknown type, fall back to pyadjoint's internal computation.
                return None
        return directional_derivative
