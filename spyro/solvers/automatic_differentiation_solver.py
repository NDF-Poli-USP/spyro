from contextlib import contextmanager
from collections.abc import Mapping
from enum import Enum

from pyadjoint import Tape, continue_annotation, pause_annotation, taylor_test

import firedrake as fire
import firedrake.adjoint as fire_ad

from ..utils.physical_parameters import (ELASTIC_PARAMETERIZATIONS,
                                         PhysicalParameters)
from ..utils.typing import ElasticMaterialParameter


def _as_list(value: object) -> list:
    """Return one value or collection as a list.

    Parameters
    ----------
    value : object, mapping, list, tuple, or None
        Value to normalize. Mappings contribute their values and ``None``
        produces an empty list.

    Returns
    -------
    list
        Normalized values.
    """
    if value is None:
        return []
    if isinstance(value, Mapping) or hasattr(value, "items"):
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

        wave.enable_automated_adjoint()
        wave.automated_adjoint.set_control_parameters(parameters)
        with wave.automated_adjoint.fresh_tape():
            wave.forward_solve()          # forward run recorded on the tape
        wave.automated_adjoint.create_reduced_functional(wave.functional_value)
        dJ = wave.automated_adjoint.compute_gradient()
        rate = wave.automated_adjoint.verify_gradient(wave.c)  # Taylor test

    Parameters
    ----------
    controls : object or iterable, optional
        Existing fields with respect to which the functional is differentiated.
        This low-level argument is used when no wave is supplied.
    ensemble : firedrake.ensemble.Ensemble, optional
        The Firedrake ensemble communicator used to sum the per-shot
        functionals and gradients across ensemble members. In practice this is
        ``wave.comm``. If ``None``, a non-ensemble
        :class:`pyadjoint.ReducedFunctional` is used instead.
    wave : Wave, optional
        Wave equation whose physical parameters are available for selection.
        When supplied, the adjoint solver selects the independent parameters
        of the active physical parameterization by default.

    Attributes
    ----------
    controls : list
        Controls passed at construction time.
    control_parameter_names : list
        Labels supplied by a control container, or ``None`` for unlabeled
        controls.
    wave : Wave or None
        Wave equation providing physical fields to the adjoint solver.
    ensemble : firedrake.ensemble.Ensemble or None
        The ensemble communicator used by the reduced functional.
    reduced_functional : firedrake.adjoint.EnsembleReducedFunctional or \
pyadjoint.ReducedFunctional or None
        The reduced functional, created lazily by
        :meth:`create_reduced_functional`.
    """

    def __init__(
        self, ensemble: object, controls: object = None, wave: object = None,
    ) -> None:
        self.wave = wave
        self.controls = []
        self.control_parameter_names = []
        self.ensemble = ensemble
        self.reduced_functional = None
        self._tape = None
        if wave is not None:
            self.set_control_parameters()
            return
        self._set_controls(controls)

    def _set_controls(self, controls: object) -> None:
        """Store labeled or unlabeled control fields as ordered lists.

        Parameters
        ----------
        controls : object or iterable
            Control fields, optionally exposed through an ``items`` method.

        Returns
        -------
        None
        """
        try:
            control_items = list(controls.items())
        except AttributeError:
            self.controls = _as_list(controls)
            self.control_parameter_names = [None] * len(self.controls)
        else:
            self.control_parameter_names = [name for name, _ in control_items]
            self.controls = [value for _, value in control_items]

    def set_control_parameters(self, parameters: object = None) -> None:
        """Select physical parameters for automated differentiation.

        Control selection belongs to the adjoint solver. For isotropic elastic
        waves, the selected parameters must form a non-empty subset of either
        the Lame or velocity parameterization. Selecting the other family asks
        the wave equation to change its physical parameterization before tape
        recording.

        Parameters
        ----------
        parameters : enum.Enum or iterable of enum.Enum, optional
            Physical parameters selected as controls. ``None`` selects every
            independent parameter of the active physical parameterization.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If recording has already started.
        ValueError
            If no wave is attached or the selection is empty or inconsistent.
        TypeError
            If a selected name is not a material-parameter enum member.
        """
        if self.wave is None:
            raise ValueError(
                "A wave equation is required to select control parameters.",
            )
        if self._tape is not None:
            raise RuntimeError(
                "Control parameters must be selected before tape recording.",
            )

        physical_parameters = self.wave.physical_parameters
        if all(
            isinstance(name, ElasticMaterialParameter)
            for name in physical_parameters
        ):
            selected = self._select_elastic_parameters(parameters)
        else:
            selected = self._select_independent_parameters(parameters)
        self._set_controls(selected)
        self.reduced_functional = None

    def _select_independent_parameters(
        self, parameters: object,
    ) -> PhysicalParameters:
        """Resolve controls for a wave without coupled parameterizations.

        Parameters
        ----------
        parameters : enum.Enum or iterable of enum.Enum, optional
            Requested physical parameter names.

        Returns
        -------
        PhysicalParameters
            Selected names mapped to independent physical fields.
        """
        physical_parameters = self.wave.physical_parameters
        if parameters is None:
            selected_names = [
                name for name, value in physical_parameters.items()
                if isinstance(value, fire.Function)
            ]
        elif isinstance(parameters, Enum):
            selected_names = [parameters]
        else:
            selected_names = list(parameters)

        if not selected_names:
            raise ValueError("At least one control parameter is required.")
        if not all(isinstance(name, Enum) for name in selected_names):
            raise TypeError(
                "Control parameters must be material-parameter enum members.",
            )
        unknown = set(selected_names) - set(physical_parameters)
        if unknown:
            names = ", ".join(name.value for name in unknown)
            raise ValueError(
                f"Control parameters {{{names}}} are not physical "
                "parameters of this wave equation.",
            )

        selected = PhysicalParameters()
        for name in physical_parameters:
            if name not in selected_names:
                continue
            field = physical_parameters[name]
            if not isinstance(field, fire.Function):
                raise TypeError(
                    f"'{name.value}' is a dependent physical parameter and "
                    "cannot be used directly as a control.",
                )
            selected.add(name, field)
        return selected

    def _select_elastic_parameters(
        self, parameters: object,
    ) -> PhysicalParameters:
        """Resolve an independent isotropic-elastic control subset.

        Parameters
        ----------
        parameters : ElasticMaterialParameter or iterable, optional
            Requested elastic physical parameter names.

        Returns
        -------
        PhysicalParameters
            Selected names mapped to independent physical fields.
        """
        current = self.wave._physical_parameterization
        if parameters is None:
            selected_names = list(ELASTIC_PARAMETERIZATIONS[current])
        elif isinstance(parameters, ElasticMaterialParameter):
            selected_names = [parameters]
        else:
            selected_names = list(parameters)

        if not selected_names:
            raise ValueError("At least one elastic control parameter is required.")
        if not all(
            isinstance(name, ElasticMaterialParameter)
            for name in selected_names
        ):
            raise TypeError(
                "Elastic controls must be ElasticMaterialParameter enum "
                "members.",
            )
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("Elastic control parameters must be unique.")

        names = set(selected_names)
        candidates = [
            parameterization
            for parameterization, family in ELASTIC_PARAMETERIZATIONS.items()
            if names <= set(family)
        ]
        if not candidates:
            formatted = ", ".join(name.value for name in selected_names)
            raise ValueError(
                "Elastic controls must be a subset of either "
                "{density, lambda, mu} or "
                "{density, p_wave_velocity, s_wave_velocity}; got "
                f"{{{formatted}}}.",
            )
        target = current if current in candidates else candidates[0]
        self.wave._set_physical_parameterization(target)

        selected = PhysicalParameters()
        for parameter in ELASTIC_PARAMETERIZATIONS[target]:
            if parameter in names:
                selected.add(
                    parameter,
                    self.wave.physical_parameters[parameter],
                )
        return selected

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
        control = [fire_ad.Control(value) for value in self.controls]

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

    def compute_gradient(self) -> list:
        """Return the gradient of the functional.

        Computes the gradient via reverse-mode differentiation of the tape and maps
        it back to the primal space (``apply_riesz=True``), yielding a
        :class:`firedrake.Function`. With an ensemble reduced functional the
        gradient is summed across the ensemble.

        Returns
        -------
        list of firedrake.Function
            One gradient for each control.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return _as_list(
            self.reduced_functional.derivative(apply_riesz=True),
        )

    def compute_derivative(self) -> list:
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
            One derivative for each control.

        Raises
        ------
        ValueError
            If the reduced functional has not been created yet.
        """
        if self.reduced_functional is None:
            raise ValueError("Reduced functional not created.")
        return _as_list(
            self.reduced_functional.derivative(apply_riesz=False),
        )

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
