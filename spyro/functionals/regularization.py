"""Spatial regularization of physical material parameters."""

from collections.abc import Mapping
from enum import Enum
import math
from numbers import Real

import firedrake as fire
from pyadjoint import stop_annotating
from ufl import H1


class H1Regularization:
    r"""Weighted squared H1 seminorm of physical model deviations.

    Parameters
    ----------
    weights : mapping of enum.Enum to float or str
        Nonnegative weights for the material parameters to penalize, or
        ``"auto"`` for a weight the inversion chooses from the starting
        model; see ``gradient_fraction``.
    references : mapping, optional
        Fixed reference fields or scalar values, defaulting to zero.
    scales : mapping, optional
        Positive constant parameter scales, defaulting to one.
    gradient_fraction : float, optional
        For ``"auto"`` weights: the size of the penalty's gradient, as a
        fraction of the data misfit's, at the starting model. Default 0.1.

    Notes
    -----
    The penalty is sum_p weight_p/2 * integral(|grad((m_p-ref_p)/scale_p)|^2).
    Reference Functions are copied without annotation at construction. This is a
    seminorm, not a mass-plus-stiffness full H1 norm or a proximal anchor.

    An ``"auto"`` weight is

    .. math::

        \beta_p = \gamma \, \frac{\|D_p J(m_0)\|}{\|D_p R_p(m_0)\|},

    with :math:`J` the data misfit, :math:`R_p` the penalty of parameter
    :math:`p` with unit weight, :math:`\gamma` the ``gradient_fraction`` and
    both norms the lumped :math:`L^2` norms of derivatives, so it does not
    depend on the scale of the data or of the model.
    """

    def __init__(self, weights: Mapping, references: Mapping | None = None,
                 scales: Mapping | None = None,
                 gradient_fraction: float = 0.1) -> None:
        self.weights = dict(weights)
        self.references = {}
        self.scales = dict(scales or {})
        self.gradient_fraction = float(gradient_fraction)
        if not all(isinstance(key, Enum) for key in self.weights):
            raise TypeError("H1 weights must be keyed by material parameter enums.")
        if (set(references or {}) | set(self.scales)) - set(self.weights):
            raise ValueError("References and scales must belong to weighted parameters.")
        if not math.isfinite(self.gradient_fraction) or self.gradient_fraction <= 0:
            raise ValueError("H1 gradient_fraction must be finite and positive.")
        for key, weight in self.weights.items():
            if weight == "auto":
                continue
            if (isinstance(weight, str) or not math.isfinite(weight)
                    or weight < 0):
                raise ValueError("H1 weights must be 'auto', or finite and nonnegative.")
            scale = self.scales.get(key, 1.0)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("H1 scales must be finite and positive.")
        with stop_annotating():
            for key, reference in (references or {}).items():
                if isinstance(reference, fire.Function):
                    self.references[key] = reference.copy(deepcopy=True)
                elif isinstance(reference, Real) and math.isfinite(reference):
                    self.references[key] = float(reference)
                else:
                    raise TypeError("H1 references must be scalar numbers or Functions.")

    @property
    def automatic(self) -> list:
        """list of enum.Enum: the parameters whose weight is still ``"auto"``."""
        return [key for key, weight in self.weights.items() if weight == "auto"]

    def _density(self, key: Enum, field: fire.Function) -> object:
        r"""Return the unit-weight integrand, half the squared scaled gradient.

        Parameters
        ----------
        key : enum.Enum
            Material parameter.
        field : firedrake.Function
            Its field.

        Returns
        -------
        ufl.Form
            :math:`\frac12 |\nabla((m_p - ref_p)/scale_p)|^2\,dx`.

        Raises
        ------
        ValueError
            If the field is not a continuous scalar Function.
        """
        if not isinstance(field, fire.Function) or field.ufl_shape:
            raise ValueError("H1 requires independent scalar Function controls.")
        if not field.ufl_element().sobolev_space <= H1:
            raise ValueError("H1 requires continuous controls; DG controls are unsupported.")
        reference = self.references.get(key, 0.0)
        scaled = (field - reference) / self.scales.get(key, 1.0)
        return 0.5 * fire.inner(fire.grad(scaled), fire.grad(scaled)) * fire.dx

    def unit_derivative(self, key: Enum, field: fire.Function) -> fire.Cofunction:
        """Return the derivative of the penalty of ``key`` with unit weight.

        Not annotated: it only sizes ``"auto"`` weights.

        Parameters
        ----------
        key : enum.Enum
            Material parameter.
        field : firedrake.Function
            Its field, where the derivative is taken.

        Returns
        -------
        firedrake.Cofunction
            :math:`D R_p(m)` with unit weight.
        """
        with stop_annotating():
            return fire.assemble(fire.derivative(self._density(key, field), field))

    def __call__(self, parameters: Mapping) -> object:
        """Assemble the physical penalty, respecting the active tape.

        Parameters
        ----------
        parameters : mapping or PhysicalParameters
            Independent continuous scalar fields keyed by material parameter.

        Returns
        -------
        float or pyadjoint.AdjFloat
            Penalty value, zero when all weights are zero.

        Raises
        ------
        ValueError
            If a weighted field is missing or is not continuous and scalar,
            or a weight is still ``"auto"``.
        """
        if self.automatic:
            raise ValueError(
                "H1 weights still 'auto' for "
                f"{[key.value for key in self.automatic]}: the inversion "
                "chooses them before its first forward solve.",
            )
        result = 0.0
        for key, weight in self.weights.items():
            if key not in parameters:
                raise ValueError(f"Missing H1 parameter: {key}.")
            if weight == 0:
                continue
            result += fire.assemble(weight * self._density(key, parameters[key]))
        return result
