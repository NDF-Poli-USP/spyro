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
    weights : mapping of enum.Enum to float
        Nonnegative weights for the material parameters to penalize.
    references : mapping, optional
        Fixed reference fields or scalar values, defaulting to zero.
    scales : mapping, optional
        Positive constant parameter scales, defaulting to one.

    Notes
    -----
    The penalty is sum_p weight_p/2 * integral(|grad((m_p-ref_p)/scale_p)|^2).
    Reference Functions are copied without annotation at construction. This is a
    seminorm, not a mass-plus-stiffness full H1 norm or a proximal anchor.
    """

    def __init__(self, weights: Mapping, references: Mapping | None = None,
                 scales: Mapping | None = None) -> None:
        self.weights = dict(weights)
        self.references = {}
        self.scales = dict(scales or {})
        if not all(isinstance(key, Enum) for key in self.weights):
            raise TypeError("H1 weights must be keyed by material parameter enums.")
        if (set(references or {}) | set(self.scales)) - set(self.weights):
            raise ValueError("References and scales must belong to weighted parameters.")
        for key, weight in self.weights.items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("H1 weights must be finite and nonnegative.")
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
            If a weighted field is missing or is not continuous and scalar.
        """
        result = 0.0
        for key, weight in self.weights.items():
            if key not in parameters:
                raise ValueError(f"Missing H1 parameter: {key}.")
            if weight == 0:
                continue
            field = parameters[key]
            if not isinstance(field, fire.Function) or field.ufl_shape:
                raise ValueError("H1 requires independent scalar Function controls.")
            if not field.ufl_element().sobolev_space <= H1:
                raise ValueError("H1 requires continuous controls; DG controls are unsupported.")
            reference = self.references.get(key, 0.0)
            scaled = (field - reference) / self.scales.get(key, 1.0)
            result += fire.assemble(
                0.5 * weight * fire.inner(fire.grad(scaled), fire.grad(scaled)) * fire.dx,
            )
        return result
