"""Declarative description of the isotropic elastic material parameters.

An isotropic elastic medium is fully determined by three independent scalar
fields, and Spyro supports two equivalent coordinate systems for them, called
*parameterizations*:

* :attr:`~spyro.utils.typing.ElasticMaterialParameterization.LAME` --- density
  with the first and second Lame parameters.
* :attr:`~spyro.utils.typing.ElasticMaterialParameterization.VELOCITY` ---
  density with the P- and S-wave velocities.

The two families are related by

.. math::

    \\mu = \\rho c_s^2, \\qquad \\lambda = \\rho c_p^2 - 2\\mu,

and, inversely,

.. math::

    c_p = \\sqrt{(\\lambda + 2\\mu)/\\rho}, \\qquad c_s = \\sqrt{\\mu/\\rho}.

Two independent choices are made on top of that, and Spyro keeps them apart:

* **The equation parameters.** One family is the *active* one: its three
  parameters are independent Firedrake ``Function`` objects and the other two
  are UFL expressions of them, so the conversion above sits inside the
  variational form and pyadjoint can differentiate through it. The active
  family is material state, owned by
  :class:`~spyro.solvers.elastic_wave.isotropic_wave.IsotropicWave`, because it
  defines the PDE coefficients whether or not an adjoint is being computed.
* **The control subset.** Any non-empty subset of the active family may be
  differentiated. That selection is an inversion concern, so it is stored only
  on :class:`~spyro.solvers.automatic_differentiation_solver.AutomatedAdjoint`.

:class:`ElasticControlSet` is the validated result of a selection. It is a
transient value object: it is used to align the active family with a requested
selection and to resolve names into fields, and is not kept as solver state.

Because the map between the families is an invertible change of variables,
gradients taken in one of them convert exactly into the other without solving
again. :data:`DERIVATIVES_BY_PARAMETERIZATION` holds the chain-rule
coefficients for that conversion.

This module holds only the declarative tables and the conversion rules; the
Firedrake state itself lives on ``IsotropicWave``.
"""

from dataclasses import dataclass
from types import MappingProxyType

from ...utils.typing import (ElasticMaterialParameter,
                             ElasticMaterialParameterization)


#: Independent parameters of each supported parameterization, in the order
#: used by the public model dictionary and by the FWI control vectors.
PARAMETERS_BY_PARAMETERIZATION = MappingProxyType({
    ElasticMaterialParameterization.LAME: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.LAMBDA,
        ElasticMaterialParameter.MU,
    ),
    ElasticMaterialParameterization.VELOCITY: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    ),
})

#: Solver attribute holding each material parameter.
ATTRIBUTE_BY_PARAMETER = MappingProxyType({
    ElasticMaterialParameter.DENSITY: "rho",
    ElasticMaterialParameter.LAMBDA: "lmbda",
    ElasticMaterialParameter.MU: "mu",
    ElasticMaterialParameter.P_WAVE_VELOCITY: "c",
    ElasticMaterialParameter.S_WAVE_VELOCITY: "c_s",
})

#: Model-dictionary keys accepted for each material parameter, canonical
#: spelling first. Declaring a material and rewriting one both go through this
#: table, so the schema is defined in a single place.
KEYS_BY_PARAMETER = MappingProxyType({
    ElasticMaterialParameter.DENSITY: ("density",),
    ElasticMaterialParameter.LAMBDA: ("lambda", "lame_first"),
    ElasticMaterialParameter.MU: ("mu", "lame_second"),
    ElasticMaterialParameter.P_WAVE_VELOCITY: ("p_wave_velocity",),
    ElasticMaterialParameter.S_WAVE_VELOCITY: ("s_wave_velocity",),
})

#: Every spelling accepted when naming a parameter: the model-dictionary keys
#: above plus the solver attribute names, so ``"c_s"`` and
#: ``"s_wave_velocity"`` are interchangeable in a control selection.
PARAMETER_BY_ALIAS = MappingProxyType({
    **{key: parameter
       for parameter, keys in KEYS_BY_PARAMETER.items() for key in keys},
    **{attribute: parameter
       for parameter, attribute in ATTRIBUTE_BY_PARAMETER.items()},
})


def resolve(parameter):
    """Return the enum value denoted by a material parameter name.

    Parameters
    ----------
    parameter : str or ElasticMaterialParameter
        Parameter name accepted by :data:`PARAMETER_BY_ALIAS`, or an enum
        value, which is returned unchanged.

    Returns
    -------
    ElasticMaterialParameter
        Canonical material parameter.

    Raises
    ------
    TypeError
        If ``parameter`` is neither a string nor an enum value.
    ValueError
        If the name is not a recognized alias.

    Examples
    --------
    ``resolve("lame_first")`` and ``resolve("lmbda")`` both return
    ``ElasticMaterialParameter.LAMBDA``.
    """
    if isinstance(parameter, ElasticMaterialParameter):
        return parameter
    if not isinstance(parameter, str):
        raise TypeError(
            "Elastic material parameters must be strings or "
            "ElasticMaterialParameter values.",
        )
    try:
        return PARAMETER_BY_ALIAS[parameter]
    except KeyError as exc:
        supported = ", ".join(PARAMETER_BY_ALIAS)
        raise ValueError(
            f"Unknown elastic material parameter '{parameter}'. "
            f"Supported names are: {supported}.",
        ) from exc


def _velocity_derivatives_from_lame(rho, c, c_s):
    """Coefficients building velocity derivatives from Lame ones.

    Differentiating :math:`\\lambda = \\rho(c_p^2 - 2c_s^2)` and
    :math:`\\mu = \\rho c_s^2` gives the Jacobian of the change of variables,
    and the chain rule contracts a Lame derivative with its columns. Note that
    the density derivative differs between the families: holding
    :math:`(\\lambda, \\mu)` fixed is not the same as holding
    :math:`(c_p, c_s)` fixed.

    Parameters
    ----------
    rho, c, c_s : firedrake.Function or UFL expression
        Current density, P- and S-wave velocity fields.

    Returns
    -------
    dict
        Maps each velocity-family parameter to the Lame parameters its
        derivative depends on, and the coefficient of each.
    """
    return {
        ElasticMaterialParameter.DENSITY: {
            ElasticMaterialParameter.DENSITY: 1.0,
            ElasticMaterialParameter.LAMBDA: c**2 - 2*c_s**2,
            ElasticMaterialParameter.MU: c_s**2,
        },
        ElasticMaterialParameter.P_WAVE_VELOCITY: {
            ElasticMaterialParameter.LAMBDA: 2*rho*c,
        },
        ElasticMaterialParameter.S_WAVE_VELOCITY: {
            ElasticMaterialParameter.LAMBDA: -4*rho*c_s,
            ElasticMaterialParameter.MU: 2*rho*c_s,
        },
    }


def _lame_derivatives_from_velocity(rho, c, c_s):
    """Coefficients building Lame derivatives from velocity ones.

    The inverse of :func:`_velocity_derivatives_from_lame`, obtained by
    differentiating :math:`c_p = \\sqrt{(\\lambda + 2\\mu)/\\rho}` and
    :math:`c_s = \\sqrt{\\mu/\\rho}`.

    Parameters
    ----------
    rho, c, c_s : firedrake.Function or UFL expression
        Current density, P- and S-wave velocity fields.

    Returns
    -------
    dict
        Maps each Lame parameter to the velocity-family parameters its
        derivative depends on, and the coefficient of each.
    """
    return {
        ElasticMaterialParameter.DENSITY: {
            ElasticMaterialParameter.DENSITY: 1.0,
            ElasticMaterialParameter.P_WAVE_VELOCITY: -c/(2*rho),
            ElasticMaterialParameter.S_WAVE_VELOCITY: -c_s/(2*rho),
        },
        ElasticMaterialParameter.LAMBDA: {
            ElasticMaterialParameter.P_WAVE_VELOCITY: 1/(2*rho*c),
        },
        ElasticMaterialParameter.MU: {
            ElasticMaterialParameter.P_WAVE_VELOCITY: 1/(rho*c),
            ElasticMaterialParameter.S_WAVE_VELOCITY: 1/(2*rho*c_s),
        },
    }


#: Chain-rule coefficients converting a complete set of derivatives into the
#: other material family, keyed by the family being converted *to*.
DERIVATIVES_BY_PARAMETERIZATION = MappingProxyType({
    ElasticMaterialParameterization.VELOCITY: _velocity_derivatives_from_lame,
    ElasticMaterialParameterization.LAME: _lame_derivatives_from_velocity,
})


def resolve_parameterization(parameterization):
    """Return the enum value denoted by a parameterization name.

    Parameters
    ----------
    parameterization : str or ElasticMaterialParameterization
        ``"lame"`` or ``"velocity"``, or an enum value, returned unchanged.

    Returns
    -------
    ElasticMaterialParameterization
        Canonical parameterization.

    Raises
    ------
    TypeError
        If ``parameterization`` is neither a string nor an enum value.
    ValueError
        If the name is not a recognized parameterization.

    Examples
    --------
    ``resolve_parameterization("velocity")`` returns
    ``ElasticMaterialParameterization.VELOCITY``.
    """
    if isinstance(parameterization, ElasticMaterialParameterization):
        return parameterization
    if not isinstance(parameterization, str):
        raise TypeError(
            "The elastic parameterization must be a string or an "
            "ElasticMaterialParameterization value.",
        )
    try:
        return ElasticMaterialParameterization(parameterization)
    except ValueError as exc:
        supported = ", ".join(
            member.value for member in ElasticMaterialParameterization
        )
        raise ValueError(
            f"Unknown elastic parameterization '{parameterization}'. "
            f"Supported names are: {supported}.",
        ) from exc


@dataclass(frozen=True)
class ElasticControlSet:
    """A validated, non-empty subset of one isotropic parameter family.

    Attributes
    ----------
    parameterization : ElasticMaterialParameterization
        Family the controls belong to. The solver keeps every parameter of
        this family as a Firedrake ``Function`` and derives the remaining
        parameters from them.
    parameters : tuple of ElasticMaterialParameter
        The selected controls, in the order given by the user, or in the
        order of :data:`PARAMETERS_BY_PARAMETERIZATION` for a complete family.

    Examples
    --------
    ``ElasticControlSet.select(["lambda", "mu"], default=VELOCITY)`` selects
    the two Lame moduli as controls even though the material was declared with
    wave velocities.
    """

    parameterization: ElasticMaterialParameterization
    parameters: tuple

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    @classmethod
    def complete(cls, parameterization):
        """Return the control set holding every parameter of one family.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Family whose parameters are all controls.

        Returns
        -------
        ElasticControlSet
            Control set with the three independent parameters of the family.
        """
        return cls(
            parameterization,
            PARAMETERS_BY_PARAMETERIZATION[parameterization],
        )

    @classmethod
    def select(cls, parameters, *, default):
        """Validate a user control selection and infer its family.

        Parameters
        ----------
        parameters : list, tuple, or None
            Control names or enum values. ``None`` selects every parameter of
            ``default``. Any non-empty, duplicate-free subset of a single
            family is accepted; a selection that fits both families (only
            ``density``) keeps ``default``.
        default : ElasticMaterialParameterization
            Family used when the selection does not determine one, typically
            the family the material was declared with.

        Returns
        -------
        ElasticControlSet
            Validated selection.

        Raises
        ------
        TypeError
            If ``parameters`` is not a list or tuple, or if an item is neither
            a string nor an ``ElasticMaterialParameter``.
        ValueError
            If the selection is empty, contains duplicates (including two
            aliases of the same parameter), mixes Lame parameters with wave
            velocities, or names an unknown parameter.
        """
        if parameters is None:
            return cls.complete(default)
        if not isinstance(parameters, (list, tuple)):
            raise TypeError(
                "Elastic control_parameters must be a list or tuple.",
            )

        selected = tuple(resolve(parameter) for parameter in parameters)
        if not selected:
            raise ValueError(
                "At least one elastic control parameter is required.",
            )
        if len(set(selected)) != len(selected):
            raise ValueError(
                "Elastic control parameters must not contain duplicates.",
            )

        families = [
            parameterization
            for parameterization, family in PARAMETERS_BY_PARAMETERIZATION.items()
            if set(selected) <= set(family)
        ]
        if not families:
            raise ValueError(
                "Elastic controls cannot mix Lame parameters with wave "
                "velocities.",
            )
        # ``density`` alone belongs to both families, so it cannot change the
        # parameterization the material was declared with.
        parameterization = default if len(families) > 1 else families[0]
        return cls(parameterization, selected)
