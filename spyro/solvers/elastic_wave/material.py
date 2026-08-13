"""Declarative description of the isotropic elastic material parameters.

Notes
-----
**What a parameterization is.** An isotropic elastic medium is fully determined
by three independent scalar fields, and there is more than one equivalent way
to choose them. Each such choice is a *parameterization*, listed by
:class:`~spyro.utils.typing.ElasticMaterialParameterization`:

``LAME``
    density :math:`\\rho`, first Lame parameter :math:`\\lambda`, and second
    Lame parameter :math:`\\mu`.
``VELOCITY``
    density :math:`\\rho`, P-wave velocity :math:`c_p`, and S-wave velocity
    :math:`c_s`.

The two are related by

.. math::

    \\mu = \\rho c_s^2, \\qquad \\lambda = \\rho c_p^2 - 2\\mu,

and, inversely,

.. math::

    c_p = \\sqrt{(\\lambda + 2\\mu)/\\rho}, \\qquad c_s = \\sqrt{\\mu/\\rho}.

**What "active" means.** Exactly one parameterization is active at a time. Its
three parameters are stored as independent Firedrake ``Function`` objects, and
the two parameters of the other one become UFL expressions of them through the
relations above. Because those expressions sit inside the variational form,
pyadjoint differentiates through them, and the gradient is taken with respect
to whichever parameterization is active. Choosing it therefore changes the
form itself, so it is material state owned by
:class:`~spyro.solvers.elastic_wave.isotropic_wave.IsotropicWave` and is fixed
before the forward solve is recorded.

**What is chosen separately.** Which subset of the active parameterization is
actually differentiated is an inversion concern, not a property of the medium,
and is stored only on
:class:`~spyro.solvers.automatic_differentiation_solver.AutomatedAdjoint`.
:class:`ElasticControlSet` is the validated result of such a selection: a
transient value object used to resolve names into fields, never kept as solver
state.

This module holds only the declarative tables and the selection rules; the
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

#: Model-dictionary key of each material parameter. Declaring a material and
#: rewriting one both go through this table, so the schema is defined in a
#: single place, and each parameter has exactly one accepted spelling.
KEY_BY_PARAMETER = MappingProxyType({
    parameter: parameter.value for parameter in ElasticMaterialParameter
})

#: Every spelling accepted when naming a parameter in a control selection: the
#: model-dictionary keys above plus the solver attribute names, so ``"c_s"``
#: and ``"s_wave_velocity"`` both select the S-wave velocity.
PARAMETER_BY_ALIAS = MappingProxyType({
    **{key: parameter for parameter, key in KEY_BY_PARAMETER.items()},
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
    """A validated, non-empty subset of one isotropic parameterization.

    Attributes
    ----------
    parameterization : ElasticMaterialParameterization
        Parameterization the controls belong to. The solver keeps its three
        parameters as Firedrake ``Function`` objects and derives the other
        two from them; see the module Notes.
    parameters : tuple of ElasticMaterialParameter
        The selected controls, in the order given by the user, or in the
        order of :data:`PARAMETERS_BY_PARAMETERIZATION` when all are selected.

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
        """Return the control set holding every parameter of one of them.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Parameterization whose three parameters are all controls.

        Returns
        -------
        ElasticControlSet
            Control set with its three independent parameters.
        """
        return cls(
            parameterization,
            PARAMETERS_BY_PARAMETERIZATION[parameterization],
        )

    @classmethod
    def select(cls, parameters, *, default):
        """Validate a user control selection and infer its parameterization.

        Parameters
        ----------
        parameters : list, tuple, or None
            Control names or enum values. ``None`` selects every parameter of
            ``default``. Any non-empty, duplicate-free subset of a single
            parameterization is accepted; a selection that fits both of them
            (only ``density``) keeps ``default``.
        default : ElasticMaterialParameterization
            Parameterization used when the selection does not determine
            one, typically the one the material was declared with.

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

        # A selection determines a parameterization when it fits inside
        # exactly one of them.
        candidates = [
            parameterization
            for parameterization, parameters in
            PARAMETERS_BY_PARAMETERIZATION.items()
            if set(selected) <= set(parameters)
        ]
        if not candidates:
            raise ValueError(
                "Elastic controls cannot mix Lame parameters with wave "
                "velocities.",
            )
        # ``density`` alone fits both, so it cannot change the
        # parameterization the material was declared with.
        parameterization = default if len(candidates) > 1 else candidates[0]
        return cls(parameterization, selected)
