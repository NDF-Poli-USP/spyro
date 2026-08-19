"""The named physical fields a wave equation is written in terms of."""

from collections.abc import Set
from enum import Enum

import firedrake as fire


DENSITY = "density"
LAMBDA = "lambda"
MU = "mu"
P_WAVE_VELOCITY = "p_wave_velocity"
S_WAVE_VELOCITY = "s_wave_velocity"


def parameter_name(name):
    """Return the canonical string name of a physical parameter.

    Parameters are named by the strings above, which are also the values of
    :class:`ElasticMaterialParameter`, so both spellings may be used
    interchangeably by callers.

    Parameters
    ----------
    name : str or enum.Enum
        Parameter name, or an enum member whose value is that name.

    Returns
    -------
    str
        Canonical parameter name.

    Raises
    ------
    TypeError
        If ``name`` is neither a string nor an enum member.

    Examples
    --------
    ``parameter_name(ElasticMaterialParameter.DENSITY)`` and
    ``parameter_name("density")`` both return ``"density"``.
    """
    if isinstance(name, Enum):
        return name.value
    if isinstance(name, str):
        return name
    raise TypeError(
        "Physical parameter names must be strings or enum members. "
        f"Received {type(name).__name__}.",
    )


class PhysicalParameters(Set):
    """The physical parameters a wave equation is written in terms of.

    The container is a set of parameter names, so an inversion can check that
    the parameters it controls are a subset of the ones the solver actually
    models, and it maps each name to the field the variational form uses.

    Some parameters are independent fields (Firedrake ``Function`` objects)
    and others are UFL expressions computed from them: an isotropic elastic
    medium declared with density and the Lame parameters carries the two wave
    speeds as expressions of those three. Only the independent ones can be
    updated, and because they are updated in place, the dependent expressions
    and the assembled variational forms follow automatically.

    Parameters
    ----------
    fields : mapping or iterable of (name, value), optional
        Initial parameters.

    Examples
    --------
    >>> parameters = PhysicalParameters()
    >>> parameters.add("density", rho)
    >>> "density" in parameters
    True
    >>> {"density"} <= parameters
    True
    >>> parameters["density"] is rho
    True
    """

    def __init__(self, fields=None):
        self._fields = {}
        if fields is None:
            return
        try:
            items = fields.items()
        except AttributeError:
            items = fields
        for name, value in items:
            self.add(name, value)

    # Set behaviour, over the parameter names.

    def __contains__(self, name):
        if not isinstance(name, (str, Enum)):
            return False
        return parameter_name(name) in self._fields

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    @classmethod
    def _from_iterable(cls, iterable):
        """Build the result of a set operation.

        Set algebra (``|``, ``&``, ``-``, ``^``) combines names alone, so the
        result carries no fields and is returned as a plain ``set`` rather
        than as a half-populated container.
        """
        return set(iterable)

    def issubset(self, other):
        """Return whether every parameter name is also present in ``other``."""
        return self <= other

    def issuperset(self, other):
        """Return whether ``other`` holds only names present here."""
        return self >= other

    # Mapping behaviour, from a parameter name to its field.

    def __getitem__(self, name):
        key = parameter_name(name)
        try:
            return self._fields[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is not a physical parameter of this wave equation. "
                f"Known parameters: {self._format_names()}.",
            ) from None

    def get(self, name, default=None):
        """Return the field of ``name``, or ``default`` if it is not modelled."""
        if name not in self:
            return default
        return self[name]

    def values(self):
        """Return the parameter fields."""
        return self._fields.values()

    def items(self):
        """Return the ``(name, field)`` pairs."""
        return self._fields.items()

    def add(self, name, value):
        """Declare one physical parameter, replacing any previous field.

        Solvers call this while initializing their material properties, which
        happens again on every forward solve, so re-declaring a parameter is
        expected and simply rebinds the name.

        Parameters
        ----------
        name : str or enum.Enum
            Parameter name.
        value : firedrake.Function or ufl.core.expr.Expr
            Independent field, or expression computed from the independent
            ones.
        """
        self._fields[parameter_name(name)] = value

    def discard(self, name):
        """Remove one parameter, if it is present."""
        self._fields.pop(parameter_name(name), None)

    def update(self, name, value):
        """Overwrite one parameter's field in place.

        The field is written into rather than replaced, so the variational
        forms already built from it, and any parameter computed from it, use
        the new values without being rebuilt.

        Parameters
        ----------
        name : str or enum.Enum
            Parameter to update.
        value : firedrake.Function, firedrake.Constant, scalar, or UFL expression
            New value. Anything that is not a ``Function`` in the same
            function space is interpolated into the existing field.

        Returns
        -------
        firedrake.Function
            The updated field.

        Raises
        ------
        KeyError
            If ``name`` is not one of the modelled parameters.
        TypeError
            If ``name`` is a dependent parameter, which cannot be assigned
            independently of the fields it is computed from.
        """
        field = self[name]
        if not isinstance(field, fire.Function):
            raise TypeError(
                f"'{parameter_name(name)}' is computed from the other physical "
                "parameters and cannot be updated on its own. Update the "
                "parameters it is computed from instead.",
            )
        if (
            isinstance(value, fire.Function)
            and value.function_space() == field.function_space()
        ):
            field.assign(value)
        else:
            field.interpolate(value)
        return field

    def copy(self, names=None):
        """Return an independent copy of some or all parameters.

        Parameters
        ----------
        names : iterable of str or enum.Enum, optional
            Parameters to copy. Defaults to all of them.

        Returns
        -------
        PhysicalParameters
            Container holding duplicated ``Function`` fields. Dependent
            parameters are UFL expressions, which are immutable, and are
            shared rather than duplicated.
        """
        names = self if names is None else names
        copied = PhysicalParameters()
        for name in names:
            field = self[name]
            if isinstance(field, fire.Function):
                duplicate = fire.Function(field.function_space(), name=field.name())
                duplicate.assign(field)
                copied.add(name, duplicate)
            else:
                copied.add(name, field)
        return copied

    def _format_names(self):
        return "{" + ", ".join(self._fields) + "}"

    def __repr__(self):
        return f"{type(self).__name__}({self._format_names()})"
