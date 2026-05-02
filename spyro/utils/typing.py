"""Define lightweight typing utilities shared across spyro modules."""

from enum import Enum


def override(func):
    """Mark a method as intentionally overriding a base implementation.

    Parameters
    ----------
    func : callable
        Method being declared as an override.

    Returns
    -------
    callable
        The same callable received as input.

    Notes
    -----
    Replace this compatibility shim with ``typing.override`` when the minimum
    supported Python version is upgraded to 3.12.
    """
    return func


class WaveType(Enum):
    """Enumerate supported wave-physics models.

    Attributes
    ----------
    NONE : int 0
        Disable wave-physics selection.
    ISOTROPIC_ACOUSTIC : int 1
        Select isotropic acoustic propagation.
    ISOTROPIC_ELASTIC : int 2
        Select isotropic elastic propagation.
    """

    NONE = 0
    ISOTROPIC_ACOUSTIC = 1
    ISOTROPIC_ELASTIC = 2


class FunctionalType(Enum):
    """Enum for different types of functionals that can be computed.

    L2Norm : L2 norm functional, commonly used in classical full waveform
    inversion (FWI) as the measure to be minimized.
    """

    L2Norm = 0


class FunctionalEvaluationMode(Enum):
    """The mode in which to evaluate the functional.

    PER_TIMESTEP: Evaluate the functional at every time step during the time
    integration.
    AFTER_SOLVE: Evaluate the functional after the time integration is complete.
    """

    PER_TIMESTEP = "per_timestep"
    AFTER_SOLVE = "after_solve"
