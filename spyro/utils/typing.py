from enum import Enum


def override(func):
    '''
    This decorator should be replaced by typing.override when Python
    version is updated to 3.12
    '''
    return func


class WaveType(Enum):
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

        PER_TIMESTEP: Evaluate the functional at every time step during the time integration.
        AFTER_SOLVE: Evaluate the functional after the time integration is complete.
    """
    PER_TIMESTEP = "per_timestep"
    AFTER_SOLVE = "after_solve"
