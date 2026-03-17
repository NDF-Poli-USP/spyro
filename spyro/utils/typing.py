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


class AdjointType(Enum):
    """Enum for the type of adjoint solver to use."""
    NONE = 0
    AUTOMATED_ADJOINT = 1
    SPYRO_ADJOINT = 2
