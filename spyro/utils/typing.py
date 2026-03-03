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
