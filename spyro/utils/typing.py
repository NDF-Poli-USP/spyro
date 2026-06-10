from enum import Enum


def override(func):
    '''
    This decorator should be replaced by typing.override when Python
    version is updated to 3.12
    '''
    return func


class WaveType(Enum):
    """Enum for different types of wave equations that can be solved.

    NONE: No wave equation.
    ISOTROPIC_ACOUSTIC: Isotropic acoustic wave equation.
    ISOTROPIC_ELASTIC: Isotropic elastic wave equation for Isotropic media.
    ANISOTROPIC_VTI_ELASTIC: Anisotropic elastic wave equation for VTI media.
    ANISOTROPIC_TTI_ELASTIC: Anisotropic elastic wave equation for TTI media.
    """
    NONE = 0
    ISOTROPIC_ACOUSTIC = 1
    ISOTROPIC_ELASTIC = 2
    ANISOTROPIC_VTI_ELASTIC = 3
    ANISOTROPIC_TTI_ELASTIC = 4


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


class LayerShapeType(Enum):
    """Enum for different types of absorbing layer shapes for ABCs.

    NOLAYER: No absorbing layer, i.e., no ABCs applied.
    RECTANGULAR: Rectangular absorbing layer`around the computational domain.
    HYPERSHAPE: Hypershape absorbing layer around the computational domain
    """
    NOLAYER = "no_layer"
    RECTANGULAR = "rectangular"
    HYPERSHAPE = "hypershape"
