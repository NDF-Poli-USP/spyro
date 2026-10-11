"""Find the minimum cells-per-wavelength meshing parameter for a wave physics."""

from .calculator import (
    CPW_PHYSICS,
    AcousticCpwPhysics,
    CpwEvaluation,
    CpwPhysics,
    IsotropicElasticCpwPhysics,
    MeshingParameterCalculator,
)
from .line_search import (
    COARSE_STEP_FRACTION,
    LINE_SEARCHES,
    MAX_CPW,
    backward_line_search,
    forward_line_search,
)
from .threshold_case import ThresholdCase

__all__ = [
    "CPW_PHYSICS",
    "COARSE_STEP_FRACTION",
    "LINE_SEARCHES",
    "MAX_CPW",
    "AcousticCpwPhysics",
    "CpwEvaluation",
    "CpwPhysics",
    "IsotropicElasticCpwPhysics",
    "MeshingParameterCalculator",
    "ThresholdCase",
    "backward_line_search",
    "forward_line_search",
]
