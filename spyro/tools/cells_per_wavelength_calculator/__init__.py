"""Find the minimum cells-per-wavelength meshing parameter of a wave problem."""

from .calculator import CpwEvaluation, MeshingParameterCalculator
from .line_search import (
    COARSE_STEP_FRACTION,
    LINE_SEARCHES,
    MAX_CPW,
    backward_line_search,
    forward_line_search,
)
from .parameters import CpwParameters
from .problems import (
    CPW_PROBLEMS,
    AcousticHomogeneousProblem,
    CpwProblem,
    HomogeneousCpwProblem,
)
from .threshold_discretization import ThresholdDiscretization
from .wave_physics import ACOUSTIC, WavePhysics

__all__ = [
    "ACOUSTIC",
    "COARSE_STEP_FRACTION",
    "CPW_PROBLEMS",
    "LINE_SEARCHES",
    "MAX_CPW",
    "AcousticHomogeneousProblem",
    "CpwEvaluation",
    "CpwParameters",
    "CpwProblem",
    "HomogeneousCpwProblem",
    "MeshingParameterCalculator",
    "ThresholdDiscretization",
    "WavePhysics",
    "backward_line_search",
    "forward_line_search",
]
