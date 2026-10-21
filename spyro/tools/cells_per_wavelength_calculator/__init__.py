"""Find the minimum cells-per-wavelength meshing parameter of a wave problem."""

from .calculator import CpwEvaluation, MeshingParameterCalculator
from .elastic_2d_material_model import elastic_2d_material
from .heterogeneous_case import (
    HeterogeneousCase,
    MaterialFileCase,
    MaterialModelCase,
)
from .heterogeneous_cases import (
    MARMOUSI2_DENSITY_FILE,
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
    MARMOUSI2_S_WAVE_VELOCITY_FILE,
    AcousticMarmousi2Case,
    ElasticMarmousi2Case,
    ElasticSyntheticCase,
)
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
    AcousticHeterogeneousProblem,
    AcousticHomogeneousProblem,
    CpwProblem,
    ElasticHeterogeneousProblem,
    ElasticHomogeneousProblem,
    HeterogeneousCpwProblem,
    HomogeneousCpwProblem,
)
from .threshold_discretization import ThresholdDiscretization
from .wave_physics import ACOUSTIC, ISOTROPIC_ELASTIC, WavePhysics

__all__ = [
    "ACOUSTIC",
    "COARSE_STEP_FRACTION",
    "CPW_PROBLEMS",
    "ISOTROPIC_ELASTIC",
    "LINE_SEARCHES",
    "MARMOUSI2_DENSITY_FILE",
    "MARMOUSI2_P_WAVE_VELOCITY_FILE",
    "MARMOUSI2_S_WAVE_VELOCITY_FILE",
    "MAX_CPW",
    "AcousticHeterogeneousProblem",
    "AcousticHomogeneousProblem",
    "AcousticMarmousi2Case",
    "CpwEvaluation",
    "CpwParameters",
    "CpwProblem",
    "ElasticHeterogeneousProblem",
    "ElasticHomogeneousProblem",
    "ElasticMarmousi2Case",
    "ElasticSyntheticCase",
    "HeterogeneousCase",
    "HeterogeneousCpwProblem",
    "HomogeneousCpwProblem",
    "MaterialFileCase",
    "MaterialModelCase",
    "MeshingParameterCalculator",
    "ThresholdDiscretization",
    "WavePhysics",
    "backward_line_search",
    "elastic_2d_material",
    "forward_line_search",
]
