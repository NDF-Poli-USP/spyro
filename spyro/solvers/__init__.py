from .forward import forward
from .forward_AD import forward as forward_AD
from .gradient import gradient
from .wave import Wave
from .CG_acoustic import AcousticWave

__all__ = [
    "forward",  # forward solver adapted for discrete adjoint
    "forward_AD",  # forward solver adapted for Automatic Differentiation
    "gradient",
    "Wave",
    "AcousticWave",
    "DG_Wave",
]
