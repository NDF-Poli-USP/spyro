"""Coordinate wrappers for reduced objectives."""

from .latent import LatentReducedFunctional
from .lumped_l2 import LumpedL2ReducedFunctional

__all__ = [
    "LatentReducedFunctional", "LumpedL2ReducedFunctional",
]
