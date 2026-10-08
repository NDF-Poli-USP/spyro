"""Reduced functionals that wrap a pyadjoint reduced functional."""

from .latent import LatentReducedFunctional
from .lumped_l2 import LumpedL2ReducedFunctional


__all__ = [
    "LatentReducedFunctional",
    "LumpedL2ReducedFunctional",
]
