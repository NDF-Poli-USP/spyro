"""Reduced functionals that wrap a pyadjoint reduced functional."""

from .implemented_adjoint import ImplementedAdjointReducedFunctional
from .lumped_l2 import LumpedL2ReducedFunctional


__all__ = [
    "ImplementedAdjointReducedFunctional",
    "LumpedL2ReducedFunctional",
]
