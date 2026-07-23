from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import firedrake as fire
from .. import meshing

class ReferenceFrequency(Enum):
    BOUNDARY = "boundary"
    SOURCE = "source"

class ExtensionMode(Enum):
    DRIVEN = "abc_driven"
    BUILTIN = "builtin"


@dataclass 
class BoundaryShape:
    pass

@dataclass 
class RectangularBoundary(BoundaryShape):
    pass

@dataclass 
class HypershapeBoundary(BoundaryShape):
    degree: int
    degree_type: str

@dataclass
class Boundary:
    ref_model: bool
    reference_freq: ReferenceFrequency
    extension_mode: ExtensionMode
    pad_length: float
    shape: BoundaryShape
    degree_eikonal: int = 2

    def __post_init__(self):
        if self.pad_length < 0: 
            raise ValueError("Pad length must be positive")

@dataclass
class PMLBoundary(Boundary):
    max_acoustic_velocity: float
    exponent: float = 2
    R: float = 1e-6
    cmax: float = 4.7
    shape: RectangularBoundary
