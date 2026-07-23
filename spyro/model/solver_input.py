from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings
from .mesh import Mesh
from .boundary import Boundary


class SourceTypes(Enum):
    RICKER = "ricker"
    MMS = "MMS"


class DelayType(Enum):
    MULTIPLE_OF_MINIMUM = "multiples_of_minimum"
    TIME = "time"


class EquationType(Enum):
    SECOND_ORDER_IN_PRESSURE = "second_order_in_pressure"


class Variant(Enum):
    LUMPED = "lumped"
    EQUISPACED = "equispaced"
    DG = "DG"


@dataclass
class Acquisition:
    receiver_locations: np.array
    delay: float
    delay_type: DelayType
    source_type: SourceTypes
    source_frequency: float
    source_locations: np.array | None = None  # can this be null?

    number_of_sources: int = field(init=False)

    def __post_init__(self):
        self.number_of_sources = len(self.source_locations)
        self.number_of_receivers = len(self.receiver_locations)

        if self.source_frequency < 1.0:
            warnings.warn(
                f"Frequency of {self.source_frequency} too low for realistic FWI."
            )
        elif self.source_frequency > 50:
            warnings.warn(
                f"Frequency of {self.source_frequency} too high for efficient FWI."
            )

        if self.source_type != SourceTypes.MMS and self.source_locations is None:
            raise ValueError("source_locations should not be None if source_type is not MMS")


@dataclass
class TimeAxis:
    initial_time: float
    final_time: float
    dt: float | None
    amplitude: float

    def __post_init__(self):
        if self.final_time < 0.0:
            raise ValueError(f"Negative time of {self.final_time} not valid.")
        if self.dt > 1.0:
            warnings.warn(f"Time step of {self.dt} too big.")
        if self.dt is None:
            warnings.warn(
                "Timestep not given. Will calculate internally when user \
                    attemps to propagate wave."
            )


@dataclass
class SolverInput:
    equation_type: EquationType
    mesh: Mesh
    boundary: Boundary
    acquisition: Acquisition
    time_axis: TimeAxis
    variant: Variant
