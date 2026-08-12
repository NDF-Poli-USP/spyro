from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import warnings
import firedrake as fire
from spyro.model.position import Position
from spyro.model.time_axis import TimeAxis


@dataclass
class Receiver:
    p: Position

    def sample(self, wave: fire.Function):
        point_evaluator = fire.PointEvaluator(wave.function_space().mesh(), [self.p.get_position()])
        return point_evaluator.evaluate(wave).item()


@dataclass
class Source(ABC):
    p: Position

    @abstractmethod
    def amplitude(self, time: TimeAxis):
        pass


@dataclass
class DataSource(Source):
    data: np.array

    def amplitude(self, time: TimeAxis) -> float:
        return self.data[time.index]


@dataclass
class RickerSource(Source):
    delay: float
    frequency: float

    def __post_init__(self):
        if self.frequency < 1.0:
            warnings.warn(
                f"Frequency of {self.frequency} too low for realistic FWI."
            )
        elif self.frequency > 50:
            warnings.warn(
                f"Frequency of {self.frequency} too high for efficient FWI."
            )

    def amplitude(self, time: TimeAxis) -> float:
        tau = time.current_time - self.delay

        a = np.pi * self.frequency * tau

        return (1.0 - 2.0 * a**2) * np.exp(-a**2)


@dataclass
class Acquisition:
    sources: list[Source]
    receivers: list[Receiver]

    def sample(self, wave: fire.Function):
        return np.array([
            receiver.sample(wave)
            for receiver in self.receivers
        ])

    def get_source_positions(self) -> list[tuple[float]]:
        return [source.p.get_position() for source in self.sources]

    def get_source_amplitudes(self, time_axis: TimeAxis) -> list[float]:
        return [
            source.amplitude(time_axis) for source in self.sources
        ]

    def to_adjoint_acquisition(self, misfits: list[np.array]):
        return Acquisition(
            sources=[
                DataSource(p=receiver.p, data=misfit) for receiver, misfit in zip(self.receivers, misfits)
            ],
            receivers=[]
        )
