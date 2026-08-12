from dataclasses import InitVar, dataclass

from attr import field
import numpy as np
import warnings
import firedrake as fire
from spyro.model.position import Position
from spyro.model.time_axis import TimeAxis

@dataclass
class Receiver:
    p: Position
    real: np.array
    synthetic: np.array = field(init=False)

    def __post_init__(self):
        self.synthetic =  np.zeros_like(self.real)

    def sample(self, wave: fire.Function, time: TimeAxis):
        self.synthetic[time.index] = wave.at(**self.p.get_positions())

    def get_misfit(self):
        residual = self.synthetic - self.data

        return 0.5 * np.sum(residual**2)

@dataclass
class Source:
    p: Position
    space: InitVar[fire.FunctionSpace]
    f: fire.Cofunction = field(init=False)

    def __post_init__(self, space: fire.FunctionSpace):
        self.f = fire.Cofunction(space.dual())

        mesh = space.mesh()

        source_mesh = fire.VertexOnlyMesh(
            mesh,
            [(self.x, self.y)]
        )

        source_space = fire.FunctionSpace(
            source_mesh,
            "DG",
            0
        )

        self.source_value = fire.Function(source_space)
        self.source_value.assign(1.0)

    def update(self, time: TimeAxis):
        amp = self.amplitude(time)
        self.f.assign(amp*self.source_value)

    def amplitude(self, time: TimeAxis):
        return 0

@dataclass
class DataSource:
    data: np.array

    def amplitude(self, time: TimeAxis) -> float:
        return self.data[time.index]

@dataclass
class RickerSource(Source):
    delay: float
    frequency: float

    def amplitude(self, time: TimeAxis) -> float:
        tau = time - self.delay

        a = np.pi * self.frequency * tau

        return (1.0 - 2.0 * a**2) * np.exp(-a**2)

@dataclass
class Acquisition:
    sources: list[Source]
    receivers: list[Receiver]

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

    def get_source_function(self):
        total = fire.Cofunction(self.sources[0].f.function_space())

        total.assign(0)

        for source in self.sources:
            total += source.f

        return total

    def get_receivers_misfit(self):
        return sum(
            receiver.get_misfit()
            for receiver in self.receivers
        )

    def update_sources(self, time: TimeAxis):
        for source in self.sources:
            source.update(time)

    def sample_at_receivers(self, wave: fire.Function, time: TimeAxis):
        for receiver in self.receivers:
            receiver.sample(wave, time)