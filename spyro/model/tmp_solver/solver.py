from abc import ABC
from dataclasses import dataclass, field

from ..models import ExecutionConfig
from ..wave import Wave

class ForwardWavePropagatorStrategy:
    def propagate(wave: Wave):
        pass

class BackwardWavePropagatorStrategy:
    def propagate(wave: Wave):
        pass

class SolverOperatorBuilder:
    def build(wave: Wave):
        pass


class PMLSolverOperatorBuilder:
    def build(wave: Wave):
        pass


@dataclass
class Solver(ABC):
    input: ExecutionConfig
    forward_wave_propagator: ForwardWavePropagatorStrategy # default can be _forward_time_integrator
    backwards_wave_propagator: BackwardWavePropagatorStrategy
    solver_operator_builder: SolverOperatorBuilder
    wave: Wave = field(init=False)

    def forward_solve(self, initial_velocity_file):
        mesh_definition = self.input.solver_input.mesh_definition
        wave = Wave(mesh_definition, initial_velocity_file)
        matrix = self.solver_operator_builder.build(self.wave, self.input.solver_input)
        return self.forward_wave_propagator.propagate(matrix, wave)

    def gradient_solve():
        pass

