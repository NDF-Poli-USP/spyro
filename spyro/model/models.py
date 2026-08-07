from dataclasses import dataclass
from .parallelism import ParallelismConfig
from .solver_input import SolverInput


@dataclass
class UpdateFrequency:
    output_frequency: float = 99999
    gradient_sampling_frequency: float


@dataclass
class OutputSpecification:
    forward_output: str | None = None
    velocity_model_filename: str | None = None
    gradient_filename: str | None = None


@dataclass
class ExecutionConfig:
    solver_input: SolverInput
    update_frequency: UpdateFrequency
    output: OutputSpecification
    parallelism_config: ParallelismConfig

    def __post_init__(self):
        self.mesh.assert_point_in_domain(self.acquisition.source_frequency)
        self.mesh.assert_point_in_domain(self.acquisition.receiver_locations)
        self.parallelismConfig.setup(self.acquisition.source_locations)

    def get_mesh(self):
        return self.mesh.get_mesh()
