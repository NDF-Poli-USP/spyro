from dataclasses import dataclass, field
from enum import Enum
from .. import utils


class ParallelismType(Enum):
    CUSTOM = "custom"
    AUTOMATIC = "automatic"
    SPATIAL = "spatial"


@dataclass
class ParallelismConfig:
    type: ParallelismType
    shot_ids_per_propagation: list | None = None
    comm: object = field(init=False)  # dont know the object type yet

    def __post_init__(self):
        if type == ParallelismType.CUSTOM and self.shot_ids_per_propagation is None:
            raise ValueError(
                "shot_ids_per_propagation not specified "
                + "for ParallelismConfig even though type is CUSTOM"
            )

    def setup(self, number_of_sources):
        self.comm = utils.mpi_init(self)
        self.comm.comm.barrier()
        if type != ParallelismType.CUSTOM:
            self.shot_ids_per_propagation = [[i] for i in range(0, number_of_sources)]
