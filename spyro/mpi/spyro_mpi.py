import copy
from enum import Enum
from typing import Dict
from ..io.basicio import parallel_print
from pydantic import BaseModel, computed_field
from mpi4py import MPI
from firedrake import Ensemble


class ParallelismType(Enum):
    AUTOMATIC = "automatic"
    SPATIAL = "spatial"
    CUSTOM = "custom"


class ParallelismConfig(BaseModel):
    number_of_sources: int

    @computed_field
    @property
    def shot_ids_per_propagation(self) -> list[list[int]]:
        return [[i] for i in range(0, self.number_of_sources)]

    def get_number_of_cores(self):
        pass


class AutomaticParallelism(ParallelismConfig):
    def get_number_of_cores(self):
        available_cores = MPI.COMM_WORLD.size
        if available_cores % self.number_of_sources != 0:
            raise ValueError(
                f"Available cores cannot be divided between sources equally {available_cores}/{self.number_of_sources}."
            )
        return available_cores // self.number_of_sources


class SpatialParallelism(ParallelismConfig):
    def get_number_of_cores(self):
        return MPI.COMM_WORLD.size


class CustomParallelism(ParallelismConfig):
    custom_shot_ids: list[list[int]]

    @computed_field
    @property
    def shot_ids_per_propagation(self) -> list[list[int]]:
        return self.custom_shot_ids

    def get_number_of_cores(self):
        available_cores = MPI.COMM_WORLD.size
        return available_cores / len(self.shot_ids_per_propagation)


class SpyroComm:
    ensemble = None
    config: ParallelismConfig = None

    @classmethod
    def initialize_with_dict(cls, config: Dict, number_of_sources: int = None):

        parallelism_type = ParallelismType(config["type"])

        if parallelism_type == ParallelismType.CUSTOM:
            cls.initialize(
                CustomParallelism(
                    number_of_sources=number_of_sources,
                    custom_shot_ids=config["shot_ids_per_propagation"],
                )
            )
        elif parallelism_type == ParallelismType.AUTOMATIC:
            cls.initialize(
                AutomaticParallelism(number_of_sources=number_of_sources)
            )
        elif parallelism_type == ParallelismType.SPATIAL:
            cls.initialize(SpatialParallelism(number_of_sources=number_of_sources))

    @classmethod
    def initialize(cls, config: ParallelismConfig):
        cls.ensemble = Ensemble(
            MPI.COMM_WORLD,
            config.get_number_of_cores(),
        )

        cls.config = config

        parallel_print(
            f"Parallelism type: {type(config)}",
            comm=cls.ensemble,
        )

        cls.barrier()

    @classmethod
    def barrier(cls):
        cls.ensemble.comm.barrier()

    @classmethod
    def communicate(cls, array):
        if cls.ensemble.comm.size == 1:
            return array

        array_reduced = copy.copy(array)

        cls.print("Spatial parallelism, reducing to comm 0")

        cls.ensemble.comm.Allreduce(array, array_reduced, op=MPI.MAX)
        return array_reduced

    @classmethod
    def get_rank(cls) -> int:
        return cls.ensemble.comm.rank

    @classmethod
    def print(cls, string: str):
        if cls.ensemble is None:
            print(string, flush=True)
            return

        if cls.ensemble.comm.rank == 0 and cls.ensemble.ensemble_comm.rank == 0:
            print(string, flush=True)
