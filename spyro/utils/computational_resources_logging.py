from memory_profiler import memory_usage
from mpi4py import MPI
import firedrake as fire
from time import time


def log_max_memory_usage_per_core(
    prefix_string: str = "Maximum memory per core:",
    comm=None,
):
    from ..io.basicio import parallel_print
    local_memory = memory_usage(-1)[0]

    if fire.COMM_WORLD.Get_size() > 1:
        max_memory = fire.COMM_WORLD.allreduce(local_memory, op=MPI.MAX)
    else:
        max_memory = local_memory
    print_string = prefix_string + f"{max_memory:.2f} MB"
    parallel_print(print_string, comm=fire.COMM_WORLD)


def log_max_current_runtime_per_core(
    t0,
    prefix_string: str = "Maximum current runtime per core:",
    comm=None,
):
    from ..io.basicio import parallel_print
    local_t1 = time()

    if fire.COMM_WORLD.Get_size() > 1:
        max_t1 = fire.COMM_WORLD.allreduce(local_t1, op=MPI.MAX)
    else:
        max_t1 = local_t1
    runtime = max_t1 - t0
    print_string = prefix_string + f"{runtime:.2f} s"
    parallel_print(print_string, comm=fire.COMM_WORLD)


def log_max_computational_resources_per_core(
    t0,
    prefix_string: str | None = None,
    comm=None,
):
    if prefix_string is not None:
        memory_string = prefix_string + " MEMORY: "
        runtime_str = prefix_string + " RUNTIME: "
    else:
        memory_string = "MEMORY: "
        runtime_str = "RUNTIME: "
    log_max_memory_usage_per_core(prefix_string=memory_string, comm=comm)
    log_max_current_runtime_per_core(t0, prefix_string=runtime_str, comm=comm)
