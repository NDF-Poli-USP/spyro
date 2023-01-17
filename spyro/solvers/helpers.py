import os

import numpy as np
from firedrake import File

from .. import io

__all__ = [
    "create_output_file",
    "display",
    "display_progress",
    "receivers_local",
    "fill",
]


def fill(usol_recv, is_local, nt, nr):
    """Fills usol_recv with -99999 value
    when it isn't local to any core"""
    usol_recv = np.asarray(usol_recv)
    for ti in range(nt):
        for rn in range(nr):
            if is_local[rn] is None:
                usol_recv[ti][rn] = -99999.0
    return usol_recv


def create_output_file(name, comm, source_num):
    """Saves shots in output file"""
    if io.is_owner(comm, source_num):
        outfile = File(
            os.getcwd()
            + "/results/shots_"
            + str(source_num)
            + "_ensemble_"
            + str(comm.ensemble_comm.rank)
            + name,
            comm=comm.comm,
        )
        return outfile


def display(comm, source_num):
    """Displays current shot and ensemble in terminal"""
    if comm.comm.rank == 0:
        print(
            "Timestepping for shot #",
            source_num + 1,
            " on ensemble member # ",
            comm.ensemble_comm.rank,
            "...",
            flush=True,
        )


def display_progress(comm, t):
    """Displays progress time"""
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(f"Simulation time is: {t:{10}.{4}} seconds", flush=True)


def parallel_print(string, comm):
    """Prints in parallel"""
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(string, flush=True)


def receivers_local(mesh, dimension, receiver_locations):
    """Locates receiveirs in cells"""
    if dimension == 2:
        return [mesh.locate_cell([z, x], tolerance=0.01) for z, x in receiver_locations]
    elif dimension == 3:
        return [
            mesh.locate_cell([z, x, y], tolerance=0.01)
            for z, x, y in receiver_locations
        ]
