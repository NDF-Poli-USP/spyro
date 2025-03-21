import os

import numpy as np
from firedrake import VTKFile

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
    when it isn't local to any core

    Parameters
    ----------
    usol_recv : list
        List of numpy arrays
    is_local : list
        List of booleans indicating if the receiver is local to the core
    nt : int
        Number of timesteps
    nr : int
        Number of receivers

    Returns
    -------
    usol_recv : list
        List of numpy arrays

    """
    usol_recv = np.asarray(usol_recv)
    for ti in range(nt):
        for rn in range(nr):
            if is_local[rn] is None:
                usol_recv[ti][rn] = -99999.0
    return usol_recv


def create_output_file(name, comm, source_num):
    """Saves shots in output file

    Parameters
    ----------
    name : str
        Name of the output file
    comm : object
        MPI communicator
    source_num : int
        Source number

    Returns
    -------
    outfile : object
        Firedrake.File object
    """
    if io.is_owner(comm, source_num):
        outfile = VTKFile(
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
    """Displays current shot and ensemble in terminal

    Parameters
    ----------
    comm : object
        MPI communicator
    source_num : int
        Source number

    """
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
    """Displays progress time

    Parameters
    ----------
    comm : object
        MPI communicator
    t : float
        Current time
    """
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(f"Simulation time is: {t:{10}.{4}} seconds", flush=True)


def receivers_local(mesh, dimension, receiver_locations):
    """Locates receivers in cells

    Parameters
    ----------
    mesh : object
        Firedrake mesh object
    dimension : int
        Dimension of the mesh
    receiver_locations : list
        List of receiver locations

    Returns
    -------
    list
        List of receiver locations in cells
    """
    if dimension == 2:
        return [
            mesh.locate_cell([z, x], tolerance=0.01)
            for z, x in receiver_locations
        ]
    elif dimension == 3:
        return [
            mesh.locate_cell([z, x, y], tolerance=0.01)
            for z, x, y in receiver_locations
        ]
