import os

import numpy as np
from firedrake import VTKFile, Function, FunctionSpace, assemble, interpolate
from pyadjoint import stop_annotating

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
    if len(usol_recv) == 0:
        usol_recv = np.asarray(usol_recv)
    elif isinstance(usol_recv[0], Function):
        usol_recv = np.asarray([u.dat.data_wo for u in usol_recv])
    else:
        usol_recv = np.asarray(usol_recv)
    for ti in range(nt):
        for rn in range(nr):
            if is_local[rn] is None:
                usol_recv[ti][rn] = -99999.0
    return usol_recv


def _input_ordering_function_space(function_space):
    """Create the matching space on a VOM input-ordering mesh."""
    mesh = function_space.mesh()
    return FunctionSpace(mesh.input_ordering, function_space.ufl_element())


def _global_receiver_values_from_vom(
    receiver_values, receiver_function_space, comm,
):
    """Gather distributed VOM values in the user-provided receiver order."""
    if len(receiver_values) == 0:
        return np.asarray(receiver_values)

    input_space = _input_ordering_function_space(receiver_function_space)
    local_function = Function(receiver_function_space)
    ordered_values = []
    with stop_annotating():
        for value in receiver_values:
            if isinstance(value, Function):
                local_function.assign(value)
            else:
                local_function.dat.data_wo[:] = value
            input_ordered = assemble(interpolate(local_function, input_space))
            local_ordered = np.array(input_ordered.dat.data_ro, copy=True)
            ordered_values.append(np.concatenate(
                comm.comm.allgather(local_ordered), axis=0
            ))
    return np.asarray(ordered_values)


def _global_receiver_step_to_vom(observed_step, receiver_function_space):
    """Project one globally ordered receiver record onto the local VOM."""
    input_space = _input_ordering_function_space(receiver_function_space)
    observed_step = np.asarray(observed_step)
    with stop_annotating():
        observed_input_ordering = Function(input_space)
        local_count = observed_input_ordering.dat.data_wo.shape[0]
        all_counts = input_space.comm.allgather(local_count)
        offset = sum(all_counts[:input_space.comm.rank])
        observed_input_ordering.dat.data_wo[:] = observed_step[
            offset:offset + local_count
        ]
        return assemble(interpolate(
            observed_input_ordering, receiver_function_space
        ))


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
