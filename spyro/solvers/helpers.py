import os

import numpy as np
from firedrake import VTKFile, Function, FunctionSpace, assemble, interpolate

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
    """Create the matching function space on a VOM input-ordering mesh.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Function space defined on a :func:`firedrake.VertexOnlyMesh`.

    Returns
    -------
    firedrake.FunctionSpace
        Function space with the same UFL element as ``function_space``, but
        defined on ``function_space.mesh().input_ordering``.

    Notes
    -----
    A vertex-only mesh (VOM) may distribute receiver points across MPI ranks.
    Firedrake also stores an ``input_ordering`` VOM that represents the same
    receiver points in the user-provided receiver order. Interpolating between
    these two VOMs is the safe way to move between local receiver ownership and
    Spyro's public global shot-record layout.
    """
    mesh = function_space.mesh()
    return FunctionSpace(mesh.input_ordering, function_space.ufl_element())


def _global_receiver_values_from_vom(receiver_functions, comm):
    """Gather local VOM receiver values in global receiver order.

    Parameters
    ----------
    receiver_functions : list of firedrake.Function
        Receiver values sampled on the distributed VOM, one function per time
        step. Each function only stores the receiver points owned by the local
        spatial MPI rank.
    comm : firedrake.Ensemble
        Spyro ensemble communicator. The spatial communicator is accessed as
        ``comm.comm`` and is used to gather the local input-ordering chunks.

    Returns
    -------
    numpy.ndarray
        Shot record with shape ``(n_timesteps, n_receivers)``. The receiver
        axis follows the original order supplied in the model dictionary.

    Notes
    -----
    The sampled receiver function lives on the distributed VOM. Its local data
    width is therefore the number of receiver points owned by the rank, not the
    global receiver count. This is why the older ``fill`` and ``MPI.MAX``
    reduction path is not valid for VOM data.

    The intermediate ``input_ordered_function`` is a Firedrake function on the
    VOM's ``input_ordering`` mesh. Firedrake builds this mesh so that receiver
    points are laid out in the same order as the user input. Interpolating the
    local VOM function onto this mesh gives each rank its local contiguous slice
    of the global shot-record order. Concatenating those slices across the
    spatial communicator produces the global record expected by the rest of
    Spyro.
    """
    if not receiver_functions:
        return np.asarray(receiver_functions)

    input_space = _input_ordering_function_space(
        receiver_functions[0].function_space()
    )
    receiver_values = []
    for receiver_function in receiver_functions:
        input_ordered_function = assemble(
            interpolate(receiver_function, input_space)
        )
        local_values = np.asarray(input_ordered_function.dat.data_ro)
        local_values = np.array(local_values, copy=True)
        all_values = comm.comm.allgather(local_values)
        receiver_values.append(np.concatenate(all_values, axis=0))

    return np.asarray(receiver_values)


def _global_receiver_step_to_vom(observed_step, receiver_function_space):
    """Project one global receiver record onto the local VOM space.

    Parameters
    ----------
    observed_step : array_like
        Receiver data for one time step in Spyro's public shot-record layout:
        one value per receiver, ordered as in the model dictionary.
    receiver_function_space : firedrake.FunctionSpace
        Function space of the local VOM receiver values for the same time step.

    Returns
    -------
    firedrake.Function
        ``observed_step`` represented on ``receiver_function_space``. On each
        MPI rank this function contains only the receiver values owned by that
        rank's local VOM partition.

    Notes
    -----
    This is the inverse operation of
    :func:`_global_receiver_values_from_vom` for a single time step. The global
    array is first copied into the local slice of the VOM ``input_ordering``
    function. Firedrake then interpolates from ``input_ordering`` back to the
    distributed VOM, giving a function that can be subtracted from the local
    simulated receiver function when computing per-timestep misfit values.
    """
    input_space = _input_ordering_function_space(receiver_function_space)
    observed_input_ordering = Function(input_space)
    observed_step = np.asarray(observed_step)

    local_count = observed_input_ordering.dat.data_wo.shape[0]
    all_counts = input_space.comm.allgather(local_count)
    offset = sum(all_counts[:input_space.comm.rank])
    observed_input_ordering.dat.data_wo[:] = observed_step[offset:offset + local_count]

    return assemble(
        interpolate(observed_input_ordering, receiver_function_space)
    )


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
