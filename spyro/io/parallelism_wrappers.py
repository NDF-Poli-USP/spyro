import os
from mpi4py import MPI
import glob
import numpy as np
import firedrake as fire


def run_in_one_core(func):
    """Decorator to execute function only on rank 0.

    Ensures the decorated function runs only on the root process (rank 0)
    of the communicator. Other processes skip execution. Useful for I/O
    operations and serial tasks in parallel environments.

    Parameters
    ----------
    func : callable
        Function to decorate. The first argument of func must be an object
        with a `comm` attribute containing an MPI communicator.

    Returns
    -------
    callable
        Wrapped function that executes only on rank 0.

    Notes
    -----
    The function checks for two types of communicators:
    - Ensemble communicator: Runs only if both `ensemble_comm.rank` == 0
      and comm.rank == 0.
    - Regular communicator: Runs only if comm.rank == 0.
    - If comm is None, the function runs normally without restrictions.

    The function does not broadcast results to other processes.

    See Also
    --------
    run_in_one_core_and_broadcast : Similar decorator that also broadcasts results.

    Examples
    --------
    >>> @run_in_one_core
    ... def save_file(obj, filename):
    ...     # Only rank 0 writes the file
    ...     with open(filename, 'w') as f:
    ...         f.write(str(obj.data))
    """

    def wrapper(*args, **kwargs):
        comm = args[0].comm
        if comm is None:
            return func(*args, **kwargs)
        else:
            if getattr(comm, "ensemble_comm", None) is not None:
                if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                    return func(*args, **kwargs)
            elif getattr(comm, "rank", None) is not None:
                if comm.rank == 0:
                    return func(*args, **kwargs)

    return wrapper


def run_in_one_core_kwarg_comm(func):
    """Decorator to execute function only on rank 0.

    Ensures the decorated function runs only on the root process (rank 0)
    of the communicator. Other processes skip execution. Useful for I/O
    operations and serial tasks in parallel environments.

    Parameters
    ----------
    func : callable
        Function to decorate. The wrapped function must receive a `comm`
        keyword argument containing an MPI communicator.

    Returns
    -------
    callable
        Wrapped function that executes only on rank 0.

    Notes
    -----
    The function checks for two types of communicators:
    - Ensemble communicator: Runs only if both `ensemble_comm.rank` == 0
      and comm.rank == 0.
    - Regular communicator: Runs only if comm.rank == 0.
    - If comm is None, the function runs normally without restrictions.

    The function does not broadcast results to other processes.

    See Also
    --------
    run_in_one_core_and_broadcast : Similar decorator that also broadcasts results.

    Examples
    --------
    >>> @run_in_one_core
    ... def save_file(obj, filename):
    ...     # Only rank 0 writes the file
    ...     with open(filename, 'w') as f:
    ...         f.write(str(obj.data))
    """

    def wrapper(*args, **kwargs):
        comm = kwargs.get("comm")
        if comm is None:
            return func(*args, **kwargs)
        else:
            if getattr(comm, "ensemble_comm", None) is not None:
                if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                    return func(*args, **kwargs)
            elif getattr(comm, "rank", None) is not None:
                if comm.rank == 0:
                    return func(*args, **kwargs)

    return wrapper


def run_in_one_core_and_broadcast(func):
    """Decorator to execute function on rank 0 and broadcast result.

    Ensures the decorated function runs only on the root process (rank 0)
    and broadcasts the return value to all other processes. Useful for
    file reading and other operations that should be performed once but
    shared across all processes.

    Parameters
    ----------
    func : callable
        Function to decorate. The first argument of func must be an object
        with a 'comm' attribute containing an MPI communicator.

    Returns
    -------
    callable
        Wrapped function that executes on rank 0 and broadcasts the result
        to all processes.

    Notes
    -----
    The function handles two types of communicators:
    - Ensemble communicator: Executes on rank (0,0), broadcasts within
      ensemble, then within spatial communicator.
    - Regular communicator: Executes on rank 0, broadcasts to all.
    - If comm is None, the function runs normally without MPI operations.

    All processes receive the same return value from the broadcast.

    See Also
    --------
    run_in_one_core : Similar decorator without broadcasting.

    Examples
    --------
    >>> @run_in_one_core_and_broadcast
    ... def load_config(obj, filename):
    ...     # Only rank 0 reads the file, result shared with all
    ...     with open(filename, 'r') as f:
    ...         return json.load(f)
    """

    def wrapper(*args, **kwargs):
        comm = args[0].comm
        if comm is None:
            return func(*args, **kwargs)
        else:
            result = None
            if getattr(comm, "ensemble_comm", None) is not None:
                # Handle ensemble communicator
                if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                    result = func(*args, **kwargs)
                # Broadcast within ensemble
                result = comm.ensemble_comm.bcast(result, root=0)
                # Broadcast within spatial communicator
                result = comm.comm.bcast(result, root=0)
            elif getattr(comm, "rank", None) is not None:
                # Handle regular communicator
                if comm.rank == 0:
                    result = func(*args, **kwargs)
                result = comm.bcast(result, root=0)
            return result

    return wrapper


def _run_for_each_shot(obj, func, *args, **kwargs):
    """Helper to run a function for each shot in spatial parallelism."""
    results = []
    for snum in range(obj.number_of_sources):
        switch_serial_shot(obj, snum)
        results.append(func(*args, **kwargs))
    return results


def ensemble_shot_record(func):
    """Decorator for read and write shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        obj = args[0]
        if obj.parallelism_type == "spatial" and obj.number_of_sources > 1:
            return _run_for_each_shot(obj, func, *args, **kwargs)
    return wrapper


def switch_serial_shot(wave, propagation_id, file_name=None, just_for_dat_management=False):
    """
    Switches the current serial shot for a given wave to shot identified with propagation ID.

    Args:
        wave (:class:`Wave`): The wave object.
        propagation_id (int): The propagation ID.

    Returns:
        None
    """
    if file_name is None:
        forward_solution_filename = _shot_filename(propagation_id, wave, prefix='tmp_shot')
        if os.path.exists(forward_solution_filename) or wave.forward_solution:
            # The adjoint propagator consumes forward_solution with pop(). When
            # switching to the next shot, reload saved snapshots even if the
            # in-memory list has been emptied.
            stacked_shot_arrays = np.load(forward_solution_filename)
            if not wave.forward_solution:
                rebuild_empty_forward_solution(wave, len(stacked_shot_arrays))
            for array_i, array in enumerate(stacked_shot_arrays):
                wave.forward_solution[array_i].dat.data[:] = array
        receiver_solution_filename = _shot_filename(propagation_id, wave, prefix='tmp_rec')
    else:
        receiver_solution_filename = _shot_filename(propagation_id, wave, prefix=file_name, random_str_in_use=False)
    wave.forward_solution_receivers = np.load(receiver_solution_filename, allow_pickle=True)


def _shot_filename(propagation_id, wave, prefix='tmp', random_str_in_use=True):
    """
    Helper to construct filenames for shot/receiver data based on propagation and wave information.

    Parameters:
    -----------
    propagation_id (int): The index identifying the current propagation.

    wave (object): A :class:`Wave` object containing shot and communication information. Must have attributes:
        - shot_ids_per_propagation: A list or dict mapping propagation IDs to shot IDs.
        - comm: The current MPI communicator.
    prefix (str, optional): Prefix for the filename. Defaults to 'tmp'.
    random_str_in_use (bool, optional): If True, includes a random string and communicator rank in
        the filename, gotten from the Wave object, and uses '.npy' extension.
        If False, omits these and uses '.dat' extension. Defaults to True.

    Returns:
    --------
    str: The constructed filename.
    """
    shot_ids = wave.shot_ids_per_propagation[propagation_id]
    if random_str_in_use:
        id_str = wave.random_id_string
        spatialcomm = wave.comm.comm.rank
        comm__str = f"_comm{spatialcomm}"
        post_fix = "npy"
    else:
        id_str = ""
        comm__str = ""
        post_fix = "dat"
    return f"{prefix}{shot_ids}{comm__str}{id_str}.{post_fix}"


def rebuild_empty_forward_solution(wave, time_steps):
    wave.forward_solution = []
    for i in range(time_steps):
        wave.forward_solution.append(fire.Function(wave.function_space))


def ensemble_save(func):
    """Decorator for saving files with parallelism.

    Parameters:
    -----------
    func: The wrapped function that performs the actual saving operation.
    Expected to accept a :class:`Wave` based object as first argument.

    Returns:
    --------
    wrapper: A decorator function that wraps the original saving function with
        parallelism logic.

    Notes:
    ------
    Handles saving in different scenarions:
    - For ensemble parallelism or single source: iterates through propagations in
      each core and saves when the propagation is owned by the current rank.
    - For spatial-only parallelism with multiple sources: loads shots from temporary
      files using the switch_serial_shot method and saves to named output files
    - Requires first object to have attributes: `comm`, `parallelism_type`, `number_of_sources`,
      and `shot_ids_per_propagation`.
    - Temporary files are loaded via :meth:`switch_serial_shot()` when using spatial-only parallelism
    """
    def wrapper(*args, **kwargs):
        obj = args[0]  # Requires first arg to be an instant or subclass of Wave
        _comm = obj.comm
        if obj.parallelism_type != "spatial" or obj.number_of_sources == 1:
            for propagation_id, shot_ids_in_propagation in enumerate(obj.shot_ids_per_propagation):
                if is_owner(_comm, propagation_id) and _comm.comm.rank == 0:
                    func(obj, **dict(kwargs, shot_ids=shot_ids_in_propagation))
        else:
            # For spatial parallelism: load propagation data from tmp files (no file_name) then save wanted data to named files
            for snum in range(obj.number_of_sources):
                switch_serial_shot(obj, snum, file_name=None)  # Load from tmp files
                if _comm.comm.rank == 0:
                    func(obj, **dict(kwargs, shot_ids=[snum]))
    return wrapper


def ensemble_load(func):
    """Decorator for loading shots for ensemble parallelism.

    For spatial parallelism with multiple sources, loads from named files directly.

    Parameters:
    -----------
    func: The wrapped function that performs the actual loading operation.
    Expected to accept a :class:`Wave` based object as first argument.

    Returns:
    --------
    wrapper: A decorator function that wraps the original loading function with
        parallelism logic.
    """
    def wrapper(*args, **kwargs):
        obj = args[0]
        _comm = obj.comm
        if obj.parallelism_type != "spatial" or obj.number_of_sources == 1:
            for propagation_id, shot_ids_in_propagation in enumerate(obj.shot_ids_per_propagation):
                if is_owner(_comm, propagation_id):
                    func(obj, **dict(kwargs, shot_ids=shot_ids_in_propagation))
        else:
            # For spatial parallelism: load data directly from named files (no switch_serial_shot needed)
            for snum in range(obj.number_of_sources):
                func(obj, **dict(kwargs, shot_ids=[snum]))
    return wrapper


def is_owner(ens_comm, rank):
    """Distribute shots between processors in using a modulus operator

    Parameters
    ----------
    ens_comm: Firedrake.ensemble_communicator
        An ensemble communicator
    rank: int
        The rank of the core

    Returns
    -------
    boolean
        `True` if `rank` owns this shot

    """
    owner = ens_comm.ensemble_comm.rank == (rank % ens_comm.ensemble_comm.size)
    return owner


def delete_tmp_files(wave):
    """Delete temporary numpy files associated with a wave object."""
    str_id = f"*{wave.random_id_string}.npy"
    for file in glob.glob(str_id):
        os.remove(file)


def ensemble_propagator(func):
    """Decorator for forward to distribute shots for ensemble parallelism

    Parameters:
    -----------
    func: The wrapped function that performs the actual propagation operation.
    Expected to accept a :class:`Wave` based object as first argument.

    Returns:
    --------
    wrapper: A decorator function that wraps the original propagator function with
        ensemble parallelism logic.
    """

    def wrapper(*args, **kwargs):
        if args[0].parallelism_type != "spatial" or args[0].number_of_sources == 1:
            shot_ids_per_propagation_list = args[0].shot_ids_per_propagation
            _comm = args[0].comm
            for propagation_id, shot_ids_in_propagation in enumerate(shot_ids_per_propagation_list):
                if is_owner(_comm, propagation_id):
                    func(*args, **dict(kwargs, source_nums=shot_ids_in_propagation))
        elif args[0].parallelism_type == "spatial" and args[0].number_of_sources > 1:
            num = args[0].number_of_sources
            starting_time = args[0].current_time
            for snum in range(num):
                args[0].reset_pressure()
                args[0].current_time = starting_time
                func(*args, **dict(kwargs, source_nums=[snum]))
                save_serial_data(args[0], snum)

    return wrapper


def save_serial_data(wave, propagation_id):
    """
    Save serial data to numpy files.

    Args:
        wave (:class:`Wave`): The wave object containing the forward solution.
        propagation_id (int): The propagation ID.

    Returns:
        None
    """
    if wave.forward_solution:
        # There are cases where forward_solution is empty, e.g. when running
        # forward_solve for the true model. In that case, we skip saving the
        # solution on the entire domain, which is not needed.
        arrays_list = [obj.dat.data[:] for obj in wave.forward_solution]
        stacked_arrays = np.stack(arrays_list, axis=0)
        np.save(_shot_filename(propagation_id, wave, prefix='tmp_shot'), stacked_arrays)
    np.save(_shot_filename(propagation_id, wave, prefix='tmp_rec'), wave.forward_solution_receivers)


def ensemble_functional(func):
    """Decorator for gradient to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        comm = args[0].comm
        if args[0].adjoint_type.name == "AUTOMATED_ADJOINT":
            # pyadjoint needs the annotated Firedrake object, not a numpy scalar
            # produced by the ensemble reduction path below.
            return func(*args, **kwargs)
        if args[0].parallelism_type != "spatial" or args[0].number_of_sources == 1:
            J = func(*args, **kwargs)
            J_total = np.zeros((1))
            J_total[0] += J
            J_total = fire.COMM_WORLD.allreduce(J_total, op=MPI.SUM)
            J_total[0] /= comm.comm.size

        elif args[0].parallelism_type == "spatial" and args[0].number_of_sources > 1:
            residual_list = args[1]
            J_total = np.zeros((1))

            for snum in range(args[0].number_of_sources):
                switch_serial_shot(args[0], snum)
                current_residual = residual_list[snum]
                J = func(args[0], current_residual)
                J_total += J
            J_total[0] /= comm.comm.size

            comm.comm.barrier()

        return J_total[0]

    return wrapper


def ensemble_gradient(func):
    """Decorator for gradient to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        comm = args[0].comm
        if args[0].parallelism_type != "spatial" or args[0].number_of_sources == 1:
            shot_ids_per_propagation_list = args[0].shot_ids_per_propagation
            for propagation_id, shot_ids_in_propagation in enumerate(shot_ids_per_propagation_list):
                if is_owner(comm, propagation_id):
                    grad = func(*args, **kwargs)
            grad_total = fire.Function(args[0].function_space)

            comm.comm.barrier()
            grad_total = comm.allreduce(grad, grad_total)
            grad_total /= comm.ensemble_comm.size

            return grad_total
        elif args[0].parallelism_type == "spatial" and args[0].number_of_sources > 1:
            num = args[0].number_of_sources
            starting_time = args[0].current_time
            grad_total = fire.Function(args[0].function_space)
            misfit_list = kwargs.get("misfit")

            for snum in range(num):
                switch_serial_shot(args[0], snum)
                current_misfit = misfit_list[snum]
                args[0].reset_pressure()
                args[0].current_time = starting_time
                grad = func(*args,
                            **dict(
                                kwargs,
                                misfit=current_misfit,
                            )
                            )
                grad_total += grad

            grad_total /= num
            comm.comm.barrier()

            return grad_total

    return wrapper
