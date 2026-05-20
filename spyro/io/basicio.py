from __future__ import with_statement

import pickle
from mpi4py import MPI
import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import griddata
import glob
import os
import warnings
from firedrake.__future__ import interpolate

fire.interpolate = interpolate


def delete_tmp_files(wave):
    """Delete temporary numpy files associated with a wave object."""
    str_id = f"*{wave.random_id_string}.npy"
    for file in glob.glob(str_id):
        os.remove(file)


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


def save_serial_data(wave, propagation_id):
    """
    Save serial data to numpy files.

    Args:
        wave (:class:`Wave`): The wave object containing the forward solution.
        propagation_id (int): The propagation ID.

    Returns:
        None
    """
    arrays_list = [obj.dat.data[:] for obj in wave.forward_solution]
    stacked_arrays = np.stack(arrays_list, axis=0)
    np.save(_shot_filename(propagation_id, wave, prefix='tmp_shot'), stacked_arrays)
    np.save(_shot_filename(propagation_id, wave, prefix='tmp_rec'), wave.forward_solution_receivers)


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
        stacked_shot_arrays = np.load(_shot_filename(propagation_id, wave, prefix='tmp_shot'))
        if len(wave.forward_solution) == 0:
            n_dts, n_dofs = np.shape(stacked_shot_arrays)
            rebuild_empty_forward_solution(wave, n_dts)
        for array_i, array in enumerate(stacked_shot_arrays):
            wave.forward_solution[array_i].dat.data[:] = array
        receiver_solution_filename = _shot_filename(propagation_id, wave, prefix='tmp_rec')
    else:
        receiver_solution_filename = _shot_filename(propagation_id, wave, prefix=file_name, random_str_in_use=False)
    wave.forward_solution_receivers = np.load(receiver_solution_filename, allow_pickle=True)


def ensemble_functional(func):
    """Decorator for gradient to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        comm = args[0].comm
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


def write_function_to_grid(function, V, grid_spacing, buffer=False):
    """Interpolate a Firedrake function to a structured grid

    Parameters
    ----------
    function : firedrake.Function
        Function to interpolate
    V : firedrake.FunctionSpace
        Function space of function
    grid_spacing : float
        Spacing of grid points
    buffer: boolean
        Determines if we use a buffer for the interpolation

    Returns
    -------
    vi : numpy.ndarray
        Interpolated values on grid points
    """
    # get DoF coordinates
    mesh = V.ufl_domain()
    W = fire.VectorFunctionSpace(mesh, V.ufl_element())
    coords = fire.assemble(fire.interpolate(mesh.coordinates, W))
    dimension, = coords.ufl_shape
    if dimension == 2:
        x, y = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif dimension == 3:
        x, y, z = coords.dat.data[:, 0], coords.dat.data[:, 1], coords.dat.data[:, 2]
    else:
        raise ValueError(f"Dimension of {dimension}, not supported, what are you doing?")

    # add buffer to avoid NaN when calling griddata
    pad = 0.005 if buffer else 0.0

    min_x = np.min(x) + pad
    max_x = np.max(x) - pad
    min_y = np.min(y) + pad
    max_y = np.max(y) - pad
    if dimension == 3:
        min_z = np.min(z) + pad
        max_z = np.max(z) - pad

    if min_x > max_x or min_y > max_y:
        raise ValueError("Buffer too large for the provided coordinate range.")

    if dimension == 3:
        if min_z > max_z:
            raise ValueError("Buffer too large for the provided coordinate range.")

    try:
        v = function.dat.data[:]
    except AttributeError:
        warnings.warn("Using numpy array instead of a firedrake function to interpolate.")
        v = function

    # target grid to interpolate to
    num_grid_x = int(round((max_x - min_x) / grid_spacing, 0)) + 1
    num_grid_y = int(round((max_y - min_y) / grid_spacing, 0)) + 1
    xi = np.linspace(min_x, max_x, num_grid_x)
    yi = np.linspace(min_y, max_y, num_grid_y)
    if dimension == 2:
        xi, yi = np.meshgrid(xi, yi)
    elif dimension == 3:
        num_grid_z = int(round((max_z - min_z) / grid_spacing, 0)) + 1
        zi = np.linspace(min_z, max_z, num_grid_z)
        xi, yi, zi = np.meshgrid(xi, yi, zi)

    # interpolate
    if dimension == 2:
        vi = griddata((x, y), v, (xi, yi), method="linear")
    elif dimension == 3:
        vi = griddata((x, y, z), v, (xi, yi, zi), method="linear")

    return vi


@ensemble_save
def save_shots(Wave_obj, file_name="shots/shot_record_", shot_ids=0):
    """Save a the shot record from last forward solve to a `pickle`.

    Parameters
    ----------
    Wave_obj: :class:`Wave` object
        A :class:`Wave` object
    source_id: int, optional by default 0
        The source number
    file_name: str, optional by default shot_number_#.dat
        The filename to save the data as a `pickle`

    Returns
    -------
    None

    """
    file_name = file_name + str(shot_ids) + ".dat"
    with open(file_name, "wb") as f:
        pickle.dump(Wave_obj.forward_solution_receivers, f)
    return None


def rebuild_empty_forward_solution(wave, time_steps):
    wave.forward_solution = []
    for i in range(time_steps):
        wave.forward_solution.append(fire.Function(wave.function_space))


@ensemble_load
def load_shots(Wave_obj, file_name="shots/shot_record_", shot_ids=0):
    """Load a `pickle` to a `numpy.ndarray`.

    Parameters
    ----------
    Wave_obj: :class:`Wave` object
        A :class:`Wave` object
    source_id: int, optional by default 0
        The source number
    filename: str, optional by default shot_number_#.dat
        The filename to save the data as a `pickle`

    Returns
    -------
    array: `numpy.ndarray`
        The data

    """
    array = np.zeros(())
    file_name = file_name + str(shot_ids) + ".dat"

    with open(file_name, "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)
        Wave_obj.forward_solution_receivers = array
    return None


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


def _check_units(c):
    if min(c.dat.data[:]) > 100.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def _grid_velocity_data_to_source_function(grid_velocity_data):
    """Build a CG1 Firedrake function on a structured mesh from grid data."""

    # Adding imports here to avoid circular imports
    from ..meshing.meshing_parameters import MeshingParameters
    from ..meshing.meshing_functions import AutomaticMesh

    vp_values = np.asarray(grid_velocity_data["vp_values"])
    length_z = grid_velocity_data["length_z"]
    length_x = grid_velocity_data["length_x"]
    length_y = grid_velocity_data.get("length_y")

    source_mesh_parameters = {
        "dimension": vp_values.ndim,
        "length_z": length_z,
        "length_x": length_x,
        "length_y": length_y,
        "mesh_type": "firedrake_mesh",
        "edge_length": grid_velocity_data["grid_spacing"],
        "abc_pad_length": grid_velocity_data.get("abc_pad_length"),
    }
    source_mesh = AutomaticMesh(
        MeshingParameters(input_mesh_dictionary=source_mesh_parameters)
    ).create_mesh()

    source_space = fire.FunctionSpace(source_mesh, "CG", 1)
    source = fire.Function(source_space)
    source_coords = source_mesh.coordinates.dat.data

    if vp_values.ndim == 2:
        z_nodes = np.unique(source_coords[:, 0])
        x_nodes = np.unique(source_coords[:, 1])
        z_index = np.searchsorted(z_nodes, source_coords[:, 0])
        x_index = np.searchsorted(x_nodes, source_coords[:, 1])
        source.dat.data[:] = vp_values[z_index, x_index]
    else:
        z_nodes = np.unique(source_coords[:, 0])
        x_nodes = np.unique(source_coords[:, 1])
        y_nodes = np.unique(source_coords[:, 2])
        z_index = np.searchsorted(z_nodes, source_coords[:, 0])
        x_index = np.searchsorted(x_nodes, source_coords[:, 1])
        y_index = np.searchsorted(y_nodes, source_coords[:, 2])
        source.dat.data[:] = vp_values[z_index, x_index, y_index]

    return source


def project_grid_velocity_data(grid_velocity_data, V):
    """Project a structured grid dictionary onto a Firedrake function space."""
    source = _grid_velocity_data_to_source_function(grid_velocity_data)
    c = fire.Function(V).interpolate(source, allow_missing_dofs=True)
    return _check_units(c)


def _hdf5_velocity_model_to_grid_velocity_data(Model, fname):
    """Convert an HDF5 velocity model into a grid velocity dictionary."""
    with h5py.File(fname, "r") as f:
        vp_values = np.asarray(f.get("velocity_model")[()])

    pad_length = Model.mesh_parameters.abc_pad_length
    pad_length = 0.0 if pad_length is None else pad_length

    if vp_values.ndim == 2:
        z_extent = Model.mesh_parameters.length_z + pad_length
        x_extent = Model.mesh_parameters.length_x + 2.0 * pad_length
        spacing_z = z_extent / float(vp_values.shape[0] - 1)
        spacing_x = x_extent / float(vp_values.shape[1] - 1)
        if not np.isclose(spacing_z, spacing_x):
            raise ValueError(
                "HDF5 grid is not uniformly spaced in z and x; cannot "
                "project it with a single Firedrake edge length."
            )
        grid_spacing = spacing_z
        length_y = None
    elif vp_values.ndim == 3:
        if Model.mesh_parameters.length_y is None:
            raise ValueError("3D HDF5 velocity model requires length_y.")

        z_extent = Model.mesh_parameters.length_z + pad_length
        x_extent = Model.mesh_parameters.length_x + 2.0 * pad_length
        y_extent = Model.mesh_parameters.length_y + 2.0 * pad_length
        spacing_z = z_extent / float(vp_values.shape[0] - 1)
        spacing_x = x_extent / float(vp_values.shape[1] - 1)
        spacing_y = y_extent / float(vp_values.shape[2] - 1)
        if not (np.isclose(spacing_z, spacing_x) and np.isclose(spacing_z, spacing_y)):
            raise ValueError(
                "HDF5 grid is not uniformly spaced in z, x and y; cannot "
                "project it with a single Firedrake edge length."
            )
        grid_spacing = spacing_z
        length_y = Model.mesh_parameters.length_y
    else:
        raise NotImplementedError("Only 2D and 3D HDF5 velocity models are supported.")

    return {
        "vp_values": vp_values,
        "grid_spacing": grid_spacing,
        "length_z": Model.mesh_parameters.length_z,
        "length_x": Model.mesh_parameters.length_x,
        "length_y": length_y,
        "abc_pad_length": pad_length,
    }


def interpolate(Model, fname, V):
    """Read and interpolate a seismic velocity model onto a Firedrake space.

    Parameters
    ----------
    Model: spyro object
        Model options and parameters.
    fname: str or dict
        The name of the HDF5 file containing the seismic velocity model, or
        a grid dictionary with keys such as ``vp_values``, ``length_z`` and
        ``length_x``.
    V: Firedrake.FunctionSpace object
        The space of the finite elements.

    Returns
    -------
    c: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes
        of the finite elements.

    """
    if isinstance(fname, dict):
        return project_grid_velocity_data(fname, V)
    elif isinstance(fname, str) and fname.endswith((".hdf5", ".h5")):
        grid_velocity_data = _hdf5_velocity_model_to_grid_velocity_data(Model, fname)
        return project_grid_velocity_data(grid_velocity_data, V)
    else:
        raise NotImplementedError


def read_mesh(mesh_parameters):
    """Reads in an external mesh and scatters it between cores.

    Parameters
    ----------
    model_parameters: spyro object
        Model options and parameters.

    Returns
    -------
    mesh: Firedrake.Mesh object
        The distributed mesh across `ens_comm`.
    """

    method = mesh_parameters.method
    ens_comm = mesh_parameters.comm
    num_propagations = ens_comm.ensemble_comm.size

    mshname = mesh_parameters.mesh_file

    if method == "CG_triangle" or method == "mass_lumped_triangle":
        mesh = fire.Mesh(
            mshname,
            comm=ens_comm.comm,
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    else:
        mesh = fire.Mesh(mshname, comm=ens_comm.comm)
    if ens_comm.comm.rank == 0 and ens_comm.ensemble_comm.rank == 0:
        print(
            "INFO: Distributing %d propagation(s) across %d core(s). \
                Each shot is using %d cores"
            % (
                num_propagations,
                fire.COMM_WORLD.size,
                fire.COMM_WORLD.size / ens_comm.ensemble_comm.size,
            ),
            flush=True,
        )
    print(
        "  rank %d on ensemble %d owns %d elements and can access %d vertices"
        % (
            mesh.comm.rank,
            ens_comm.ensemble_comm.rank,
            mesh.num_cells(),
            mesh.num_vertices(),
        ),
        flush=True,
    )

    return mesh


def parallel_print(string, comm):
    """
    Just prints a string once. Without any comm it just prints,
    without ensemble_comm it prints in comm 0,
    with ensemble_comm it prints in ensemble 0 and comm 0.

    Parameters
    ----------
    string: str
        The string to print
    comm: Firedrake.ensemble_communicator
        An ensemble communicator
    """
    if comm is None:
        print(string, flush=True)
    else:
        if getattr(comm, "ensemble_comm", None) is not None:
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print(string, flush=True)
        elif getattr(comm, "rank", None) is not None:
            if comm.rank == 0:
                print(string, flush=True)


def saving_source_and_receiver_location_in_csv(model, folder_name=None):
    """
    Saving the source and receiver locations in a csv file

    Parameters
    ----------
    model: spyro object
        Model options and parameters.
    folder_name: str, optional by default None
        The folder name to save the csv file
    """
    if folder_name is None:
        folder_name = "results/"

    file_name = folder_name + "sources.txt"
    file_obj = open(file_name, "w")
    file_obj.write("Z,\tX \n")
    for source in model["acquisition"]["source_locations"]:
        z, x = source
        string = str(z) + ",\t" + str(x) + " \n"
        file_obj.write(string)
    file_obj.close()

    file_name = folder_name + "receivers.txt"
    file_obj = open(file_name, "w")
    file_obj.write("Z,\tX \n")
    for receiver in model["acquisition"]["receiver_locations"]:
        z, x = receiver
        string = str(z) + ",\t" + str(x) + " \n"
        file_obj.write(string)
    file_obj.close()

    return None
