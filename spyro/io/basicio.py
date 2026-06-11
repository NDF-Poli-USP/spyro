from __future__ import with_statement

import pickle
from mpi4py import MPI
import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import segyio
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
    m = V.ufl_domain()
    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.assemble(fire.interpolate(m.coordinates, W))
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
    xi = np.arange(min_x, max_x, grid_spacing)
    yi = np.arange(min_y, max_y, grid_spacing)
    if dimension == 2:
        xi, yi = np.meshgrid(xi, yi)
    elif dimension == 3:
        zi = np.arange(min_z, max_z, grid_spacing)
        xi, yi, zi = np.meshgrid(xi, yi, zi)

    # interpolate
    if dimension == 2:
        vi = griddata((x, y), v, (xi, yi), method="linear")
    elif dimension == 3:
        vi = griddata((x, y, z), v, (xi, yi, zi), method="linear")

    return vi


def create_segy(function, V, grid_spacing, filename):
    """Write the velocity data into a segy file named filename

    Parameters
    ----------
    function : firedrake.Function
        Function to interpolate
    V : firedrake.FunctionSpace
        Function space of function
    grid_spacing : float
        Spacing of grid points
    filename: str
        Name of the segy file to save

    Returns
    -------
    None
    """
    velocity = write_function_to_grid(function, V, grid_spacing, buffer=True)
    spec = segyio.spec()

    velocity = np.flipud(velocity.T)

    spec.sorting = 2  # not sure what this means
    spec.format = 1  # not sure what this means
    spec.samples = range(velocity.shape[0])
    spec.ilines = range(velocity.shape[1])
    spec.xlines = range(velocity.shape[0])

    assert np.sum(np.isnan(velocity[:])) == 0

    with segyio.create(filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = velocity[:, tr]


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


def interpolate(Model, fname, V):
    """Read and interpolate a seismic velocity model stored
    in a HDF5 file onto the nodes of a finite element space.

    Parameters
    ----------
    Model: spyro object
        Model options and parameters.
    fname: str
        The name of the HDF5 file containing the seismic velocity model.
    V: Firedrake.FunctionSpace object
        The space of the finite elements.

    Returns
    -------
    c: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes
        of the finite elements.

    """
    m = V.ufl_domain()

    add_pad = False
    if Model.mesh_parameters.abc_pad_length is not None:
        if Model.mesh_parameters.abc_pad_length > 1e-15:
            add_pad = True
    if add_pad:
        abc_pad_length = Model.mesh_parameters.abc_pad_length
        minz = -Model.mesh_parameters.length_z - abc_pad_length
        maxz = 0.0
        minx = 0.0 - abc_pad_length
        maxx = Model.mesh_parameters.length_x + abc_pad_length
        miny = 0.0 - abc_pad_length
        maxy = Model.mesh_parameters.length_y + abc_pad_length
    else:
        minz = -Model.mesh_parameters.length_z
        maxz = 0.0
        minx = 0.0
        maxx = Model.mesh_parameters.length_x
        miny = 0.0
        maxy = Model.mesh_parameters.length_y

    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.assemble(fire.interpolate(m.coordinates, W))
    # (z,x) or (z,x,y)
    sd = coords.dat.data.shape[1]
    if sd == 2:
        qp_z, qp_x = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif sd == 3:
        qp_z, qp_x, qp_y = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise NotImplementedError

    with h5py.File(fname, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])

        if sd == 2:
            nrow, ncol = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)

            # make sure no out-of-bounds
            qp_z2 = [
                minz if z < minz else maxz if z > maxz else z for z in qp_z
            ]
            qp_x2 = [
                minx if x < minx else maxx if x > maxx else x for x in qp_x
            ]

            interpolant = RegularGridInterpolator((z, x), Z)
            tmp = interpolant((qp_z2, qp_x2))
        elif sd == 3:
            nrow, ncol, ncol2 = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)
            y = np.linspace(miny, maxy, ncol2)

            # make sure no out-of-bounds
            qp_z2 = [
                minz if z < minz else maxz if z > maxz else z for z in qp_z
            ]
            qp_x2 = [
                minx if x < minx else maxx if x > maxx else x for x in qp_x
            ]
            qp_y2 = [
                miny if y < miny else maxy if y > maxy else y for y in qp_y
            ]

            interpolant = RegularGridInterpolator((z, x, y), Z)
            tmp = interpolant((qp_z2, qp_x2, qp_y2))

    c = fire.Function(V)
    c.dat.data[:] = tmp
    c = _check_units(c)
    return c


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


def read_segy_velocity_model(fname):
    """Read a velocity model from a SEG-Y file.

    Parameters
    ----------
    fname : str
        Filename of the SEG-Y velocity model.

    Returns
    -------
    vp : numpy.ndarray
        Velocity model array in ``(z, x)`` order.
    nx : int
        Number of traces in the SEG-Y file, corresponding to the x direction.
    nz : int
        Number of samples per trace, corresponding to the z direction.

    Raises
    ------
    ImportError
        If ``segyio`` is not installed.
    """
    with segyio.open(fname, "r", ignore_geometry=True) as segy:
        nx = len(segy.trace)
        nz = len(segy.samples)
        vp = np.zeros((nz, nx), dtype=np.float32)

        for i in range(nx):
            vp[:, i] = segy.trace[i]

    vp = np.flipud(vp)

    return vp, nx, nz


def _parse_axes_order(axes_order):
    """Convert an axis-order specification to axis names.

    Parameters
    ----------
    axes_order : str, tuple, or list
        Order of the axes in the raw binary file. Axis names may be provided as
        strings, such as ``"z x y"`` or ``"zxy"``. Numeric permutations may also
        be used, such as ``(0, 1, 2)``, ``(1, 0, 2)``, ``"1 0 2"``, or ``"201"``.

    Returns
    -------
    tuple of str
        Axis names in the order found in the raw binary file.

    Raises
    ------
    TypeError
        If ``axes_order`` is not a supported type or if a tuple/list mixes
        strings and integers.
    ValueError
        If the supplied axis order does not contain each axis exactly once.

    Examples
    --------
    >>> _parse_axes_order("z x y")
    ('z', 'x', 'y')
    >>> _parse_axes_order((1, 0, 2))
    ('x', 'z', 'y')
    >>> _parse_axes_order("201")
    ('y', 'z', 'x')
    """
    axis_from_int = {0: "z", 1: "x", 2: "y"}

    if isinstance(axes_order, str):
        clean = axes_order.lower().replace(",", " ").strip()
        parts = clean.split()

        if len(parts) == 1 and len(parts[0]) == 3:
            parts = list(parts[0])

        if all(part in ("0", "1", "2") for part in parts):
            ints = [int(part) for part in parts]
            if sorted(ints) != [0, 1, 2]:
                raise ValueError(
                    "numeric axes_order must contain 0, 1, 2 exactly once. "
                    "Use 0=z, 1=x, 2=y. Examples: '1 0 2', '201'."
                )
            return tuple(axis_from_int[i] for i in ints)

        if sorted(parts) != ["x", "y", "z"]:
            raise ValueError(
                "axes_order must contain x, y, z exactly once or 0, 1, 2 exactly once. "
                "Examples: 'z x y', 'zxy', (1, 0, 2), '1 0 2', '201'."
            )

        return tuple(parts)

    if isinstance(axes_order, (tuple, list)):
        if len(axes_order) != 3:
            raise ValueError("axes_order tuple/list must have length 3.")

        if all(isinstance(axis, (int, np.integer)) for axis in axes_order):
            ints = [int(axis) for axis in axes_order]
            if sorted(ints) != [0, 1, 2]:
                raise ValueError(
                    "numeric axes_order tuple/list must contain 0, 1, 2 exactly once. "
                    "Use 0=z, 1=x, 2=y. Example: (2, 0, 1)."
                )
            return tuple(axis_from_int[i] for i in ints)

        if all(isinstance(axis, str) for axis in axes_order):
            parts = [axis.lower().strip() for axis in axes_order]
            if sorted(parts) != ["x", "y", "z"]:
                raise ValueError(
                    "axis-name axes_order tuple/list must contain x, y, z exactly once. "
                    "Example: ('x', 'z', 'y')."
                )
            return tuple(parts)

        raise TypeError(
            "axes_order tuple/list must contain either all integers or all strings."
        )

    raise TypeError(
        "axes_order must be a string like 'z x y'/'zxy'/'1 0 2'/'102' "
        "or a tuple/list like (2, 0, 1)."
    )


def read_bin_velocity_model(
    filename,
    nz,
    nx,
    ny,
    byte_order="little",
    axes_order="z x y",
    axes_order_sort="C",
    dtype=np.float32,
):
    """Read a velocity model from a binary file.

    Parameters
    ----------
    filename : str
        Filename of the raw binary velocity model.
    nz : int
        Number of grid points in the z direction.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    byte_order : {'little', 'big'}, optional
        Byte order of the binary file. If the selected value
        produces NaN or Inf values, the other byte order is changed.
    axes_order : str, tuple, or list, optional
        Axis order in the raw binary file. It may be given as axis names, such
        as ``"z x y"`` or ``"x z y"``, or as numeric permutations using the
        convention ``0 = z``, ``1 = x``, and ``2 = y``.
    axes_order_sort : {'C', 'F'}, optional
        Memory layout used to reshape the raw binary values. Use ``'C'`` for
        C-order files and ``'F'`` for Fortran-order files.
    dtype : str or numpy.dtype, optional
        Floating-point dtype of the binary file. If the selected dtype does not
        match the expected file size, ``float32`` and ``float64`` are tested and
        the matching dtype is used.

    Returns
    -------
    vp : numpy.ndarray
        Velocity model in canonical ``(z, x, y)`` order.
    nz : int
        Number of grid points in the z direction.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.

    Raises
    ------
    ValueError
        If dimensions are missing, if ``byte_order`` or ``axes_order_sort`` is
        invalid, or if the file size does not match any supported dtype.
    """
    if None in (nz, nx, ny):
        raise ValueError(
            "Please specify the number of grid points in each dimension (nz, nx, ny)."
        )

    byte_order = str(byte_order).lower()
    if byte_order not in ("little", "big"):
        raise ValueError("byte_order must be 'little' or 'big'.")

    axes_order_sort = axes_order_sort.upper()
    if axes_order_sort not in ("C", "F"):
        raise ValueError("axes_order_sort must be 'C' or 'F'.")

    expected_elements = int(nz) * int(nx) * int(ny)
    actual_bytes = os.path.getsize(filename)

    dtype = np.dtype(dtype)
    expected_bytes = expected_elements * dtype.itemsize

    if actual_bytes != expected_bytes:
        dtype_candidates = [np.dtype("float32"), np.dtype("float64")]
        matched_dtype = None

        for candidate in dtype_candidates:
            candidate_bytes = expected_elements * candidate.itemsize
            if actual_bytes == candidate_bytes:
                matched_dtype = candidate
                break

        if matched_dtype is None:
            raise ValueError(
                f"File size mismatch: {filename}\n"
                f"Actual file size: {actual_bytes} bytes.\n"
                f"Expected elements: {expected_elements}.\n"
                f"Selected dtype={dtype} expects {expected_bytes} bytes.\n"
                "No supported dtype matched the file size. "
                "Supported dtype candidates: float32, float64."
            )

        warnings.warn(
            f"Selected dtype={dtype} does not match the file size. "
            f"Using dtype={matched_dtype} instead."
        )
        dtype = matched_dtype

    if byte_order == "little":
        dt = dtype.newbyteorder("<")
        other_byte_order = "big"
        other_dt = dtype.newbyteorder(">")
    else:
        dt = dtype.newbyteorder(">")
        other_byte_order = "little"
        other_dt = dtype.newbyteorder("<")

    print(f"Reading binary file: {filename}")
    print(f"Selected byte_order: {byte_order}")
    print(f"Selected/resolved dtype: {dtype}")

    vp = np.fromfile(filename, dtype=dt)

    if vp.size != expected_elements:
        raise ValueError(
            f"Unexpected number of values read from {filename}.\n"
            f"Expected {expected_elements}, got {vp.size}."
        )

    invalid_count = int(np.sum(~np.isfinite(vp)))

    if invalid_count > 0:
        vp_other = np.fromfile(filename, dtype=other_dt)
        other_invalid_count = int(np.sum(~np.isfinite(vp_other)))

        if other_invalid_count < invalid_count:
            warnings.warn(
                f"Selected byte_order='{byte_order}' produced {invalid_count} "
                f"NaN/Inf values. Using byte_order='{other_byte_order}' instead, "
                f"which produced {other_invalid_count} NaN/Inf values."
            )
            byte_order = other_byte_order
            vp = vp_other
        else:
            warnings.warn(
                f"Selected byte_order='{byte_order}' produced {invalid_count} "
                f"NaN/Inf values, but byte_order='{other_byte_order}' produced "
                f"{other_invalid_count}. Keeping byte_order='{byte_order}'."
            )

    raw_axes = _parse_axes_order(axes_order)
    sizes = {"z": nz, "x": nx, "y": ny}
    raw_shape = [sizes[axis] for axis in raw_axes]

    vp = vp.reshape(*raw_shape, order=axes_order_sort)

    final_axes = ("z", "x", "y")
    transpose_order = [raw_axes.index(axis) for axis in final_axes]

    vp = np.flipud(vp.transpose(transpose_order))

    return vp, nz, nx, ny


def write_velocity_model(
    filename: str,
    ofname: str = None,
    model_type: str = "bin",
    nz: int = None,
    nx: int = None,
    ny: int = None,
    byte_order: str = "little",
    axes_order: str = "z x y",
    axes_order_sort: str = "C",
    dtype: str = "float32",
):
    """Read and write a velocity model as an HDF5 file.

    Parameters
    ----------
    filename : str
        Filename of the velocity model. Supported model types are raw binary
        files and SEG-Y files.
    ofname : str, optional
        Output HDF5 filename. If the name does not end with ``.hdf5``, the
        extension is appended automatically. If omitted, ``filename`` is used.
    model_type : {'bin', 'segy'}, optional
        Type of input velocity model. Use ``'bin'`` for raw binary files and
        ``'segy'`` for SEG-Y files.
    nz : int, optional
        Required for binary velocity models. Number of grid points in the
        z direction.
    ny : int, optional
        Required for binary velocity models. Number of grid points in the
        y direction.
    nx : int, optional
        Required for binary velocity models. Number of grid points in the
        x direction.
    byte_order : {'little', 'big'}, optional
        Byte order of a binary velocity model. If the selected order produces
        NaN or Inf values, the opposite byte order is tested and used when it
        produces fewer invalid values.
    axes_order : str, tuple, or list, optional
        Axis order of a binary velocity model. Axis names such as ``'z x y'``
        and numeric permutations such as ``(0, 1, 2)`` are accepted. The numeric
        convention is ``0 = z``, ``1 = x``, and ``2 = y``.
    axes_order_sort : {'F', 'C'}, optional
        Sort style of the binary data. Use ``'F'`` for Fortran-style ordering
        and ``'C'`` for C-style ordering.
    dtype : str or numpy.dtype, optional
        Data type of a binary velocity model. Supported automatic size-match
        corrections are ``float32`` and ``float64``.

    Returns
    -------
    str
        Path to the written HDF5 file.

    Raises
    ------
    ValueError
        If ``model_type`` is not ``'bin'`` or ``'segy'``.

    Examples
    --------
    >>> write_velocity_model(
    ...     "avenir.segy",
    ...     ofname="avenir",
    ...     model_type="segy",
    ... )
    'avenir.hdf5'

    >>> write_velocity_model(
    ...     "avenir3d.bin",
    ...     ofname="avenir3d",
    ...     model_type="bin",
    ...     nz=401,
    ...     nx=201,
    ...     ny=201,
    ...     byte_order="little",
    ...     axes_order=(0, 1, 2),
    ...     axes_order_sort="F",
    ...     dtype="float32",
    ... )
    'avenir3d.hdf5'
    """

    model_type = model_type.lower()

    if ofname is None:
        warnings.warn("No output filename specified, name will be `filename`")
        ofname = filename

    if model_type == "bin":
        vp, nz, nx, ny = read_bin_velocity_model(
            filename=filename,
            nz=nz,
            nx=nx,
            ny=ny,
            byte_order=byte_order,
            axes_order=axes_order,
            axes_order_sort=axes_order_sort,
            dtype=dtype,
        )

    elif model_type == "segy":
        vp, nx, nz = read_segy_velocity_model(filename)
        ny = 0

    else:
        raise ValueError(
            "model_type must be either 'bin' or 'segy'. "
            f"Got model_type={model_type!r}."
        )

    if not str(ofname).endswith(".hdf5"):
        ofname = str(ofname) + ".hdf5"

    print(f"Writing velocity model: {ofname}", flush=True)

    with h5py.File(ofname, "w") as f:
        f.create_dataset("velocity_model", data=vp, dtype="f")
        f.attrs["shape"] = vp.shape
        f.attrs["units"] = "m/s"

    return ofname
