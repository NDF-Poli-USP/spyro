"""IO utilities for spyro wave propagation.

This module provides functions for managing input/output operations related to
wave propagation, including file I/O for shots, receivers, and mesh data. It
includes decorators for handling parallel I/O across ensemble and spatial
parallelism modes.
"""

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
    """Run a function for each shot in spatial parallelism."""
    results = []
    for snum in range(obj.number_of_sources):
        switch_serial_shot(obj, snum)
        results.append(func(*args, **kwargs))
    return results


def ensemble_shot_record(func):
    """Decorate to read and write shots for ensemble parallelism."""

    def wrapper(*args, **kwargs):
        obj = args[0]
        if obj.parallelism_type == "spatial" and obj.number_of_sources > 1:
            return _run_for_each_shot(obj, func, *args, **kwargs)

    return wrapper


def ensemble_save(func):
    """Decorate to save files with parallelism.

    Parameters
    ----------
    func : callable
        The wrapped function that performs the actual saving operation.
        Expected to accept a `Wave` based object as the first argument.

    Returns
    -------
    wrapper : callable
        A decorator function that wraps the original saving function with
        parallelism logic.

    Notes
    -----
    Handles saving in different scenarios:

    - For ensemble parallelism or single source: iterates through propagations
      in each core and saves when the propagation is owned by the current rank.
    - For spatial-only parallelism with multiple sources: loads shots from
      temporary files using the `switch_serial_shot` method and saves to named
      output files.
    - Requires first object to have attributes: `comm`, `parallelism_type`,
      `number_of_sources`, and `shot_ids_per_propagation`.
    - Temporary files are loaded via `switch_serial_shot()` when using
      spatial-only parallelism.
    """

    def wrapper(*args, **kwargs):
        """Define the decorator function."""
        obj = args[0]  # Requires first arg to be an instant or subclass of Wave
        _comm = obj.comm
        if obj.parallelism_type != "spatial" or obj.number_of_sources == 1:
            for propagation_id, shot_ids_in_propagation in enumerate(
                obj.shot_ids_per_propagation
            ):
                if is_owner(_comm, propagation_id) and _comm.comm.rank == 0:
                    func(obj, **dict(kwargs, shot_ids=shot_ids_in_propagation))
        else:
            # For spatial parallelism: load propagation data from tmp files
            # (no file_name) then save wanted data to named files
            for snum in range(obj.number_of_sources):
                switch_serial_shot(obj, snum, file_name=None)
                if _comm.comm.rank == 0:
                    func(obj, **dict(kwargs, shot_ids=[snum]))

    return wrapper


def ensemble_load(func):
    """Decorate to load shots for ensemble parallelism.

    For spatial parallelism with multiple sources, loads from named files
    directly.

    Parameters
    ----------
    func : callable
        The wrapped function that performs the actual loading operation.
        Expected to accept a :class:`Wave` based object as first argument.

    Returns
    -------
    wrapper : callable
        A decorator function that wraps the original loading function with
        parallelism logic.
    """

    def wrapper(*args, **kwargs):
        """Define the decorator function."""
        obj = args[0]
        _comm = obj.comm
        if obj.parallelism_type != "spatial" or obj.number_of_sources == 1:
            for propagation_id, shot_ids_in_propagation in enumerate(
                obj.shot_ids_per_propagation
            ):
                if is_owner(_comm, propagation_id):
                    func(obj, **dict(kwargs, shot_ids=shot_ids_in_propagation))
        else:
            # For spatial parallelism: load data directly from named files
            # (no switch_serial_shot needed)
            for snum in range(obj.number_of_sources):
                func(obj, **dict(kwargs, shot_ids=[snum]))

    return wrapper


def ensemble_propagator(func):
    """Decorate to distribute shots for ensemble parallelism.

    Parameters
    ----------
    func : callable
        The wrapped function that performs the actual propagation operation.
        Expected to accept a :class:`Wave` based object as first argument.

    Returns
    -------
    wrapper : callable
        A decorator function that wraps the original propagator function with
        ensemble parallelism logic.
    """

    def wrapper(*args, **kwargs):
        """Define the decorator function."""
        if args[0].parallelism_type != "spatial" or args[0].number_of_sources == 1:
            shot_ids_per_propagation_list = args[0].shot_ids_per_propagation
            _comm = args[0].comm
            for propagation_id, shot_ids_in_propagation in enumerate(
                shot_ids_per_propagation_list
            ):
                if is_owner(_comm, propagation_id):
                    u, u_r = func(
                        *args,
                        **dict(kwargs, source_nums=shot_ids_in_propagation),
                    )
                    return u, u_r
        elif args[0].parallelism_type == "spatial" and args[0].number_of_sources > 1:
            num = args[0].number_of_sources
            starting_time = args[0].current_time
            for snum in range(num):
                args[0].reset_pressure()
                args[0].current_time = starting_time
                u, u_r = func(*args, **dict(kwargs, source_nums=[snum]))
                save_serial_data(args[0], snum)

            return u, u_r

    return wrapper


def _shot_filename(propagation_id, wave, prefix="tmp", random_str_in_use=True):
    """Construct filename for shot/receiver data.

    Helper to construct filenames for shot/receiver data based on propagation
    and wave information.

    Parameters
    ----------
    propagation_id : int
        The index identifying the current propagation.
    wave : :class:`Wave`
        A Wave object containing shot and communication information. Must have:
        shot_ids_per_propagation, comm attributes.
    prefix : str, optional
        Prefix for the filename. Default is 'tmp'.
    random_str_in_use : bool, optional
        If True, includes random string and communicator rank in filename
        and uses '.npy' extension. If False, uses '.dat' extension.
        Default is True.

    Returns
    -------
    str
        The constructed filename.
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
    """Save serial forward solution data to NumPy files.

    Parameters
    ----------
    wave : :class:`Wave`
        The wave object containing the forward solution.
    propagation_id : int
        The propagation ID.

    Returns
    -------
    None
    """
    arrays_list = [obj.dat.data[:] for obj in wave.forward_solution]
    stacked_arrays = np.stack(arrays_list, axis=0)
    np.save(_shot_filename(propagation_id, wave, prefix="tmp_shot"), stacked_arrays)
    np.save(
        _shot_filename(propagation_id, wave, prefix="tmp_rec"),
        wave.forward_solution_receivers,
    )


def switch_serial_shot(
    wave, propagation_id, file_name=None, just_for_dat_management=False
):
    """Switch the current serial shots to specified propagation.

    Switches the current serial shots for a given wave to the shots identified
    by propagation ID.

    Parameters
    ----------
    wave : :class:`Wave`
        The wave object.
    propagation_id : int
        The propagation ID identifying which shots to load.
    file_name : str, optional
        The file name prefix. If None, loads from temporary files.
    just_for_dat_management : bool, optional
        Flag for data management. Default is False.

    Returns
    -------
    None
    """
    if file_name is None:
        stacked_shot_arrays = np.load(
            _shot_filename(propagation_id, wave, prefix="tmp_shot")
        )
        if len(wave.forward_solution) == 0:
            n_dts, n_dofs = np.shape(stacked_shot_arrays)
            rebuild_empty_forward_solution(wave, n_dts)
        for array_i, array in enumerate(stacked_shot_arrays):
            wave.forward_solution[array_i].dat.data[:] = array
        receiver_solution_filename = _shot_filename(
            propagation_id, wave, prefix="tmp_rec"
        )
    else:
        receiver_solution_filename = _shot_filename(
            propagation_id, wave, prefix=file_name, random_str_in_use=False
        )
    wave.forward_solution_receivers = np.load(
        receiver_solution_filename, allow_pickle=True
    )
    wave.receivers_output = wave.forward_solution_receivers


def ensemble_functional(func):
    """Decorate for functional computation in ensemble parallelism."""

    def wrapper(*args, **kwargs):
        """Define the decorator function."""
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
    """Decorate to distribute shots for gradient computation in ensemble parallelism."""

    def wrapper(*args, **kwargs):
        """Define the decorator function."""
        comm = args[0].comm
        if args[0].parallelism_type != "spatial" or args[0].number_of_sources == 1:
            shot_ids_per_propagation_list = args[0].shot_ids_per_propagation
            for propagation_id, shot_ids_in_propagation in enumerate(
                shot_ids_per_propagation_list
            ):
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
                grad = func(
                    *args,
                    **dict(
                        kwargs,
                        misfit=current_misfit,
                    ),
                )
                grad_total += grad

            grad_total /= num
            comm.comm.barrier()

            return grad_total

    return wrapper


def write_function_to_grid(function, V, grid_spacing, buffer=False):
    """Interpolate a Firedrake function to a structured grid.

    Parameters
    ----------
    function : firedrake.Function
        Function to interpolate.
    V : firedrake.FunctionSpace
        Function space of the function.
    grid_spacing : float
        Spacing of grid points.
    buffer : bool, optional
        Whether to use a buffer for the interpolation. Default is False.

    Returns
    -------
    numpy.ndarray
        Interpolated values on grid points.
    """
    # get DoF coordinates
    m = V.ufl_domain()
    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.assemble(fire.interpolate(m.coordinates, W))
    (dimension,) = coords.ufl_shape
    if dimension == 2:
        x, y = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif dimension == 3:
        x, y, z = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise ValueError(
            f"Dimension of {dimension}, not supported, what are you doing?"
        )

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
        warnings.warn(
            "Using numpy array instead of a firedrake function to interpolate."
        )
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
    """Write velocity data to a SEG-Y file.

    Parameters
    ----------
    function : firedrake.Function
        Function to interpolate.
    V : firedrake.FunctionSpace
        Function space of the function.
    grid_spacing : float
        Spacing of grid points.
    filename : str
        Name of the SEG-Y file to save.

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
    """Save the shot record from last forward solve to a pickle file.

    Parameters
    ----------
    Wave_obj : :class:`Wave`
        A :class:`Wave`  object.
    file_name : str, optional
        The filename to save the data to. Default is 'shots/shot_record_'.
    shot_ids : int, optional
        The shot number. Default is 0.

    Returns
    -------
    None
    """
    file_name = file_name + str(shot_ids) + ".dat"
    with open(file_name, "wb") as f:
        pickle.dump(Wave_obj.forward_solution_receivers, f)
    return None


def rebuild_empty_forward_solution(wave, time_steps):
    """Rebuild the forward solution list with empty functions.

    Parameters
    ----------
    wave : :class:`Wave`
        The :class:`Wave` object to rebuild.
    time_steps : int
        Number of time steps to create functions for.

    Returns
    -------
    None
    """
    wave.forward_solution = []
    for i in range(time_steps):
        wave.forward_solution.append(fire.Function(wave.function_space))


@ensemble_load
def load_shots(Wave_obj, file_name="shots/shot_record_", shot_ids=0):
    """Load shot data from a pickle file to a NumPy array.

    Parameters
    ----------
    Wave_obj : :class:`Wave`
        A :class:`Wave`  object.
    file_name : str, optional
        The filename to load the data from. Default is 'shots/shot_record_'.
    shot_ids : int, optional
        The shot number. Default is 0.

    Returns
    -------
    None
    """
    array = np.zeros(())
    file_name = file_name + str(shot_ids) + ".dat"

    with open(file_name, "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)
        Wave_obj.forward_solution_receivers = array
    return None


def is_owner(ens_comm, rank):
    """Determine shot ownership using modulus operator distribution.

    Parameters
    ----------
    ens_comm : Firedrake.ensemble_communicator
        A Firedrake ensemble communicator.
    rank : int
        The rank of the core.

    Returns
    -------
    bool
        True if `rank` owns this shot.
    """
    owner = ens_comm.ensemble_comm.rank == (rank % ens_comm.ensemble_comm.size)
    return owner


def _check_units(c):
    """Verify and convert velocity units from m/s to km/s if needed.

    Parameters
    ----------
    c : firedrake.Function
        Velocity field to check.

    Returns
    -------
    firedrake.Function
        Velocity field with units in km/s.
    """
    if min(c.dat.data[:]) > 100.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def interpolate(Model, fname, V):
    """Read and interpolate a seismic velocity model from HDF5 file.

    Interpolates a seismic velocity model stored in a HDF5 file onto the
    nodes of a finite element space.

    Parameters
    ----------
    Model : spyro_obj
        Model options and parameters object.
    fname : str
        Path to the HDF5 file containing the seismic velocity model.
    V : firedrake.FunctionSpace
        The finite element space for interpolation.

    Returns
    -------
    c : firedrake.Function
        P-wave seismic velocity interpolated onto the FE nodes.
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
            qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
            qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]

            interpolant = RegularGridInterpolator((z, x), Z)
            tmp = interpolant((qp_z2, qp_x2))
        elif sd == 3:
            nrow, ncol, ncol2 = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)
            y = np.linspace(miny, maxy, ncol2)

            # make sure no out-of-bounds
            qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
            qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]
            qp_y2 = [miny if y < miny else maxy if y > maxy else y for y in qp_y]

            interpolant = RegularGridInterpolator((z, x, y), Z)
            tmp = interpolant((qp_z2, qp_x2, qp_y2))

    c = fire.Function(V)
    c.dat.data[:] = tmp
    c = _check_units(c)
    return c


def read_mesh(mesh_parameters):
    """Read external mesh and distribute across processors.

    Parameters
    ----------
    mesh_parameters : mesh_parameters_obj
        Mesh parameters object containing method, comm, and mesh_file.

    Returns
    -------
    mesh : firedrake.Mesh
        The distributed mesh across ensemble communicator.
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
    """Print a string once from appropriate rank.

    Prints the string only once: from rank 0 if no ensemble_comm, or from
    ensemble rank 0 and comm rank 0 if ensemble_comm is present.

    Parameters
    ----------
    string : str
        The string to print.
    comm : Firedrake.ensemble_communicator, optional
        A Firedrake ensemble communicator or standard MPI communicator.
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
    """Save source and receiver locations to CSV files.

    Parameters
    ----------
    model : dict
        Model dictionary with acquisition parameters.
    folder_name : str, optional
        Folder to save CSV files. Default is 'results/'.

    Returns
    -------
    None
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
