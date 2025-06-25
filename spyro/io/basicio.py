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


def _ensemble_save_load_loop(obj, func, kwargs, require_rank0=False):
    """
    Helper to loop over shots for ensemble parallelism in save or load operations.
    """
    _comm = obj.comm
    if obj.parallelism_type != "spatial" or obj.number_of_sources == 1:
        for propagation_id, shot_ids_in_propagation in enumerate(obj.shot_ids_per_propagation):
            if is_owner(_comm, propagation_id) and (not require_rank0 or _comm.comm.rank == 0):
                func(obj, **dict(kwargs, shot_ids=shot_ids_in_propagation))
    else:
        for snum in range(obj.number_of_sources):
            switch_serial_shot(obj, snum)
            if not require_rank0 or _comm.comm.rank == 0:
                func(obj, **dict(kwargs, shot_ids=[snum]))


def ensemble_save(func):
    """Decorator for read and write shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        obj = args[0]
        _ensemble_save_load_loop(obj, func, kwargs, require_rank0=True)
    return wrapper


def ensemble_load(func):
    """Decorator for read and write shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        obj = args[0]
        _ensemble_save_load_loop(obj, func, kwargs, require_rank0=False)
    return wrapper


def ensemble_propagator(func):
    """Decorator for forward to distribute shots for ensemble parallelism"""

    def wrapper(*args, **kwargs):
        if args[0].parallelism_type != "spatial" or args[0].number_of_sources == 1:
            shot_ids_per_propagation_list = args[0].shot_ids_per_propagation
            _comm = args[0].comm
            for propagation_id, shot_ids_in_propagation in enumerate(shot_ids_per_propagation_list):
                if is_owner(_comm, propagation_id):
                    u, u_r = func(*args, **dict(kwargs, source_nums=shot_ids_in_propagation))
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


def _shot_filename(prefix, propagation_id, wave):
    """Helper to construct filenames for shot/receiver data."""
    spatialcomm = wave.comm.comm.rank
    id_str = wave.random_id_string
    return f"{prefix}{propagation_id}_comm{spatialcomm}{id_str}.npy"


def save_serial_data(wave, propagation_id):
    """
    Save serial data to numpy files.

    Args:
        wave (Wave): The wave object containing the forward solution.
        propagation_id (int): The propagation ID.

    Returns:
        None
    """
    arrays_list = [obj.dat.data[:] for obj in wave.forward_solution]
    stacked_arrays = np.stack(arrays_list, axis=0)
    np.save(_shot_filename('tmp_shot', propagation_id, wave), stacked_arrays)
    np.save(_shot_filename('tmp_rec', propagation_id, wave), wave.forward_solution_receivers)


def switch_serial_shot(wave, propagation_id):
    """
    Switches the current serial shot for a given wave to shot identified with propagation ID.

    Args:
        wave (Wave): The wave object.
        propagation_id (int): The propagation ID.

    Returns:
        None
    """
    stacked_shot_arrays = np.load(_shot_filename('tmp_shot', propagation_id, wave))
    if len(wave.forward_solution) == 0:
        n_dts, n_dofs = np.shape(stacked_shot_arrays)
        rebuild_empty_forward_solution(wave, n_dts)
    for array_i, array in enumerate(stacked_shot_arrays):
        wave.forward_solution[array_i].dat.data[:] = array
    wave.forward_solution_receivers = np.load(_shot_filename('tmp_rec', propagation_id, wave))
    wave.receivers_output = wave.forward_solution_receivers


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


def write_function_to_grid(function, V, grid_spacing):
    """Interpolate a Firedrake function to a structured grid

    Parameters
    ----------
    function : firedrake.Function
        Function to interpolate
    V : firedrake.FunctionSpace
        Function space of function
    grid_spacing : float
        Spacing of grid points

    Returns
    -------
    xi : numpy.ndarray
        x coordinates of grid points
    yi : numpy.ndarray
        y coordinates of grid points
    zi : numpy.ndarray
        Interpolated values on grid points
    """
    # get DoF coordinates
    m = V.ufl_domain()
    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.interpolate(m.coordinates, W)
    x, y = coords.dat.data[:, 0], coords.dat.data[:, 1]

    # add buffer to avoid NaN when calling griddata
    min_x = np.amin(x) + 0.005
    max_x = np.amax(x) - 0.005
    min_y = np.amin(y) + 0.005
    max_y = np.amax(y) - 0.005

    try:
        z = function.dat.data[:]
    except AttributeError:
        warnings.warn("Using numpy array instead of a firedrake function to interpolate.")
        z = function

    # target grid to interpolate to
    xi = np.arange(min_x, max_x, grid_spacing)
    yi = np.arange(min_y, max_y, grid_spacing)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method="linear")

    return zi


def create_segy(function, V, grid_spacing, filename):
    """Write the velocity data into a segy file named filename

    Parameters
    ----------
    velocity:
        Firedrake function representing the values of the velocity
        model to save
    filename: str
        Name of the segy file to save

    Returns
    -------
    None
    """
    velocity = write_function_to_grid(function, V, grid_spacing)
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
    Wave_obj: `spyro.Wave` object
        A `spyro.Wave` object
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
    Wave_obj: `spyro.Wave` object
        A `spyro.Wave` object
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
    sd = V.mesh().geometric_dimension()
    m = V.ufl_domain()
    add_pad = False
    if Model.mesh_parameters.abc_pad_length is not None:
        if Model.mesh_parameters.abc_pad_length > 0.1:
            add_pad = True
    if add_pad:
        minz = -Model.mesh_parameters.length_z - Model.mesh_parameters.abc_pad_length
        maxz = 0.0
        minx = 0.0 - Model.mesh_parameters.abc_pad_length
        maxx = Model.mesh_parameters.length_x + Model.mesh_parameters.abc_pad_length
        miny = 0.0 - Model.mesh_parameters.abc_pad_length
        maxy = Model.mesh_parameters.length_y + Model.mesh_parameters.abc_pad_length
    else:
        minz = -Model.mesh_parameters.length_z
        maxz = 0.0
        minx = 0.0
        maxx = Model.mesh_parameters.length_x
        miny = 0.0
        maxy = Model.mesh_parameters.length_y

    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.interpolate(m.coordinates, W)
    # (z,x) or (z,x,y)
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
