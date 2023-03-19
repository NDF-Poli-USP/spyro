from __future__ import with_statement

import os
import pickle

import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import segyio

from .. import domains


def ensemble_save(func):
    """Decorator for read and write shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[1]
        custom_file_name = kwargs.get('file_name')
        for snum in range(num):
            if is_owner(_comm, snum) and _comm.comm.rank == 0:
                if custom_file_name is None:
                    func(*args, **dict(kwargs, file_name = "shots/shot_record_"+str(snum+1)+".dat"))
                else:
                    func(*args, **dict(kwargs, file_name = custom_file_name+"_"+str(snum+1)+".dat"))
    return wrapper


def ensemble_load(func):
    """Decorator for read and write shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[1]
        custom_file_name = kwargs.get('file_name')
        for snum in range(num):
            if is_owner(_comm, snum):
                if custom_file_name is None:
                    values = func(*args, **dict(kwargs, file_name = "shots/shot_record_"+str(snum+1)+".dat"))
                else:
                    values = func(*args, **dict(kwargs, file_name = custom_file_name+"_"+str(snum+1)+".dat"))
                return values 
    return wrapper


def ensemble_plot(func):
    """Decorator for `plot_shots` to distribute shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[1]
        for snum in range(num):
            if is_owner(_comm, snum) and _comm.comm.rank == 0:
                func(*args, **dict(kwargs, file_name = str(snum+1)))

    return wrapper


def ensemble_forward(func):
    """Decorator for forward to distribute shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                u, u_r = func(*args, **dict(kwargs, source_num=snum))
                return u, u_r

    return wrapper


def ensemble_solvers_ad(func):
    """Decorator for fwi to distribute shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        solver_type = args[0]
        num = args[1]
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                if solver_type == "fwi":
                    Jm, dm0 = func(*args, **dict(kwargs, sn=snum))
                    return Jm, dm0
                elif solver_type == "fwd":
                    func(*args, **dict(kwargs, sn=snum))
                else:
                    assert (), "This solver_type is not avaiable"
               
    return wrapper


def ensemble_forward_elastic_waves(func):
    """Decorator for forward elastic waves to distribute shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                u, uz_r, ux_r, uy_r = func(*args, **dict(kwargs, source_num=snum))
                return u, uz_r, ux_r, uy_r

    return wrapper


def ensemble_gradient(func):
    """Decorator for gradient to distribute shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        save_adjoint = kwargs.get("save_adjoint")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                if save_adjoint:
                    grad, u_adj = func(*args, **kwargs)
                    return grad, u_adj
                else:
                    grad = func(*args, **kwargs)
                    return grad

    return wrapper


def ensemble_gradient_elastic_waves(func):
    """Decorator for gradient (elastic waves) to distribute shots for ensemble parallelism"""
    def wrapper(*args, **kwargs):
        acq = args[0].get("acquisition")
        save_adjoint = kwargs.get("save_adjoint")
        num = len(acq["source_pos"])
        _comm = args[2]
        for snum in range(num):
            if is_owner(_comm, snum):
                if save_adjoint:
                    grad_lambda, grad_mu, u_adj = func(*args, **kwargs)
                    return grad_lambda, grad_mu, u_adj
                else:
                    grad_lambda, grad_mu = func(*args, **kwargs)
                    return grad_lambda, grad_mu

    return wrapper


def write_function_to_grid(function, V, grid_spacing):
    """Interpolate a Firedrake function to a structured grid"""
    # get DoF coordinates
    m = V.ufl_domain()
    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.interpolate(m.coordinates, W)
    x, y = coords.dat.data[:, 0], coords.dat.data[:, 1]

    # add buffer to avoid NaN when calling griddata
    min_x = np.amin(x) + 0.01
    max_x = np.amax(x) - 0.01
    min_y = np.amin(y) + 0.01
    max_y = np.amax(y) - 0.01

    #z = function.dat.data[:] * 1000.0  # convert from km/s to m/s
    z = function.dat.data[:] 

    # target grid to interpolate to
    xi = np.arange(min_x, max_x, grid_spacing)
    yi = np.arange(min_y, max_y, grid_spacing)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method="linear")

    return xi, yi, zi


def create_segy(velocity, filename):
    """Write the velocity data into a segy file named filename"""
    spec = segyio.spec()

    velocity = np.flipud(velocity.T)

    spec.sorting = 2 # not sure what this means
    spec.format = 1 # not sure what this means
    spec.samples = range(velocity.shape[0])
    spec.ilines = range(velocity.shape[1])
    spec.xlines = range(velocity.shape[0])

    assert np.sum(np.isnan(velocity[:])) == 0

    with segyio.create(filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = velocity[:, tr]


@ensemble_save
def save_shots(model, comm, array, file_name=None):
    """Save a `numpy.ndarray` to a `pickle`.

    Parameters
    ----------
    filename: str, optional by default shot_number_#.dat
        The filename to save the data as a `pickle`
    array: `numpy.ndarray`
        The data to save a pickle (e.g., a shot)

    Returns
    -------
    None

    """
    with open(file_name, "wb") as f:
        pickle.dump(array, f)
    return None


@ensemble_load
def load_shots(model, comm, file_name=None):
    """Load a `pickle` to a `numpy.ndarray`.

    Parameters
    ----------
    filename: str, optional by default shot_number_#.dat
        The filename to save the data as a `pickle`

    Returns
    -------
    array: `numpy.ndarray`
        The data

    """

    with open(file_name, "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)
    return array


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
    return ens_comm.ensemble_comm.rank == (rank % ens_comm.ensemble_comm.size)




def _check_units(c):
    #if min(c.dat.data[:]) > 100.0:
    if max(c.dat.data[:]) > 1000.0: # FIXME check with Alexandre
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def interpolate(fname, model, mesh, V, guess=False, field="velocity_model"):
    """Read and interpolate a seismic velocity or density model stored
    in a HDF5 file onto the nodes of a finite element space.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    mesh: Firedrake.mesh object
        A mesh object read in by Firedrake.
    V: Firedrake.FunctionSpace object
        The space of the finite elements.
    guess: boolean, optinal
        Is it a guess model or a `exact` model?
    field: string
        velocity_model (default) or density_model

    Returns
    -------
    c: Firedrake.Function
        P-wave (or S-wave) seismic velocity or density model interpolated onto the nodes of the finite elements.

    """
    sd = V.mesh().geometric_dimension()
    m = V.ufl_domain()
    if model["BCs"]["status"]:
        minz = -model["mesh"]["Lz"] - model["BCs"]["lz"]
        maxz = 0.0
        minx = 0.0 - model["BCs"]["lx"]
        maxx = model["mesh"]["Lx"] + model["BCs"]["lx"]
        miny = 0.0 - model["BCs"]["ly"]
        maxy = model["mesh"]["Ly"] + model["BCs"]["ly"]
    else:
        minz = -model["mesh"]["Lz"]
        maxz = 0.0
        minx = 0.0
        maxx = model["mesh"]["Lx"]
        miny = 0.0
        maxy = model["mesh"]["Ly"]

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
        Z = np.asarray(f.get(field)[()])

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
    if field=="velocity_model":
        c = _check_units(c)
    return c


def read_mesh(model, ens_comm, distribution_parameters=None):
    """Reads in an external mesh and scatters it between cores.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    ens_comm: Firedrake.ensemble_communicator
        An ensemble communicator

    Returns
    -------
    mesh: Firedrake.Mesh object
        The distributed mesh across `ens_comm`.
    V: Firedrake.FunctionSpace object
        The space of the finite elements

    """

    method = model["opts"]["method"]
    degree = model["opts"]["degree"]

    num_sources = len(model["acquisition"]["source_pos"])
    mshname = model["mesh"]["meshfile"]

    if distribution_parameters == None:
        distribution_parameters = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}

    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            mshname,
            comm=ens_comm.comm,
            distribution_parameters=distribution_parameters,
        )
    else:
        mesh = fire.Mesh(mshname, comm=ens_comm.comm)
    if ens_comm.comm.rank == 0 and ens_comm.ensemble_comm.rank == 0:
        print(
            "INFO: Distributing %d shot(s) across %d core(s). Each shot is using %d cores"
            % (
                num_sources,
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
    # Element type
    element = domains.space.FE_method(mesh, method, degree)
    # Space of problem
    return mesh, fire.FunctionSpace(mesh, element)
