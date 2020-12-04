from __future__ import with_statement

import os
import pickle

import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .. import domains

__all__ = ["save_shots", "load_shots", "read_mesh", "interpolate"]


def save_shots(filename, array):
    """Save a `numpy.ndarray` to a `pickle`.

    Parameters
    ----------
    filename: str
        The filename to save the data as a `pickle`
    array: `numpy.ndarray`
        The data to save a pickle (e.g., a shot)

    Returns
    -------
    None

    """
    with open(filename, "wb") as f:
        pickle.dump(array, f)
    return None


def load_shots(filename):
    """Load a `pickle` to a `numpy.ndarray`.

    Parameters
    ----------
    filename: str
        The filename to save the data as a `pickle`

    Returns
    -------
    array: `numpy.ndarray`
        The data

    """

    with open(filename, "rb") as f:
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
    if min(c.dat.data[:]) > 1000.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def interpolate(model, mesh, V, guess=False):
    """Read and interpolate a seismic velocity model stored
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

    Returns
    -------
    c: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the finite elements.

    """
    sd = V.mesh().geometric_dimension()
    m = V.ufl_domain()
    if model["PML"]["status"]:
        minz = -model["mesh"]["Lz"] - model["PML"]["lz"]
        maxz = 0.0
        minx = 0.0 - model["PML"]["lx"]
        maxx = model["mesh"]["Lx"] + model["PML"]["lx"]
        miny = 0.0 - model["PML"]["ly"]
        maxy = model["mesh"]["Ly"] + model["PML"]["ly"]
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

    if guess:
        fname = model["mesh"]["initmodel"]
    else:
        fname = model["mesh"]["truemodel"]

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


def read_mesh(model, ens_comm):
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

    num_sources = model["acquisition"]["num_sources"]
    mshname = model["mesh"]["meshfile"]

    if method == "CG" or method == "KMV":
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
            "INFO: Distributing %d shot(s) across %d processor(s). Each shot is using %d cores"
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
