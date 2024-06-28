import firedrake as fire
import os
from SeismicMesh import write_velocity_model
from scipy.interpolate import RegularGridInterpolator
import h5py
import numpy as np


def _check_units(c):
    if min(c.dat.data[:]) > 100.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def interpolate_to_pvd(hdf5_filename, V, pvd_filename):
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

    file_name = hdf5_filename
    dimension = V.mesh().geometric_dimension()
    mesh = V.ufl_domain()

    # Getting dimensions
    z_data = mesh.coordinates.dat.data[:, 0]
    x_data = mesh.coordinates.dat.data[:, 1]
    length_z = np.abs(np.min(z_data))
    length_x = np.max(x_data)

    minz = -length_z
    maxz = 0.0
    minx = 0.0
    maxx = length_x
    miny = 0.0
    maxy = 0.0

    W = fire.VectorFunctionSpace(mesh, V.ufl_element())
    coords = fire.interpolate(mesh.coordinates, W)
    # (z,x) or (z,x,y)
    if dimension == 2:
        qp_z, qp_x = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif dimension == 3:
        qp_z, qp_x, qp_y = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise NotImplementedError

    with h5py.File(file_name, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])

        if dimension == 2:
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
        elif dimension == 3:
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
    fire.File(pvd_filename).write(c)
    return c


def project_to_pvd(hdf5_filename, V, pvd_filename):
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

    file_name = hdf5_filename
    dimension = V.mesh().geometric_dimension()
    mesh = V.ufl_domain()

    # Getting dimensions
    z_data = mesh.coordinates.dat.data[:, 0]
    x_data = mesh.coordinates.dat.data[:, 1]
    length_z = np.abs(np.min(z_data))
    length_x = np.max(x_data)

    minz = -length_z
    maxz = 0.0
    minx = 0.0
    maxx = length_x
    miny = 0.0
    maxy = 0.0

    W = fire.VectorFunctionSpace(mesh, V.ufl_element())
    coords = fire.interpolate(mesh.coordinates, W)
    # (z,x) or (z,x,y)
    if dimension == 2:
        qp_z, qp_x = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif dimension == 3:
        qp_z, qp_x, qp_y = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise NotImplementedError

    with h5py.File(file_name, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])

        if dimension == 2:
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
        elif dimension == 3:
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
    fire.File(pvd_filename).write(c)
    return c
