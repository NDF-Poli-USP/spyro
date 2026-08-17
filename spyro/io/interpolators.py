"""Methods related to intepolating grids or functions."""

import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _grid_velocity_data_to_source_function(grid_velocity_data, comm=None):
    """Build a CG1 Firedrake function on a structured mesh from grid data."""
    # Adding imports here to avoid circular imports
    from ..meshing.meshing_parameters import MeshingParameters
    from ..meshing.meshing_functions import AutomaticMesh

    vp_values = np.asarray(grid_velocity_data["vp_values"])
    length_z = grid_velocity_data["length_z"]
    length_x = grid_velocity_data["length_x"]
    length_y = grid_velocity_data.get("length_y")
    grid_spacing = grid_velocity_data.get("grid_spacing")
    grid_spacing_z = grid_velocity_data.get("grid_spacing_z", grid_spacing)
    grid_spacing_x = grid_velocity_data.get("grid_spacing_x", grid_spacing)
    grid_spacing_y = grid_velocity_data.get("grid_spacing_y", grid_spacing)
    source_mesh_parameters = {
        "dimension": vp_values.ndim,
        "length_z": length_z,
        "length_x": length_x,
        "length_y": length_y,
        "mesh_type": "firedrake_mesh",
        "edge_length": grid_spacing,
        "edge_length_z": grid_spacing_z,
        "edge_length_x": grid_spacing_x,
        "edge_length_y": grid_spacing_y,
        "abc_pad_length": grid_velocity_data.get("abc_pad_length"),
    }
    source_mesh = AutomaticMesh(
        MeshingParameters(input_mesh_dictionary=source_mesh_parameters, comm=comm)
    ).create_mesh()

    source_space = fire.FunctionSpace(source_mesh, "CG", 1)
    source = fire.Function(source_space)
    source_coords = source_mesh.coordinates.dat.data

    z_nodes = np.unique(source_coords[:, 0])
    x_nodes = np.unique(source_coords[:, 1])
    z_index = np.searchsorted(z_nodes, source_coords[:, 0])
    x_index = np.searchsorted(x_nodes, source_coords[:, 1])

    if vp_values.ndim == 2:
        source.dat.data[:] = vp_values[z_index, x_index]
    else:
        y_nodes = np.unique(source_coords[:, 2])
        y_index = np.searchsorted(y_nodes, source_coords[:, 2])
        source.dat.data[:] = vp_values[z_index, x_index, y_index]

    return source


def project_grid_velocity_data(grid_velocity_data, V, comm=None):
    """Project a structured grid dictionary onto a Firedrake function space."""
    from ..plots.plots import debug_pvd

    source = _grid_velocity_data_to_source_function(grid_velocity_data, comm=comm)
    debug_pvd(source, "check_source.pvd")
    c = fire.Function(V).interpolate(source, allow_missing_dofs=True)
    debug_pvd(c, "check_c.pvd")
    return _check_units(c)


def _hdf5_velocity_model_to_grid_velocity_data(Model, fname):
    """Convert an HDF5 velocity model into a grid velocity dictionary."""
    with h5py.File(fname, "r") as f:
        vp_values = np.asarray(f.get("velocity_model")[()])

    pad_length = Model.mesh_parameters.abc_pad_length
    pad_length = 0.0 if pad_length is None else pad_length

    z_extent = Model.mesh_parameters.length_z + pad_length
    x_extent = Model.mesh_parameters.length_x + 2.0 * pad_length
    spacing_z = z_extent / float(vp_values.shape[0] - 1)
    spacing_x = x_extent / float(vp_values.shape[1] - 1)
    if vp_values.ndim == 2:
        grid_spacing = spacing_z if np.isclose(spacing_z, spacing_x) else None
        length_y = None
    elif vp_values.ndim == 3:
        if Model.mesh_parameters.length_y is None:
            raise ValueError("3D HDF5 velocity model requires length_y.")

        y_extent = Model.mesh_parameters.length_y + 2.0 * pad_length
        spacing_y = y_extent / float(vp_values.shape[2] - 1)
        grid_spacing = (
            spacing_z
            if np.isclose(spacing_z, spacing_x) and np.isclose(spacing_z, spacing_y)
            else None
        )
        length_y = Model.mesh_parameters.length_y
    else:
        raise NotImplementedError("Only 2D and 3D HDF5 velocity models are supported.")

    grid_velocity_data = {
        "vp_values": vp_values,
        "grid_spacing": grid_spacing,
        "grid_spacing_z": spacing_z,
        "grid_spacing_x": spacing_x,
        "length_z": Model.mesh_parameters.length_z,
        "length_x": Model.mesh_parameters.length_x,
        "length_y": length_y,
        "abc_pad_length": pad_length,
    }
    if vp_values.ndim == 3:
        grid_velocity_data["grid_spacing_y"] = spacing_y
    return grid_velocity_data


def interpolate(Model, fname, V, fast_interpolate=False):
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
    if fast_interpolate is True:
        return fast_interpolation(Model, fname, V)
    if isinstance(fname, dict):
        return project_grid_velocity_data(fname, V, comm=Model.comm)
    elif isinstance(fname, str) and fname.endswith((".hdf5", ".h5")):
        grid_velocity_data = _hdf5_velocity_model_to_grid_velocity_data(Model, fname)
        return project_grid_velocity_data(grid_velocity_data, V, comm=Model.comm)
    else:
        raise NotImplementedError


def fast_interpolation(Model, fname, V):
    """Read and interpolate fast a seismic velocity model from HDF5.

    Interpolates a seismic velocity model stored in a HDF5 file onto the
    nodes of a finite element space.

    Parameters
    ----------
    Model : spyro object
        Model options and parameters.
    fname : str
        The name of the HDF5 file containing the seismic velocity model.
    V : firedrake.FunctionSpace
        The finite element space for interpolation.

    Returns
    -------
    c : Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes
        of the finite elements.
    """
    m = V.ufl_domain()

    add_pad = False
    if Model.mesh_parameters.abc_pad_length is not None:
        if Model.mesh_parameters.abc_pad_length > 0.0:
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
