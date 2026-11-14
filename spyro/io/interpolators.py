"""Methods related to intepolating grids or functions."""

import firedrake as fire
import h5py
import numpy as np
from mpi4py import MPI
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solvers.wave import Wave


def _grid_velocity_data_to_source_function(grid_velocity_data: dict, comm=None):
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


def project_grid_velocity_data(
    grid_velocity_data: dict, V: fire.FunctionSpace, comm=None
):
    """Project a structured grid dictionary onto a Firedrake function space."""
    from ..plots.debug_plots import debug_pvd
    source = _grid_velocity_data_to_source_function(grid_velocity_data, comm=comm)
    debug_pvd(source, "check_source.pvd")
    c = fire.Function(V).interpolate(source, allow_missing_dofs=True)
    debug_pvd(c, "check_c.pvd")
    return _check_units(c)


def _hdf5_velocity_model_to_grid_velocity_data(Model, fname: str | Path):
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


def _nearest_nonzero_sample(
    axes: tuple[np.ndarray, ...], values: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Return the value of the nonzero grid sample nearest to each point.

    Parameters
    ----------
    axes : `tuple` of `numpy.ndarray`
        Increasing sample coordinates along each grid axis.
    values : `numpy.ndarray`
        Grid values, with one axis for each entry of `axes` and at least one
        nonzero value.
    points : `numpy.ndarray`
        Coordinates of the points, of shape `(n_points, len(axes))`, each in
        a grid cell with only zero samples.

    Returns
    -------
    `numpy.ndarray`
        Values at the points, of shape `(n_points,)`.

    Notes
    -----
    Only nonzero samples next to a zero sample along an axis are searched.
    The nearest nonzero sample to a point in a cell of zero samples is one of
    them: from any other nonzero sample, the step along an axis towards the
    cell reaches a nearer nonzero sample.
    """
    nonzero = values != 0.0
    next_to_zero = np.zeros_like(nonzero)
    for axis in range(values.ndim):
        along = np.moveaxis(nonzero, axis, 0)
        result = np.moveaxis(next_to_zero, axis, 0)
        result[1:] |= along[1:] & ~along[:-1]
        result[:-1] |= along[:-1] & ~along[1:]
    index = np.nonzero(next_to_zero)
    samples = np.column_stack([axis[i] for axis, i in zip(axes, index)])
    _, nearest = KDTree(samples).query(points)
    return values[index][nearest]


def _sample_grid(
    axes: tuple[np.ndarray, ...],
    values: np.ndarray,
    points: np.ndarray,
    sharp_zero_interface: bool,
    water_points: np.ndarray | None,
) -> np.ndarray:
    """Sample a regular grid at points, optionally without blending zeros.

    Linear interpolation gives a point in a grid cell with both zero and
    nonzero samples, such as a water and rock cell of a Vs model, a value
    between the two that no material has. With `sharp_zero_interface`, a
    grid with zeros is instead sampled as follows.

    On a mesh that follows the interface between zero and nonzero values,
    `water_points` marks the points of the zero side, such as the water:

    - points of the zero side get zero.
    - other points in a cell with both zero and nonzero samples get the
      linear interpolation of the nonzero samples only, with their weights
      rescaled to add up to one.
    - other points in a cell with only zero samples, where the mesh puts the
      interface above the samples, get the nearest nonzero sample. So do
      points in a pocket of zeros inside the nonzero region.

    On a mesh that does not follow it, points in a cell with both zero and
    nonzero samples get the nearest sample, zero or not.

    All other points keep the linear interpolation, and a grid without zeros
    is sampled as without the option.

    Parameters
    ----------
    axes : `tuple` of `numpy.ndarray`
        Increasing sample coordinates along each grid axis.
    values : `numpy.ndarray`
        Grid values, with one axis for each entry of `axes`.
    points : `numpy.ndarray`
        Coordinates of the points, of shape `(n_points, len(axes))`. They
        must lie inside the grid.
    sharp_zero_interface : `bool`
        If `False`, return the linear interpolation.
    water_points : `numpy.ndarray` or `None`
        Boolean mask of the points on the zero side of a mesh that follows
        the interface, or `None` if the mesh does not follow it.

    Returns
    -------
    `numpy.ndarray`
        Values at the points, of shape `(n_points,)`.

    Notes
    -----
    A point whose interpolation weights on the zero samples, or on the
    nonzero samples, add up to less than `1e-10` is taken to have none. A
    point on a row of samples can otherwise fall a rounding error inside the
    neighbouring cell.
    """
    linear = RegularGridInterpolator(axes, values)(points)
    nonzero = values != 0.0
    if not sharp_zero_interface or nonzero.all() or not nonzero.any():
        return linear

    # The weights of the zero samples are 1 - nonzero_weight
    nonzero_weight = RegularGridInterpolator(axes, nonzero.astype(float))(points)
    # A point on a sample row can fall a rounding error inside the next cell
    tolerance = 1e-10
    has_nonzero = nonzero_weight > tolerance
    mixed = has_nonzero & (nonzero_weight < 1.0 - tolerance)
    if water_points is None:
        linear[mixed] = RegularGridInterpolator(axes, values, method="nearest")(
            points[mixed]
        )
        return linear

    linear[mixed] /= nonzero_weight[mixed]
    linear[water_points] = 0.0
    only_zeros = ~water_points & ~has_nonzero
    if only_zeros.any():
        linear[only_zeros] = _nearest_nonzero_sample(axes, values, points[only_zeros])
    return linear


def interpolate(
    Model: "Wave",
    fname: str | Path | dict,
    V: fire.FunctionSpace,
    fast_interpolate: bool = False,
    sharp_zero_interface: bool = False,
) -> fire.Function:
    """Read and interpolate a seismic velocity model onto a Firedrake space.

    Parameters
    ----------
    Model : `Wave`
        Model options and parameters.
    fname : `str`, `pathlib.Path` or `dict`
        The name of the HDF5 file containing the seismic velocity model, or
        a grid dictionary with keys such as `vp_values`, `length_z` and
        `length_x`.
    V : `firedrake.FunctionSpace`
        The space of the finite elements.
    fast_interpolate : `bool`, optional
        If `True`, sample the HDF5 grid at the nodes of `V` with
        :func:`fast_interpolation`. Otherwise, interpolate from a CG1
        function on a mesh built at the grid resolution. Default is `False`.
    sharp_zero_interface : `bool`, optional
        If `True`, do not blend zero and nonzero grid samples, as described
        in :func:`fast_interpolation`. Requires `fast_interpolate`. Default
        is `False`.

    Returns
    -------
    `firedrake.Function`
        Seismic velocity interpolated onto the nodes of the finite elements.

    Raises
    ------
    ValueError
        If `sharp_zero_interface` is `True` and `fast_interpolate` is not,
        or as raised by :func:`fast_interpolation`.
    NotImplementedError
        If `fname` is neither a grid dictionary nor an HDF5 file name.
    """
    if fast_interpolate is True:
        return fast_interpolation(
            Model, fname, V, sharp_zero_interface=sharp_zero_interface
        )
    if sharp_zero_interface:
        raise ValueError("sharp_zero_interface requires fast_interpolate=True.")
    if isinstance(fname, dict):
        return project_grid_velocity_data(fname, V, comm=Model.comm)
    elif isinstance(fname, str) and fname.endswith((".hdf5", ".h5")):
        grid_velocity_data = _hdf5_velocity_model_to_grid_velocity_data(Model, fname)
        return project_grid_velocity_data(grid_velocity_data, V, comm=Model.comm)
    else:
        raise NotImplementedError


def fast_interpolation(
    Model: "Wave",
    fname: str | Path,
    V: fire.FunctionSpace,
    sharp_zero_interface: bool = False,
) -> fire.Function:
    """Read and interpolate fast a seismic velocity model from HDF5.

    Interpolates a seismic velocity model stored in a HDF5 file onto the
    nodes of a finite element space.

    Parameters
    ----------
    Model : `Wave`
        Model options and parameters.
    fname : `str` or `pathlib.Path`
        The name of the HDF5 file containing the seismic velocity model.
    V : `firedrake.FunctionSpace`
        The finite element space for interpolation.
    sharp_zero_interface : `bool`, optional
        If `True`, a node in a grid cell with both zero and nonzero samples,
        such as a water and rock cell of a Vs model, does not get a value
        between the two. If the mesh is marked as following the interface
        (`water_interface` in the mesh parameters), nodes of the water cells
        get zero and the other nodes get a rock value, see
        :func:`_sample_grid`. Otherwise, the node gets the value of the
        nearest sample. Grids without zeros are unchanged. Default is
        `False`.

    Returns
    -------
    `firedrake.Function`
        Seismic velocity interpolated onto the nodes of the finite elements.

    Raises
    ------
    ValueError
        If a grid with zeros is read with `sharp_zero_interface` on a mesh
        marked with `water_interface` whose water cells are not tagged, see
        :func:`spyro.meshing.meshing_utils.water_nodes`.

    Notes
    -----
    `sharp_zero_interface` sets the values at the nodes. Between them, the
    field still varies as the finite element functions do, so in a
    continuous space Vs goes from the rock value to zero across the water
    elements that touch the seabed.
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

            axes = (z, x)
            points = np.column_stack((qp_z2, qp_x2))
        elif sd == 3:
            nrow, ncol, ncol2 = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)
            y = np.linspace(miny, maxy, ncol2)

            # make sure no out-of-bounds
            qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
            qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]
            qp_y2 = [miny if y < miny else maxy if y > maxy else y for y in qp_y]

            axes = (z, x, y)
            points = np.column_stack((qp_z2, qp_x2, qp_y2))

    water_points = None
    if sharp_zero_interface and Model.mesh_parameters.water_interface:
        # Imported here to avoid circular imports
        from ..meshing.meshing_utils import water_nodes

        # Only grids with zeros need the water nodes, so Vp does not
        water_points = (
            water_nodes(V) if np.any(Z == 0.0) else np.zeros(len(points), bool)
        )
    tmp = _sample_grid(axes, Z, points, sharp_zero_interface, water_points)

    c = fire.Function(V)
    c.dat.data[:] = tmp
    c = _check_units(c)
    return c


def _check_units(c: fire.Function) -> fire.Function:
    """Verify and convert velocity units from m/s to km/s if needed.

    Parameters
    ----------
    c : `firedrake.Function`
        Velocity field to check.

    Returns
    -------
    `firedrake.Function`
        Velocity field with units in km/s.

    Notes
    -----
    The field is taken to be in m/s if its smallest nonzero value, over all
    processes, is above 100. Zeros are left out so that a Vs field with a
    water layer is converted like the Vp field of the same model.
    """
    values = c.dat.data_ro
    nonzero = values[values != 0.0]
    local_minimum = nonzero.min() if nonzero.size else np.inf
    minimum = c.comm.allreduce(local_minimum, op=MPI.MIN)
    if np.isfinite(minimum) and minimum > 100.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c
