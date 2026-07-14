from __future__ import with_statement

import pickle
import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import griddata
import os
import warnings
import segyio
from .parallelism_wrappers import ensemble_save, ensemble_load
from ..tools.version_control import is_firedrake_new


if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate
    fire.interpolate = interpolate


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
        grid_spacing = spacing_z if np.isclose(spacing_z, spacing_x) else None
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
    nz : int
        Number of samples per trace, corresponding to the z direction.
    nx : int
        Number of traces in the SEG-Y file, corresponding to the x direction.

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

    return vp, nz, nx


def _parse_axes_order(axes_order, ndim=3):
    """Convert an axis-order specification to axis names.

    Parameters
    ----------
    axes_order : str, tuple, or list
        Axis order in the raw binary file.

        For 2D models, accepted examples include:

        - ``"z x"``
        - ``"zx"``
        - ``"x z"``
        - ``(0, 1)``
        - ``(1, 0)``

        Three-dimensional specifications are also accepted for 2D models.
        In that case, the y axis is removed:

        - ``"z x y"`` becomes ``("z", "x")``
        - ``(2, 0, 1)`` becomes ``("z", "x")``

    ndim : {2, 3}, optional
        Number of dimensions in the velocity model.

    Returns
    -------
    tuple of str
        Axis names in the order found in the raw binary file.

    Raises
    ------
    TypeError
        If ``axes_order`` is not a string, tuple, or list, or if it mixes
        integer and string entries.
    ValueError
        If the axis specification is invalid.
    """
    if ndim not in (2, 3):
        raise ValueError("ndim must be either 2 or 3.")

    axis_from_int = {
        0: "z",
        1: "x",
        2: "y",
    }

    if isinstance(axes_order, str):
        clean = axes_order.lower().replace(",", " ").strip()
        parts = clean.split()

        # Compact forms such as "zx", "xz", "zxy", "201".
        if len(parts) == 1 and len(parts[0]) in (2, 3):
            parts = list(parts[0])

    elif isinstance(axes_order, (tuple, list)):
        parts = list(axes_order)

    else:
        raise TypeError(
            "axes_order must be a string, tuple, or list."
        )

    # Integer specification: (0, 1), (1, 0), (2, 0, 1), etc.
    if all(isinstance(axis, (int, np.integer)) for axis in parts):
        integer_axes = [int(axis) for axis in parts]

        if ndim == 2:
            if len(integer_axes) == 3:
                if sorted(integer_axes) != [0, 1, 2]:
                    raise ValueError(
                        "A 3-entry numeric axes_order must contain "
                        "0, 1, and 2 exactly once."
                    )

                # Remove y for a 2D model.
                integer_axes = [
                    axis for axis in integer_axes if axis != 2
                ]

            if sorted(integer_axes) != [0, 1]:
                raise ValueError(
                    "For a 2D model, numeric axes_order must contain "
                    "0 and 1 exactly once."
                )

        else:
            if sorted(integer_axes) != [0, 1, 2]:
                raise ValueError(
                    "For a 3D model, numeric axes_order must contain "
                    "0, 1, and 2 exactly once."
                )

        return tuple(axis_from_int[axis] for axis in integer_axes)

    # String specification containing numeric characters:
    # "0 1", "10", "201", etc.
    if all(
        isinstance(axis, str) and axis.strip() in ("0", "1", "2")
        for axis in parts
    ):
        integer_axes = [int(axis.strip()) for axis in parts]

        if ndim == 2:
            if len(integer_axes) == 3:
                if sorted(integer_axes) != [0, 1, 2]:
                    raise ValueError(
                        "A 3-entry numeric axes_order must contain "
                        "0, 1, and 2 exactly once."
                    )

                integer_axes = [
                    axis for axis in integer_axes if axis != 2
                ]

            if sorted(integer_axes) != [0, 1]:
                raise ValueError(
                    "For a 2D model, numeric axes_order must contain "
                    "0 and 1 exactly once."
                )

        else:
            if sorted(integer_axes) != [0, 1, 2]:
                raise ValueError(
                    "For a 3D model, numeric axes_order must contain "
                    "0, 1, and 2 exactly once."
                )

        return tuple(axis_from_int[axis] for axis in integer_axes)

    # Axis-name specification.
    if all(isinstance(axis, str) for axis in parts):
        named_axes = [axis.lower().strip() for axis in parts]

        if ndim == 2:
            if len(named_axes) == 3:
                if sorted(named_axes) != ["x", "y", "z"]:
                    raise ValueError(
                        "A 3-entry axis order must contain x, y, and z "
                        "exactly once."
                    )

                # Remove y for a 2D model.
                named_axes = [
                    axis for axis in named_axes if axis != "y"
                ]

            if sorted(named_axes) != ["x", "z"]:
                raise ValueError(
                    "For a 2D model, axes_order must contain "
                    "z and x exactly once."
                )

        else:
            if sorted(named_axes) != ["x", "y", "z"]:
                raise ValueError(
                    "For a 3D model, axes_order must contain "
                    "z, x, and y exactly once."
                )

        return tuple(named_axes)

    raise TypeError(
        "axes_order must contain either only integers or only strings."
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
    """Read a 2D or 3D velocity model from a binary file.

    A two-dimensional model is selected by setting ``ny=0``. The returned
    velocity array then has shape ``(nz, nx)``.

    A three-dimensional model uses ``ny>0`` and returns an array with shape
    ``(nz, nx, ny)``.

    Parameters
    ----------
    filename : str
        Filename of the raw binary velocity model.
    nz : int
        Number of grid points in the z direction.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction. Set to zero for a 2D model.
    byte_order : {'little', 'big'}, optional
        Byte order of the binary file. If the selected byte order produces
        NaN or Inf values, the opposite byte order is tested and used when
        it produces fewer invalid values.
    axes_order : str, tuple, or list, optional
        Axis order in the raw binary file.
        For 2D models, examples include ``"z x"``, ``"x z"``, ``(0, 1)``,
        and ``(1, 0)``. Three-dimensional specifications such as
        ``"z x y"`` are also accepted; the y axis is ignored when ``ny=0``.
    axes_order_sort : {'C', 'F'}, optional
        Memory layout used to reshape the raw binary values.
    dtype : str or numpy.dtype, optional
        Floating-point dtype. If its size does not match the file size,
        ``float32`` and ``float64`` are tested.

    Returns
    -------
    vp : numpy.ndarray
        Velocity model in canonical ``(z, x)`` or ``(z, x, y)`` order.
    nz : int
        Number of grid points in z.
    nx : int
        Number of grid points in x.
    ny : int
        Zero for a 2D model, otherwise the number of grid points in y.

    Raises
    ------
    ValueError
        If dimensions or input options are invalid.
    """
    if nz is None or nx is None or ny is None:
        raise ValueError(
            "Please specify nz, nx, and ny. "
            "Use ny=0 for a 2D binary velocity model."
        )

    nz = int(nz)
    nx = int(nx)
    ny = int(ny)

    if nz <= 0 or nx <= 0:
        raise ValueError("nz and nx must be greater than zero.")

    if ny < 0:
        raise ValueError("ny must be zero for 2D or greater than zero for 3D.")

    is_2d = ny == 0

    byte_order = str(byte_order).lower()
    if byte_order not in ("little", "big"):
        raise ValueError("byte_order must be 'little' or 'big'.")

    axes_order_sort = str(axes_order_sort).upper()
    if axes_order_sort not in ("C", "F"):
        raise ValueError("axes_order_sort must be 'C' or 'F'.")

    if is_2d:
        expected_elements = nz * nx
    else:
        expected_elements = nz * nx * ny

    actual_bytes = os.path.getsize(filename)

    dtype = np.dtype(dtype)
    expected_bytes = expected_elements * dtype.itemsize

    # Correct a wrong dtype using the expected file size.
    if actual_bytes != expected_bytes:
        matched_dtype = None

        for candidate in (
            np.dtype("float32"),
            np.dtype("float64"),
        ):
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
                "Supported dtypes are float32 and float64."
            )

        warnings.warn(
            f"Selected dtype={dtype} does not match the file size. "
            f"Using dtype={matched_dtype} instead."
        )
        dtype = matched_dtype

    if byte_order == "little":
        selected_dtype = dtype.newbyteorder("<")
        other_byte_order = "big"
        other_dtype = dtype.newbyteorder(">")
    else:
        selected_dtype = dtype.newbyteorder(">")
        other_byte_order = "little"
        other_dtype = dtype.newbyteorder("<")

    print(f"Reading binary file: {filename}")
    print(f"Selected byte_order: {byte_order}")
    print(f"Selected/resolved dtype: {dtype}")
    print(f"Model dimension: {'2D' if is_2d else '3D'}")

    vp = np.fromfile(filename, dtype=selected_dtype)

    if vp.size != expected_elements:
        raise ValueError(
            f"Unexpected number of values read from {filename}.\n"
            f"Expected {expected_elements}, got {vp.size}."
        )

    # Try the opposite byte order only when the selected byte order
    # produces NaN or Inf values.
    invalid_count = int(np.sum(~np.isfinite(vp)))

    if invalid_count > 0:
        vp_other = np.fromfile(filename, dtype=other_dtype)
        other_invalid_count = int(
            np.sum(~np.isfinite(vp_other))
        )

        if other_invalid_count < invalid_count:
            warnings.warn(
                f"Selected byte_order='{byte_order}' produced "
                f"{invalid_count} NaN/Inf values. "
                f"Using byte_order='{other_byte_order}' instead, "
                f"which produced {other_invalid_count} NaN/Inf values."
            )
            byte_order = other_byte_order
            vp = vp_other

        else:
            warnings.warn(
                f"Selected byte_order='{byte_order}' produced "
                f"{invalid_count} NaN/Inf values, but "
                f"byte_order='{other_byte_order}' produced "
                f"{other_invalid_count}. "
                f"Keeping byte_order='{byte_order}'."
            )

    ndim = 2 if is_2d else 3
    raw_axes = _parse_axes_order(axes_order, ndim=ndim)

    if is_2d:
        sizes = {
            "z": nz,
            "x": nx,
        }
        final_axes = ("z", "x")

    else:
        sizes = {
            "z": nz,
            "x": nx,
            "y": ny,
        }
        final_axes = ("z", "x", "y")

    raw_shape = tuple(sizes[axis] for axis in raw_axes)

    vp = vp.reshape(raw_shape, order=axes_order_sort)

    transpose_order = tuple(
        raw_axes.index(axis) for axis in final_axes
    )

    vp = vp.transpose(transpose_order)
    vp = np.flipud(vp)

    return vp, nz, nx, ny


def write_velocity_model(
    filename: str,
    ofname: str = None,
    model_type: str = "bin",
    nz: int = None,
    nx: int = None,
    ny: int = None,
    byte_order: str = "little",
    axes_order="z x y",
    axes_order_sort: str = "C",
    dtype: str = "float32",
):
    """Read and write a velocity model as an HDF5 file.

    Binary models may be either 2D or 3D. Set ``ny=0`` for a 2D model.

    Parameters
    ----------
    filename : str
        Input binary or SEG-Y filename.
    ofname : str, optional
        Output HDF5 filename.
    model_type : {'bin', 'segy'}, optional
        Input file type.
    nz : int, optional
        Number of grid points in z for a binary model.
    nx : int, optional
        Number of grid points in x for a binary model.
    ny : int, optional
        Number of grid points in y. Set to zero for a 2D binary model.
    byte_order : {'little', 'big'}, optional
        Binary file byte order.
    axes_order : str, tuple, or list, optional
        Axis order in the binary file.
    axes_order_sort : {'C', 'F'}, optional
        Binary file memory ordering.
    dtype : str or numpy.dtype, optional
        Binary data type.

    Returns
    -------
    str
        Path to the generated HDF5 file.
    """
    model_type = model_type.lower()

    if ofname is None:
        warnings.warn(
            "No output filename specified, name will be `filename`"
        )
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
        vp, nz, nx = read_segy_velocity_model(filename)
        ny = 0

    else:
        raise ValueError(
            "model_type must be either 'bin' or 'segy'. "
            f"Got model_type={model_type!r}."
        )

    if not str(ofname).endswith(".hdf5"):
        ofname = str(ofname) + ".hdf5"

    print(f"Writing velocity model: {ofname}", flush=True)

    with h5py.File(ofname, "w") as h5:
        h5.create_dataset(
            "velocity_model",
            data=vp,
            dtype="f",
        )
        h5.attrs["shape"] = vp.shape
        h5.attrs["units"] = "m/s"

    return ofname
