"""IO utilities for spyro wave propagation.

This module provides functions for managing input/output operations related to
wave propagation, including file I/O for shots, receivers, and mesh data.
"""

from __future__ import with_statement

import pickle
import firedrake as fire
import h5py
import numpy as np
from scipy.interpolate import griddata
import os
import warnings
from .parallelism_wrappers import ensemble_save, ensemble_load
from ..tools.version_control import is_firedrake_new
from .segy_io import read_segy_velocity_model

if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate

    fire.interpolate = interpolate


def write_function_to_grid(function, V, grid_spacing, buffer=False):
    """Interpolate a Firedrake function to a structured grid.

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
    (dimension,) = coords.ufl_shape
    if dimension == 2:
        x, y = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif dimension == 3:
        x, y, z = coords.dat.data[:, 0], coords.dat.data[:, 1], coords.dat.data[:, 2]
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
def save_shots(wave, file_name="shots/shot_record_", shot_ids=0):
    """Save the shot record from last forward solve to a pickle file.

    Parameters
    ----------
    wave : :class:`Wave`
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
        pickle.dump(wave.forward_solution_receivers, f)
    return None


@ensemble_load
def load_shots(wave, file_name="shots/shot_record_", shot_ids=0):
    """Load a `pickle` to a `numpy.ndarray`.

    Parameters
    ----------
    wave: :class:`spyro.solvers.Wave` object
        A :class:`spyro.solvers.Wave` object
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
        wave.forward_solution_receivers = array
    return None


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


def parallel_print(string, comm=None):
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
        raise TypeError("axes_order must be a string, tuple, or list.")

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
                integer_axes = [axis for axis in integer_axes if axis != 2]

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
    if all(isinstance(axis, str) and axis.strip() in ("0", "1", "2") for axis in parts):
        integer_axes = [int(axis.strip()) for axis in parts]

        if ndim == 2:
            if len(integer_axes) == 3:
                if sorted(integer_axes) != [0, 1, 2]:
                    raise ValueError(
                        "A 3-entry numeric axes_order must contain "
                        "0, 1, and 2 exactly once."
                    )

                integer_axes = [axis for axis in integer_axes if axis != 2]

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
                        "A 3-entry axis order must contain x, y, and z " "exactly once."
                    )

                # Remove y for a 2D model.
                named_axes = [axis for axis in named_axes if axis != "y"]

            if sorted(named_axes) != ["x", "z"]:
                raise ValueError(
                    "For a 2D model, axes_order must contain " "z and x exactly once."
                )

        else:
            if sorted(named_axes) != ["x", "y", "z"]:
                raise ValueError(
                    "For a 3D model, axes_order must contain "
                    "z, x, and y exactly once."
                )

        return tuple(named_axes)

    raise TypeError("axes_order must contain either only integers or only strings.")


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
            "Please specify nz, nx, and ny. " "Use ny=0 for a 2D binary velocity model."
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
        other_invalid_count = int(np.sum(~np.isfinite(vp_other)))

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

    transpose_order = tuple(raw_axes.index(axis) for axis in final_axes)

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
