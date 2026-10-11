"""Model dictionaries used by the cells-per-wavelength calculator."""

import math
import warnings

import numpy as np

import spyro
from .heterogeneous_case import HeterogeneousCase

#: Mesh type used for each supported finite element method.
MESH_TYPES = {
    "mass_lumped_triangle": "SeismicMesh",
    "spectral_quadrilateral": "firedrake_mesh",
}

#: Velocities above this value are assumed to be in m/s instead of km/s.
METERS_PER_SECOND_THRESHOLD = 500.0

# Acquisition of the homogeneous acoustic model, in wavelengths. The
# receivers form a square grid centred below the source.
#: Vertical distance from the source to the centre of the receiver grid.
HOMOGENEOUS_RECEIVER_GRID_DEPTH = 10.0
#: Side of the square receiver grid.
HOMOGENEOUS_RECEIVER_GRID_WIDTH = 5.0
#: Receivers per side of the grid, and in reduced (testing) mode.
HOMOGENEOUS_RECEIVERS_PER_SIDE = 6
HOMOGENEOUS_RECEIVERS_PER_SIDE_REDUCED = 2

#: Time step of the homogeneous acoustic analytical reference, in s.
HOMOGENEOUS_REFERENCE_DT = 5e-4

# Acquisition of the heterogeneous acoustic model, in km and s.
HETEROGENEOUS_SOURCE_LOCATION = (-0.3, 3.0)
#: Horizontal offsets of the first and last receivers from the source.
HETEROGENEOUS_RECEIVER_OFFSETS = (2.0, 10.0)
HETEROGENEOUS_RECEIVER_QUANTITY = 500
HETEROGENEOUS_FINAL_TIME = 7.5
HETEROGENEOUS_DT = 1e-4

# Acquisition of the heterogeneous isotropic elastic model, in km and s.
ELASTIC_HETEROGENEOUS_SOURCE_LOCATION = (-0.5, 5.0)
#: Depth and horizontal span of the receiver line.
ELASTIC_HETEROGENEOUS_RECEIVER_DEPTH = -4.0
ELASTIC_HETEROGENEOUS_RECEIVER_X_RANGE = (3.0, 6.9)
ELASTIC_HETEROGENEOUS_RECEIVER_QUANTITY = 40
ELASTIC_HETEROGENEOUS_FINAL_TIME = 2.0
ELASTIC_HETEROGENEOUS_DT = 1e-4


def set_mesh_type(method: str) -> str:
    """Return the mesh type used for a finite element method.

    Parameters
    ----------
    method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.

    Returns
    -------
    str
        ``"SeismicMesh"`` for triangles, ``"firedrake_mesh"`` for
        quadrilaterals.

    Raises
    ------
    ValueError
        If ``method`` is not supported.
    """
    if method not in MESH_TYPES:
        raise ValueError("Method is not mass_lumped_triangle or spectral_quadrilateral")
    return MESH_TYPES[method]


def _warn_if_meters_per_second(velocity: float) -> None:
    """Warn if a velocity looks like it is in m/s instead of km/s.

    Parameters
    ----------
    velocity : float
        Velocity to check.
    """
    if velocity > METERS_PER_SECOND_THRESHOLD:
        warnings.warn("Velocity in meters per second")


def _or_default(value: object, default: object) -> object:
    """Return a case override, or the default when it is not given.

    Parameters
    ----------
    value : object
        Case override, None when not given.
    default : object
        Value used when ``value`` is None.

    Returns
    -------
    object
        ``value`` unless it is None, else ``default``.
    """
    return default if value is None else value


def _round_up_to_time_step(time: float, dt: float) -> float:
    """Round a time up to a whole number of time steps.

    A time series of ``round(time / dt) + 1`` samples then ends exactly at
    the returned time, which :func:`~spyro.io.time_io.interpolate_time_series`
    assumes when it resamples the reference.

    Parameters
    ----------
    time : float
        Time to round.
    dt : float
        Time step.

    Returns
    -------
    float
        Smallest multiple of ``dt`` not below ``time``.
    """
    # The small tolerance keeps times already on the grid in place
    return math.ceil(time / dt - 1e-9) * dt


def _build_acoustic_model_dictionary(
    *,
    method: str,
    degree: int,
    length_z: float,
    length_x: float,
    source_locations: list[tuple[float, float]],
    receiver_locations: list[tuple[float, float]],
    frequency: float,
    final_time: float,
    dt: float,
    pml_pad_length: float | None = None,
    periodic: bool = False,
    cells_per_wavelength: float | None = None,
    velocity_model_file_name: str | None = None,
) -> dict:
    """Assemble a 2D acoustic model dictionary for the cpw calculator.

    Parameters
    ----------
    method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    degree : int
        Spatial polynomial degree.
    length_z : float
        Domain length in the z direction, without absorbing layer.
    length_x : float
        Domain length in the x direction, without absorbing layer.
    source_locations : list of tuple of float
        Source ``(z, x)`` locations.
    receiver_locations : list of tuple of float
        Receiver ``(z, x)`` locations.
    frequency : float
        Ricker source peak frequency.
    final_time : float
        Final time of the simulation.
    dt : float
        Time step of the simulation.
    pml_pad_length : float, optional
        Thickness of a PML absorbing layer. Default is None, meaning no
        absorbing layer.
    periodic : bool, optional
        If True, use a periodic ``"firedrake_mesh"`` for either method.
        Default is False, with the mesh type given by :func:`set_mesh_type`.
    cells_per_wavelength : float, optional
        Cells-per-wavelength of the mesh. Default is None, meaning the mesh
        is sized when it is built.
    velocity_model_file_name : str, optional
        Velocity model file. Default is None, for a velocity set later.

    Returns
    -------
    dict
        Spyro model dictionary for :class:`spyro.AcousticWave`.

    Raises
    ------
    ValueError
        If ``method`` is not supported, or a periodic mesh is given a PML.
    """
    mesh_type = set_mesh_type(method)
    if periodic:
        if pml_pad_length is not None:
            raise ValueError("A periodic mesh cannot have a PML.")
        # Only Firedrake meshes can be periodic
        mesh_type = "firedrake_mesh"

    if pml_pad_length is None:
        absorbing_boundary_conditions = {"status": False}
    else:
        absorbing_boundary_conditions = {
            "status": True,
            "abc_type": "PML",
            "exponent": 2,
            "cmax": 4.5,
            "R": 1e-6,
            "pad_length": pml_pad_length,
        }

    return {
        "options": {
            "method": method,
            "degree": degree,
            "dimension": 2,
            "automatic_adjoint": False,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": length_z,
            "length_x": length_x,
            "length_y": 0.0,
            "cells_per_wavelength": cells_per_wavelength,
            "mesh_type": mesh_type,
            "periodic": periodic,
        },
        "absorving_boundary_conditions": absorbing_boundary_conditions,
        "acquisition": {
            "source_type": "ricker",
            "source_locations": source_locations,
            "frequency": frequency,
            "receiver_locations": receiver_locations,
            "use_vertex_only_mesh": True,
        },
        "synthetic_data": {
            "real_velocity_file": velocity_model_file_name,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": final_time,
            "dt": dt,
            "amplitude": 1,
            "output_frequency": 1000,
            "gradient_sampling_frequency": 100,
        },
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": None,
        },
    }


def create_acoustic_model_for_meshing_parameter_2D_homogeneous(
    method: str,
    degree: int,
    frequency: float,
    velocity: float,
    dt: float = HOMOGENEOUS_REFERENCE_DT,
    reduced: bool = False,
) -> dict:
    """Create a 2D homogeneous acoustic model for the cpw calculator.

    The receivers form a square grid of side
    :data:`HOMOGENEOUS_RECEIVER_GRID_WIDTH` wavelengths, centred
    :data:`HOMOGENEOUS_RECEIVER_GRID_DEPTH` wavelengths below the source.
    The final time lets the wave reach the farthest receiver, rounded up
    to a whole number of time steps. The domain is
    a periodic box without absorbing layer, sized so that no wave from a
    periodic image of the source reaches a receiver before the final time.
    The receiver data then matches the free-space analytical solution.

    Parameters
    ----------
    method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    degree : int
        Spatial polynomial degree.
    frequency : float
        Ricker source peak frequency.
    velocity : float
        Constant acoustic velocity, which sets the wavelength.
    dt : float, optional
        Time step used by the initial object and the analytical reference.
        Default is :data:`HOMOGENEOUS_REFERENCE_DT`.
    reduced : bool, optional
        If True, use a 2 by 2 receiver grid instead of 6 by 6, for testing.
        Default is False.

    Returns
    -------
    dict
        Spyro model dictionary for :class:`spyro.AcousticWave`.
    """
    _warn_if_meters_per_second(velocity)
    wavelength = velocity / frequency

    grid_depth = HOMOGENEOUS_RECEIVER_GRID_DEPTH * wavelength
    half_width = HOMOGENEOUS_RECEIVER_GRID_WIDTH * wavelength / 2.0
    deepest_offset = grid_depth + half_width
    farthest_offset = np.hypot(deepest_offset, half_width)

    # The wavelet delay is about 1.2 periods, plus two periods of wavelet tail
    final_time = _round_up_to_time_step(
        farthest_offset / velocity + 3.0 / frequency, dt
    )

    # A periodic image is one box length away along an axis, so it is at
    # least (length - source-receiver span) away from every receiver. Keep
    # its arrival one wavelength past the final time.
    image_distance = velocity * final_time + wavelength
    length_z = image_distance + deepest_offset
    length_x = image_distance + half_width

    # Centre the acquisition in the box
    source_z = -(length_z - deepest_offset) / 2.0
    source_x = length_x / 2.0
    receivers_per_side = (
        HOMOGENEOUS_RECEIVERS_PER_SIDE_REDUCED
        if reduced
        else HOMOGENEOUS_RECEIVERS_PER_SIDE
    )
    receiver_locations = spyro.create_2d_grid(
        source_z - grid_depth + half_width,
        source_z - deepest_offset,
        source_x - half_width,
        source_x + half_width,
        receivers_per_side,
    )

    return _build_acoustic_model_dictionary(
        method=method,
        degree=degree,
        length_z=length_z,
        length_x=length_x,
        source_locations=[(source_z, source_x)],
        receiver_locations=receiver_locations,
        frequency=frequency,
        final_time=final_time,
        dt=dt,
        periodic=True,
    )


def create_acoustic_model_for_meshing_parameter_2D_heterogeneous(
    method: str,
    degree: int,
    frequency: float,
    minimum_velocity: float,
    case: HeterogeneousCase,
    cells_per_wavelength: float,
) -> dict:
    """Create a 2D heterogeneous acoustic model for the cpw calculator.

    By default, a source at :data:`HETEROGENEOUS_SOURCE_LOCATION` records on
    a line of receivers at the source depth; the case can replace the
    acquisition and time axis. The domain is surrounded by a PML one
    wavelength thick. The reference is numerical, so the same layer is
    present in the reference and in every evaluated solve. The P-wave
    velocity file of the case is written, if needed, before returning.

    Parameters
    ----------
    method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    degree : int
        Spatial polynomial degree.
    frequency : float
        Ricker source peak frequency.
    minimum_velocity : float
        Minimum velocity in the domain, which sets the PML thickness.
    case : HeterogeneousCase
        Domain, material and acquisition overrides.
    cells_per_wavelength : float
        Cells-per-wavelength of the initial mesh.

    Returns
    -------
    dict
        Spyro model dictionary for :class:`spyro.AcousticWave`.
    """
    receiver_locations = case.receiver_locations
    if receiver_locations is None:
        source_z, source_x = HETEROGENEOUS_SOURCE_LOCATION
        first_offset, last_offset = HETEROGENEOUS_RECEIVER_OFFSETS
        receiver_locations = spyro.create_transect(
            (source_z, source_x + first_offset),
            (source_z, source_x + last_offset),
            HETEROGENEOUS_RECEIVER_QUANTITY,
        )
    material_files = case.write_material_files(("p_wave_velocity",))

    return _build_acoustic_model_dictionary(
        method=method,
        degree=degree,
        length_z=case.length_z,
        length_x=case.length_x,
        source_locations=_or_default(
            case.source_locations, [HETEROGENEOUS_SOURCE_LOCATION]
        ),
        receiver_locations=receiver_locations,
        frequency=frequency,
        final_time=_or_default(case.final_time, HETEROGENEOUS_FINAL_TIME),
        dt=_or_default(case.dt, HETEROGENEOUS_DT),
        pml_pad_length=minimum_velocity / frequency,
        cells_per_wavelength=cells_per_wavelength,
        velocity_model_file_name=material_files["p_wave_velocity"],
    )


def create_elastic_model_for_meshing_parameter_2D_homogeneous(
    method: str,
    degree: int,
    frequency: float,
    p_wave_velocity: float,
    s_wave_velocity: float,
    density: float,
    dt: float,
    reduced: bool = False,
) -> dict:
    """Create a 2D homogeneous isotropic elastic model for the cpw calculator.

    A horizontal (x-direction) point force is placed in a periodic box. The receivers lie on
    a horizontal line ``7`` S-wavelengths below the source, spanning
    horizontal offsets from ``0`` to ``7`` S-wavelengths. The final time lets
    the S-wave reach the farthest receiver, rounded up to a whole number of
    time steps, and the box is sized so that no
    P-wave from a periodic image of the source reaches any receiver before
    the final time.

    Parameters
    ----------
    method : str
        Finite element method, either ``"mass_lumped_triangle"`` or
        ``"spectral_quadrilateral"``.
    degree : int
        Spatial polynomial degree.
    frequency : float
        Ricker source peak frequency.
    p_wave_velocity : float
        P-wave velocity.
    s_wave_velocity : float
        S-wave velocity, which sets the wavelength.
    density : float
        Density.
    dt : float
        Time step used by the initial object and the analytical reference.
    reduced : bool, optional
        If True, use 3 receivers instead of 11, for testing. Default is
        False.

    Returns
    -------
    dict
        Spyro model dictionary for :class:`spyro.IsotropicWave`.
    """
    if method not in ("mass_lumped_triangle", "spectral_quadrilateral"):
        raise ValueError("Method is not mass_lumped_triangle or spectral_quadrilateral")
    _warn_if_meters_per_second(s_wave_velocity)

    s_wavelength = s_wave_velocity / frequency
    delay = 1.0 / frequency

    # Acquisition, in S-wavelengths, following Lyu et al. (2024)
    # doi: 10.1029/2023JB027576
    vertical_offset = 7 * s_wavelength
    maximum_horizontal_offset = 7 * s_wavelength
    receiver_quantity = 3 if reduced else 11
    farthest_offset = np.hypot(vertical_offset, maximum_horizontal_offset)

    # S-wave reaches the farthest receiver, plus two periods of wavelet tail
    final_time = _round_up_to_time_step(
        farthest_offset / s_wave_velocity + delay + 2.0 / frequency, dt
    )

    # Nearest periodic image is at least (length - source-receiver span) away
    # in some direction; keep its P arrival after the final time.
    length = p_wave_velocity * final_time + maximum_horizontal_offset + s_wavelength

    source_z = -(length - vertical_offset) / 2.0
    source_x = (length - maximum_horizontal_offset) / 2.0
    receiver_z = source_z - vertical_offset
    receiver_locations = [
        (receiver_z, float(receiver_x))
        for receiver_x in np.linspace(
            source_x,
            source_x + maximum_horizontal_offset,
            receiver_quantity,
        )
    ]

    return {
        "options": {
            "method": method,
            "degree": degree,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": length,
            "length_x": length,
            "mesh_type": "firedrake_mesh",
            "periodic": True,
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(source_z, source_x)],
            "frequency": frequency,
            "delay": delay,
            "delay_type": "time",
            "receiver_locations": receiver_locations,
            "amplitude": np.array([0.0, 1.0]),
            "use_vertex_only_mesh": True,
        },
        "synthetic_data": {
            "type": "object",
            "density": density,
            "p_wave_velocity": p_wave_velocity,
            "s_wave_velocity": s_wave_velocity,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": final_time,
            "dt": dt,
            "output_frequency": 1000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": True,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def create_elastic_model_for_meshing_parameter_2D_heterogeneous(
    method: str,
    degree: int,
    frequency: float,
    case: HeterogeneousCase,
) -> dict:
    """Create a 2D heterogeneous isotropic elastic model for the cpw calculator.

    By default, a horizontal (x-direction) point force at
    :data:`ELASTIC_HETEROGENEOUS_SOURCE_LOCATION` records on a horizontal
    line of receivers at :data:`ELASTIC_HETEROGENEOUS_RECEIVER_DEPTH`; the
    case can replace the acquisition and time axis. The domain has
    first-order Stacey absorbing boundaries and no absorbing layer. The
    reference is numerical, so the same boundaries are present in the
    reference and in every evaluated solve.

    The P-wave velocity, S-wave velocity and density files of the case are
    written, if needed, before the dictionary is returned, and the
    dictionary reads the material from them.

    Parameters
    ----------
    method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    degree : int
        Spatial polynomial degree.
    frequency : float
        Ricker source peak frequency.
    case : HeterogeneousCase
        Domain, material and acquisition overrides.

    Returns
    -------
    dict
        Spyro model dictionary for :class:`spyro.IsotropicWave`.

    Raises
    ------
    ValueError
        If ``method`` is not supported.
    """
    if method not in MESH_TYPES:
        raise ValueError("Method is not mass_lumped_triangle or spectral_quadrilateral")
    receiver_locations = case.receiver_locations
    if receiver_locations is None:
        receiver_x_first, receiver_x_last = ELASTIC_HETEROGENEOUS_RECEIVER_X_RANGE
        receiver_locations = spyro.create_transect(
            (ELASTIC_HETEROGENEOUS_RECEIVER_DEPTH, receiver_x_first),
            (ELASTIC_HETEROGENEOUS_RECEIVER_DEPTH, receiver_x_last),
            ELASTIC_HETEROGENEOUS_RECEIVER_QUANTITY,
        )
    material_files = case.write_material_files(
        ("p_wave_velocity", "s_wave_velocity", "density")
    )

    return {
        "options": {
            "method": method,
            "degree": degree,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": case.length_z,
            "length_x": case.length_x,
            # Uniform for both methods; its boundary ids are the ones the
            # Stacey boundary terms integrate over
            "mesh_type": "firedrake_mesh",
            "periodic": False,
        },
        "absorving_boundary_conditions": {
            "status": True,
            "abc_type": "nrbc",
            "nrbc": {"type": "Stacey", "dt_scheme": "backward"},
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": _or_default(
                case.source_locations, [ELASTIC_HETEROGENEOUS_SOURCE_LOCATION]
            ),
            "frequency": frequency,
            "delay": 1.0 / frequency,
            "delay_type": "time",
            "receiver_locations": receiver_locations,
            "amplitude": np.array([0.0, 1.0]),
            "use_vertex_only_mesh": True,
        },
        "synthetic_data": {
            "type": "file",
            **material_files,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": _or_default(
                case.final_time, ELASTIC_HETEROGENEOUS_FINAL_TIME
            ),
            "dt": _or_default(case.dt, ELASTIC_HETEROGENEOUS_DT),
            "output_frequency": 1000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }
