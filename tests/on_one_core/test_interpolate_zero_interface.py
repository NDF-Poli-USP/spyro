"""Tests of the interpolation of material models with a zero layer."""

from pathlib import Path

import firedrake as fire
import h5py
import numpy as np
import pytest

import spyro
from spyro.io.interpolators import _sample_grid
from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave

GRID_SPACING = 0.1
WATER_BOTTOM = -0.4  # Last water sample
ROCK_TOP = -0.5  # First rock sample
# Nodes in the grid cell between them, nearer the water and nearer the rock
NEAR_WATER = -0.43
NEAR_ROCK = -0.47


def rock(z: np.ndarray, x: np.ndarray, y: np.ndarray | float = 0.0) -> np.ndarray:
    """Return the rock value, linear so that linear interpolation is exact.

    Parameters
    ----------
    z : `numpy.ndarray`
        Depth coordinate, in km.
    x : `numpy.ndarray`
        Lateral coordinate, in km.
    y : `numpy.ndarray` or `float`, optional
        Second lateral coordinate, in km. Default is `0.0`.

    Returns
    -------
    `numpy.ndarray`
        Value in km/s.
    """
    return 2.0 + x + 0.5 * y - z


def grid_axes(dimension: int) -> tuple[np.ndarray, ...]:
    """Return the sample coordinates of a unit grid.

    Parameters
    ----------
    dimension : `int`
        2 for `(z, x)` or 3 for `(z, x, y)`.

    Returns
    -------
    `tuple` of `numpy.ndarray`
        `z` from -1 to 0, then `x` (and `y`) from 0 to 1.
    """
    n = round(1.0 / GRID_SPACING) + 1
    return (np.linspace(-1.0, 0.0, n),) + (np.linspace(0.0, 1.0, n),) * (dimension - 1)


def grid_values(axes: tuple[np.ndarray, ...], water: bool = True) -> np.ndarray:
    """Return the grid of `rock`, zero above `ROCK_TOP` if `water`.

    Parameters
    ----------
    axes : `tuple` of `numpy.ndarray`
        Sample coordinates, from `grid_axes`.
    water : `bool`, optional
        If `True`, the samples above `ROCK_TOP` are zero. Default is `True`.

    Returns
    -------
    `numpy.ndarray`
        Grid values.
    """
    coords = np.meshgrid(*axes, indexing="ij")
    values = rock(*coords)
    if water:
        values[coords[0] > ROCK_TOP + 1e-9] = 0.0
    return values


def nearest_sample(
    axes: tuple[np.ndarray, ...], values: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Return the value of the grid sample nearest to each point.

    Parameters
    ----------
    axes : `tuple` of `numpy.ndarray`
        Sample coordinates, uniformly spaced by `GRID_SPACING`.
    values : `numpy.ndarray`
        Grid values.
    points : `numpy.ndarray`
        Coordinates of shape `(n_points, len(axes))`.

    Returns
    -------
    `numpy.ndarray`
        Nearest sample values.
    """
    index = [
        np.rint((points[:, i] - axis[0]) / GRID_SPACING).astype(int)
        for i, axis in enumerate(axes)
    ]
    return values[tuple(index)]


def interface_points(dimension: int) -> np.ndarray:
    """Return points on rows across the water bottom and in the open layers.

    Parameters
    ----------
    dimension : `int`
        2 for `(z, x)` or 3 for `(z, x, y)`.

    Returns
    -------
    `numpy.ndarray`
        Coordinates of shape `(n_points, dimension)`, off the grid laterally.
    """
    rows = [-0.85, -0.6, ROCK_TOP, NEAR_ROCK, NEAR_WATER, WATER_BOTTOM, -0.25]
    lateral = [np.linspace(0.0, 1.0, 8)] * (dimension - 1)
    coords = np.meshgrid(rows, *lateral, indexing="ij")
    return np.column_stack([c.ravel() for c in coords])


@pytest.mark.parametrize("dimension", [2, 3])
def test_sample_grid_marked_mesh_takes_rock_values(dimension: int):
    """Check that nodes in mixed cells get the rock samples of the cell only.

    Parameters
    ----------
    dimension : `int`
        Grid dimension.

    Returns
    -------
    None
    """
    axes = grid_axes(dimension)
    values = grid_values(axes)
    points = interface_points(dimension)
    sampled = _sample_grid(axes, values, points, True, True)
    linear = _sample_grid(axes, values, points, False, True)

    z, lateral = points[:, 0], points[:, 1:].T
    mixed = (z > ROCK_TOP) & (z < WATER_BOTTOM)
    # The only rock samples of these cells are on the ROCK_TOP row
    assert np.allclose(sampled[mixed], rock(ROCK_TOP, *lateral)[mixed])
    assert np.all(linear[mixed] < rock(ROCK_TOP, *lateral)[mixed] - 0.1)
    # Points on the WATER_BOTTOM row are a rounding error away from the mixed
    # cell, as the grid row is at -0.3999999999999999
    assert np.array_equal(sampled[~mixed], linear[~mixed])
    assert np.allclose(sampled[z >= WATER_BOTTOM], 0.0, rtol=0.0, atol=1e-12)
    assert np.allclose(sampled[z <= ROCK_TOP], rock(*points[z <= ROCK_TOP].T))


@pytest.mark.parametrize("dimension", [2, 3])
def test_sample_grid_unmarked_mesh_takes_nearest_sample(dimension: int):
    """Check that nodes in mixed cells get the nearest sample, zero or not.

    Parameters
    ----------
    dimension : `int`
        Grid dimension.

    Returns
    -------
    None
    """
    axes = grid_axes(dimension)
    values = grid_values(axes)
    points = interface_points(dimension)
    sampled = _sample_grid(axes, values, points, True, False)
    linear = _sample_grid(axes, values, points, False, False)

    z = points[:, 0]
    mixed = (z > ROCK_TOP) & (z < WATER_BOTTOM)
    nearest = nearest_sample(axes, values, points)
    assert np.array_equal(sampled[mixed], nearest[mixed])
    assert np.all(sampled[z == NEAR_WATER] == 0.0)
    assert np.all(sampled[z == NEAR_ROCK] > 0.0)
    assert np.array_equal(sampled[~mixed], linear[~mixed])


@pytest.mark.parametrize("mesh_follows_interface", [True, False])
def test_sample_grid_without_zeros_is_linear(mesh_follows_interface: bool):
    """Check that a grid with no zeros is sampled as without the option.

    Parameters
    ----------
    mesh_follows_interface : `bool`
        Whether the points belong to a mesh that follows the interface.

    Returns
    -------
    None
    """
    axes = grid_axes(2)
    values = grid_values(axes, water=False)
    points = interface_points(2)
    assert np.array_equal(
        _sample_grid(axes, values, points, True, mesh_follows_interface),
        _sample_grid(axes, values, points, False, mesh_follows_interface),
    )


def wave_and_space(
    water_interface: bool = False,
    wave_class: type[spyro.solvers.wave.Wave] = spyro.AcousticWave,
) -> tuple[spyro.solvers.wave.Wave, fire.FunctionSpace]:
    """Build a wave on a user mesh with node rows inside the mixed cell.

    Parameters
    ----------
    water_interface : `bool`, optional
        Whether the mesh is declared to follow the water bottom. Default is
        `False`.
    wave_class : `type`, optional
        Wave to build. Default is `spyro.AcousticWave`.

    Returns
    -------
    `tuple`
        Wave and CG1 space on its mesh.
    """
    # Lx and Ly are the coordinates of the upper corner, not lengths
    mesh = fire.RectangleMesh(10, 7, 0.0, 1.0, originX=-1.0)
    z = mesh.coordinates.dat.data[:, 0]
    z[np.isclose(z, WATER_BOTTOM)] = NEAR_WATER
    z[np.isclose(z, ROCK_TOP)] = NEAR_ROCK
    dictionary = {
        "options": {
            "cell_type": "T",
            "variant": "equispaced",
            "degree": 1,
            "dimension": 2,
        },
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "user_mesh": mesh,
            "mesh_file": None,
            "water_interface": water_interface,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 1.0,
            "dt": 0.0005,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.9, 0.5)],
            "frequency": 5.0,
            "delay": 1.5,
            "receiver_locations": [(-0.9, 0.6)],
        },
    }
    return wave_class(dictionary=dictionary), fire.FunctionSpace(mesh, "CG", 1)


def write_model(directory: Path, water: bool = True, shift: float = 0.0) -> str:
    """Write the 2D grid to an HDF5 file.

    Parameters
    ----------
    directory : `pathlib.Path`
        Directory of the file.
    water : `bool`, optional
        If `True`, the grid has a water layer. Default is `True`.
    shift : `float`, optional
        Value added to the grid, with no water layer. Default is `0.0`.

    Returns
    -------
    `str`
        File name.
    """
    filename = str(directory / f"model_water_{water}_shift_{shift}.hdf5")
    with h5py.File(filename, "w") as f:
        values = grid_values(grid_axes(2), water) + shift
        f.create_dataset("velocity_model", data=values)
    return filename


def test_interpolate_blends_by_default(tmp_path: Path):
    """Check that by default nodes in mixed cells blend water and rock.

    Parameters
    ----------
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    wave, V = wave_and_space(water_interface=True)
    c = spyro.io.interpolate(wave, write_model(tmp_path), V, fast_interpolate=True)

    z, x = V.mesh().coordinates.dat.data_ro.T
    near_water = np.isclose(z, NEAR_WATER)
    near_rock = np.isclose(z, NEAR_ROCK)
    # Weights of the rock row: 0.3 and 0.7
    assert np.allclose(c.dat.data_ro[near_water], 0.3 * rock(ROCK_TOP, x[near_water]))
    assert np.allclose(c.dat.data_ro[near_rock], 0.7 * rock(ROCK_TOP, x[near_rock]))


def test_sharp_zero_interface_requires_fast_interpolate(tmp_path: Path):
    """Check that the option is refused on the projection path.

    Parameters
    ----------
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    wave, V = wave_and_space()
    with pytest.raises(ValueError, match="fast_interpolate"):
        spyro.io.interpolate(wave, write_model(tmp_path), V, sharp_zero_interface=True)


def interpolate_both_ways(
    water_interface: bool, filename: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a model file with and without `sharp_zero_interface`.

    Parameters
    ----------
    water_interface : `bool`
        Whether the mesh is declared to follow the water bottom.
    filename : `str`
        HDF5 model file.

    Returns
    -------
    `tuple` of `numpy.ndarray`
        Node coordinates, values with the option and values without it.
    """
    wave, V = wave_and_space(water_interface)
    assert wave.mesh_parameters.water_interface is water_interface
    sharp, linear = (
        spyro.io.interpolate(
            wave, filename, V, fast_interpolate=True, sharp_zero_interface=sharp
        ).dat.data_ro
        for sharp in (True, False)
    )
    return V.mesh().coordinates.dat.data_ro, sharp, linear


def test_interpolate_marked_mesh_takes_rock_values(tmp_path: Path):
    """Check that a marked mesh gets the rock row in the mixed cell.

    Parameters
    ----------
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    coords, sharp, linear = interpolate_both_ways(True, write_model(tmp_path))
    z, x = coords.T
    mixed = np.isclose(z, NEAR_WATER) | np.isclose(z, NEAR_ROCK)
    axes = grid_axes(2)
    rock_row = grid_values(axes)[np.isclose(axes[0], ROCK_TOP)][0]
    assert np.allclose(sharp[mixed], np.interp(x[mixed], axes[1], rock_row))
    assert np.all(sharp[z > WATER_BOTTOM] == 0.0)
    assert np.array_equal(sharp[~mixed], linear[~mixed])


def test_interpolate_unmarked_mesh_takes_nearest_sample(tmp_path: Path):
    """Check that a mesh that is not marked gets the nearest sample.

    Parameters
    ----------
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    coords, sharp, linear = interpolate_both_ways(False, write_model(tmp_path))
    z = coords[:, 0]
    mixed = np.isclose(z, NEAR_WATER) | np.isclose(z, NEAR_ROCK)
    axes = grid_axes(2)
    nearest = nearest_sample(axes, grid_values(axes), coords)
    assert np.array_equal(sharp[mixed], nearest[mixed])
    assert np.all(sharp[np.isclose(z, NEAR_WATER)] == 0.0)
    assert np.all(sharp[np.isclose(z, NEAR_ROCK)] > 0.0)
    assert np.array_equal(sharp[~mixed], linear[~mixed])


@pytest.mark.parametrize("water_interface", [True, False])
def test_interpolate_without_zeros_is_unchanged(water_interface: bool, tmp_path):
    """Check that a model with no zeros, such as Vp, is not changed.

    Parameters
    ----------
    water_interface : `bool`
        Whether the mesh is declared to follow the water bottom.
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    _, sharp, linear = interpolate_both_ways(
        water_interface, write_model(tmp_path, water=False)
    )
    assert np.array_equal(sharp, linear)


@pytest.mark.parametrize("sharp_zero_interface", [None, False])
def test_isotropic_wave_reads_sharp_shear_velocity(
    sharp_zero_interface: bool | None, tmp_path: Path
):
    """Check the elastic file reader, which takes the rock Vs by default.

    Parameters
    ----------
    sharp_zero_interface : `bool` or `None`
        Value passed to the reader, or `None` to keep its default.
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    wave, _ = wave_and_space(water_interface=True, wave_class=IsotropicWave)
    files = {
        "p_wave_velocity": write_model(tmp_path, water=False, shift=3.0),
        "s_wave_velocity": write_model(tmp_path),
        "density": write_model(tmp_path, water=False, shift=-1.0),
    }
    options = {}
    if sharp_zero_interface is not None:
        options["sharp_zero_interface"] = sharp_zero_interface
    wave.initialize_model_parameters_from_file({"type": "file", **files}, **options)

    V = wave.c_s.function_space()
    W = fire.VectorFunctionSpace(V.mesh(), V.ufl_element())
    z, x = fire.Function(W).interpolate(V.mesh().coordinates).dat.data_ro.T
    mixed = np.isclose(z, NEAR_WATER) | np.isclose(z, NEAR_ROCK)
    linear = {
        key: spyro.io.interpolate(wave, filename, V, fast_interpolate=True).dat.data_ro
        for key, filename in files.items()
    }
    c_s = wave.c_s.dat.data_ro
    if sharp_zero_interface is None:
        axes = grid_axes(2)
        rock_row = grid_values(axes)[np.isclose(axes[0], ROCK_TOP)][0]
        assert np.allclose(c_s[mixed], np.interp(x[mixed], axes[1], rock_row))
    else:
        assert np.array_equal(c_s[mixed], linear["s_wave_velocity"][mixed])
    assert np.array_equal(c_s[~mixed], linear["s_wave_velocity"][~mixed])
    assert np.array_equal(wave.c.dat.data_ro, linear["p_wave_velocity"])
    assert np.array_equal(wave.rho.dat.data_ro, linear["density"])


def test_isotropic_wave_sharp_zero_interface_requires_fast_interpolate(tmp_path):
    """Check that the elastic reader refuses its default on the slow path.

    Parameters
    ----------
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    wave, _ = wave_and_space(water_interface=True, wave_class=IsotropicWave)
    synthetic_data = {
        "type": "file",
        "p_wave_velocity": write_model(tmp_path, water=False, shift=3.0),
        "s_wave_velocity": write_model(tmp_path),
        "density": 1.0,
    }
    with pytest.raises(ValueError, match="fast_interpolate"):
        wave.initialize_model_parameters_from_file(
            synthetic_data, fast_interpolate=False
        )
