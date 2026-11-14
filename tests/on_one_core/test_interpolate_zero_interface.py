"""Tests of the interpolation of material models with a zero layer."""

from pathlib import Path

import firedrake as fire
import h5py
import numpy as np
import pytest

import spyro
from spyro.io.interpolators import _sample_grid

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


@pytest.fixture
def wave_and_space(tmp_path: Path) -> tuple[spyro.AcousticWave, fire.FunctionSpace]:
    """Build a wave on a user mesh with node rows inside the mixed cell.

    Parameters
    ----------
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    `tuple`
        `spyro.AcousticWave` and CG1 space on its mesh.
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
    return spyro.AcousticWave(dictionary=dictionary), fire.FunctionSpace(mesh, "CG", 1)


def write_model(directory: Path) -> str:
    """Write the 2D grid with a water layer to an HDF5 file.

    Parameters
    ----------
    directory : `pathlib.Path`
        Directory of the file.

    Returns
    -------
    `str`
        File name.
    """
    filename = str(directory / "vs.hdf5")
    with h5py.File(filename, "w") as f:
        f.create_dataset("velocity_model", data=grid_values(grid_axes(2)))
    return filename


def test_interpolate_blends_by_default(wave_and_space, tmp_path: Path):
    """Check that by default nodes in mixed cells blend water and rock.

    Parameters
    ----------
    wave_and_space : `tuple`
        Wave and CG1 space from the fixture.
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    wave, V = wave_and_space
    c = spyro.io.interpolate(wave, write_model(tmp_path), V, fast_interpolate=True)

    z, x = V.mesh().coordinates.dat.data_ro.T
    near_water = np.isclose(z, NEAR_WATER)
    near_rock = np.isclose(z, NEAR_ROCK)
    # Weights of the rock row: 0.3 and 0.7
    assert np.allclose(c.dat.data_ro[near_water], 0.3 * rock(ROCK_TOP, x[near_water]))
    assert np.allclose(c.dat.data_ro[near_rock], 0.7 * rock(ROCK_TOP, x[near_rock]))


def test_sharp_zero_interface_requires_fast_interpolate(wave_and_space, tmp_path):
    """Check that the option is refused on the projection path.

    Parameters
    ----------
    wave_and_space : `tuple`
        Wave and CG1 space from the fixture.
    tmp_path : `pathlib.Path`
        Pytest temporary directory.

    Returns
    -------
    None
    """
    wave, V = wave_and_space
    with pytest.raises(ValueError, match="fast_interpolate"):
        spyro.io.interpolate(wave, write_model(tmp_path), V, sharp_zero_interface=True)
