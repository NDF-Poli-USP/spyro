"""Tests for the Animate-based velocity-adapted meshing prototype."""

import firedrake as fire
import numpy as np
import pytest

import spyro
from spyro.meshing import animate_based_wrappers
from spyro.meshing.animate_based_wrappers import (
    create_animate_velocity_mesh,
    sample_grid_velocity,
)

FREQUENCY = 5.0
CPW = 2.67
BBOX = (-1.0, 0.0, 0.0, 2.0)
V_SLOW = 1.5
V_FAST = 3.0


def two_layer_grid(nz: int = 41, nx: int = 81) -> np.ndarray:
    """Build a two-layer velocity grid in spyro's ``(z, x)`` convention.

    Parameters
    ----------
    nz : int, optional
        Number of samples in depth.
    nx : int, optional
        Number of samples in the horizontal direction.

    Returns
    -------
    numpy.ndarray
        Grid with ``V_FAST`` below ``z = -0.5`` and ``V_SLOW`` above it.
    """
    z = np.linspace(BBOX[0], BBOX[1], nz)
    vp = np.full((nz, nx), V_SLOW)
    vp[z < -0.5, :] = V_FAST
    return vp


def longest_edges(mesh: fire.MeshGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Return the longest edge and the centroid depth of every cell.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Triangular mesh.

    Returns
    -------
    edge : numpy.ndarray
        Longest edge length of each cell.
    depth : numpy.ndarray
        z coordinate of each cell centroid.
    """
    xy = mesh.coordinates.dat.data_ro[
        mesh.coordinates.function_space().cell_node_list
    ]
    edge = np.max([
        np.linalg.norm(xy[:, (i + 1) % 3] - xy[:, i], axis=1) for i in range(3)
    ], axis=0)
    return edge, xy[:, :, 0].mean(axis=1)


def test_sample_grid_velocity_orientation():
    """Linear grids are reproduced exactly and row 0 is the deepest sample."""
    nz, nx = 11, 21
    z = np.linspace(BBOX[0], BBOX[1], nz)
    x = np.linspace(BBOX[2], BBOX[3], nx)
    vp_grid = 2.0 + 0.5 * z[:, None] + 0.25 * x[None, :]
    mesh = spyro.meshing.rectangle_mesh(7, 9, 1.0, 2.0)

    vp = sample_grid_velocity(fire.FunctionSpace(mesh, "CG", 2), vp_grid, BBOX)

    zz, xx = fire.SpatialCoordinate(mesh)
    error = fire.assemble((vp - (2.0 + 0.5 * zz + 0.25 * xx)) ** 2 * fire.dx)
    assert np.sqrt(error) < 1e-12


def test_check_animate_raises_without_animate(monkeypatch):
    """A clear ImportError is raised when Animate is not installed."""
    monkeypatch.setattr(animate_based_wrappers, "animate", None)
    with pytest.raises(ImportError, match="Animate is not available"):
        create_animate_velocity_mesh(two_layer_grid(), BBOX, FREQUENCY, CPW)


def test_invalid_number_of_iterations():
    """At least one adaptation pass is required."""
    pytest.importorskip("animate")
    with pytest.raises(ValueError, match="num_iterations"):
        create_animate_velocity_mesh(
            two_layer_grid(), BBOX, FREQUENCY, CPW, num_iterations=0
        )


def test_two_layer_mesh_follows_wavelength():
    """Cells honour the cpw size in each layer and the mesh beats uniform."""
    pytest.importorskip("animate")
    vp_grid = two_layer_grid()
    h_slow = V_SLOW / (FREQUENCY * CPW)
    h_fast = V_FAST / (FREQUENCY * CPW)

    mesh = create_animate_velocity_mesh(vp_grid, BBOX, FREQUENCY, CPW)

    coords = mesh.coordinates.dat.data_ro
    assert np.allclose(coords.min(axis=0), [BBOX[0], BBOX[2]])
    assert np.allclose(coords.max(axis=0), [BBOX[1], BBOX[3]])

    edge, depth = longest_edges(mesh)
    slow = depth > -0.5 + h_fast
    fast = depth < -0.5 - 2 * h_fast
    # Mmg targets unit edges in the metric; allow its usual tolerance.
    assert np.median(edge[slow]) < 1.15 * h_slow
    assert np.median(edge[fast]) > 1.5 * h_slow
    assert np.median(edge[fast]) < 1.15 * h_fast

    uniform = spyro.meshing.rectangle_mesh(
        int(np.ceil(1.0 / h_slow)), int(np.ceil(2.0 / h_slow)), 1.0, 2.0
    )
    assert mesh.num_cells() < uniform.num_cells()
