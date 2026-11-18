"""Extreme mesh coordinates of a mesh distributed over two ranks.

Run with ``mpiexec -n 2 pytest tests/parallel/test_extreme_coordinates_parallel.py``.
Each rank holds only part of the mesh, but the extremes must be those of the
whole mesh, or the absorbing layer checks fail on every rank.
"""
import numpy as np
import pytest
from firedrake import COMM_WORLD, RectangleMesh

from spyro.meshing.meshing_operations import MeshOps


@pytest.mark.parallel(2)
def test_extreme_coordinates_span_the_whole_mesh() -> None:
    """Every rank gets the limits of the whole mesh, not of its partition."""
    assert COMM_WORLD.size == 2
    mesh = RectangleMesh(4, 40, 1.0, 10.0, comm=COMM_WORLD)
    local_coordinates = mesh.coordinates.dat.data_ro
    # A partition that already spans the domain would not test anything
    assert np.ptp(local_coordinates[:, 1]) < 10.0

    min_coordinates, max_coordinates = MeshOps.extract_extreme_coordinates(mesh)

    np.testing.assert_allclose(min_coordinates, [0.0, 0.0])
    np.testing.assert_allclose(max_coordinates, [1.0, 10.0])
