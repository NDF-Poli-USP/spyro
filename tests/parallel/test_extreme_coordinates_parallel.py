"""Extreme mesh coordinates of a mesh distributed over two ranks.

Run with ``mpiexec -n 2 pytest tests/parallel/test_extreme_coordinates_parallel.py``.
Each rank holds only part of the mesh, but the extremes must be those of the
whole mesh, or the absorbing layer checks fail on every rank.
"""
import numpy as np
import pytest
from firedrake import COMM_WORLD, RectangleMesh

from spyro.meshing.meshing_operations import MeshOps
from spyro.version_control import is_firedrake_pyop3


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


@pytest.mark.parallel(2)
def test_extreme_coordinates_with_a_rank_without_nodes() -> None:
    """Check the extremes when a rank holds no node of the mesh.

    A mesh of one quadrilateral on two ranks leaves one rank without cells or
    nodes. That rank must contribute nothing to the reduction, and both ranks
    must get the limits of the cell.

    Returns
    -------
    None
    """
    mesh = RectangleMesh(1, 1, 1.0, 2.0, quadrilateral=True, comm=COMM_WORLD)
    num_nodes = mesh.coordinates.dat.data_ro_with_halos.shape[0]
    # A partition with nodes on every rank would not test anything
    assert 0 in COMM_WORLD.allgather(num_nodes)

    min_coordinates, max_coordinates = MeshOps.extract_extreme_coordinates(mesh)

    np.testing.assert_allclose(min_coordinates, [0.0, 0.0])
    np.testing.assert_allclose(max_coordinates, [1.0, 2.0])


@pytest.mark.skipif(
    is_firedrake_pyop3(), reason="dat_version is a counter of PyOP2 data"
)
@pytest.mark.parallel(2)
def test_extreme_coordinates_do_not_modify_the_mesh() -> None:
    """Check that the coordinates are only read.

    Accessing the coordinates for writing increments the version of their
    data, which marks them as modified (caches such as the spatial index are
    then rebuilt) and invalidates their halos.

    Returns
    -------
    None
    """
    mesh = RectangleMesh(4, 40, 1.0, 10.0, comm=COMM_WORLD)
    version = mesh.coordinates.dat.dat_version

    MeshOps.extract_extreme_coordinates(mesh)

    assert mesh.coordinates.dat.dat_version == version
