"""Nodes of the water cells of a mesh distributed over two ranks.

Run with ``mpiexec -n 2 pytest tests/parallel/test_water_nodes_parallel.py``.
A node on the seabed can have its rock cells on another rank, so the water
nodes must not be found from the cells of one rank alone.
"""

import numpy as np
import pytest
from firedrake import COMM_SELF, COMM_WORLD, FunctionSpace

from spyro.meshing.meshing_utils import water_nodes
from tests.on_one_core.test_meshing_water_nodes import (
    SEABED,
    node_depths,
    tagged_mesh,
)


@pytest.mark.parallel(2)
@pytest.mark.parametrize("water_padding_from", [None, 0.6])
@pytest.mark.parametrize("degree", [1, 2])
def test_water_nodes_in_parallel(degree: int, water_padding_from: float | None) -> None:
    """Check the water nodes of every rank against the serial mesh.

    Parameters
    ----------
    degree : `int`
        Degree of the Lagrange space.
    water_padding_from : `float` or `None`
        Where the water padding tag starts, see `tagged_mesh`.

    Returns
    -------
    None
    """
    assert COMM_WORLD.size == 2
    V = FunctionSpace(tagged_mesh(water_padding_from), "CG", degree)
    in_water = water_nodes(V)
    assert np.array_equal(in_water, node_depths(V) > SEABED + 1e-9)

    serial = FunctionSpace(tagged_mesh(water_padding_from, COMM_SELF), "CG", degree)
    assert COMM_WORLD.allreduce(int(in_water.sum())) == int(water_nodes(serial).sum())
