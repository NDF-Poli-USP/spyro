"""Tests of the nodes of the water cells of a marked mesh."""

import firedrake as fire
import numpy as np
import pytest
from mpi4py import MPI

from spyro.meshing.meshing_utils import (
    WATER_CELL_TAG,
    WATER_PADDING_CELL_TAG,
    water_nodes,
)

SEABED = -0.5
ROCK_CELL_TAG = WATER_CELL_TAG + 1


def tagged_mesh(
    water_padding_from: float | None = None, comm: MPI.Comm = fire.COMM_WORLD
) -> fire.MeshGeometry:
    """Return a mesh whose cells above `SEABED` are tagged as water.

    Parameters
    ----------
    water_padding_from : `float` or `None`, optional
        If given, the water cells beyond this `x` are tagged as the padding
        beside the water of a gmsh mesh. Default is `None`.
    comm : `mpi4py.MPI.Comm`, optional
        Communicator of the mesh. Default is `firedrake.COMM_WORLD`.

    Returns
    -------
    `firedrake.MeshGeometry`
        Unit square in `(z, x)`, with a node row at `SEABED`.
    """
    # Lx and Ly are the coordinates of the upper corner, not lengths
    mesh = fire.RectangleMesh(8, 6, 0.0, 1.0, originX=-1.0, comm=comm)
    DG0 = fire.FunctionSpace(mesh, "DG", 0)
    z, x = fire.SpatialCoordinate(mesh)
    in_water = fire.conditional(z > SEABED, 1, 0)
    beside = (
        0
        if water_padding_from is None
        else fire.conditional(x > water_padding_from, 1, 0)
    )
    indicators = [in_water * (1 - beside), in_water * beside, 1 - in_water]
    return fire.RelabeledMesh(
        mesh,
        [fire.Function(DG0).interpolate(i) for i in indicators],
        [WATER_CELL_TAG, WATER_PADDING_CELL_TAG, ROCK_CELL_TAG],
    )


def node_depths(V: fire.FunctionSpace) -> np.ndarray:
    """Return the `z` coordinate of the owned nodes of `V`.

    Parameters
    ----------
    V : `firedrake.FunctionSpace`
        Scalar Lagrange space.

    Returns
    -------
    `numpy.ndarray`
        Depth of each owned node, in the order of `V`.
    """
    W = fire.VectorFunctionSpace(V.mesh(), V.ufl_element())
    return fire.Function(W).interpolate(V.mesh().coordinates).dat.data_ro[:, 0]


@pytest.mark.parametrize("water_padding_from", [None, 0.6])
@pytest.mark.parametrize("degree", [1, 2])
def test_water_nodes_exclude_the_seabed(
    degree: int, water_padding_from: float | None
) -> None:
    """Check that only the nodes strictly above the seabed are water nodes.

    Parameters
    ----------
    degree : `int`
        Degree of the Lagrange space, 2 to include nodes on cell edges.
    water_padding_from : `float` or `None`
        Where the water padding tag starts, see `tagged_mesh`.

    Returns
    -------
    None
    """
    V = fire.FunctionSpace(tagged_mesh(water_padding_from), "CG", degree)
    assert np.array_equal(water_nodes(V), node_depths(V) > SEABED + 1e-9)


def test_water_nodes_require_water_cells() -> None:
    """Check that a mesh without water cells is refused.

    Returns
    -------
    None
    """
    V = fire.FunctionSpace(fire.UnitSquareMesh(2, 2), "CG", 1)
    with pytest.raises(ValueError, match="tagged"):
        water_nodes(V)


def test_water_nodes_refuse_a_mesh_of_water_only() -> None:
    """Check that a mesh whose every cell is tagged as water is refused.

    A gmsh mesh made without a water interface tags its rock with the number
    of the water.

    Returns
    -------
    None
    """
    mesh = fire.UnitSquareMesh(2, 2)
    one = fire.Function(fire.FunctionSpace(mesh, "DG", 0)).assign(1.0)
    mesh = fire.RelabeledMesh(mesh, [one], [WATER_CELL_TAG])
    with pytest.raises(ValueError, match="water interface"):
        water_nodes(fire.FunctionSpace(mesh, "CG", 1))
