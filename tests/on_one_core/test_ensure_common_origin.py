"""Common origin of the meshes with a pad.

``AutomaticMesh.ensure_common_origin`` moves a mesh with a pad so that its
minimum x (and y in 3D) is ``-pad``, in either direction. The helpers of this
module are also used by ``tests/parallel/test_ensure_common_origin_parallel.py``.
"""

from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest
from mpi4py import MPI

from spyro.meshing.meshing_functions import AutomaticMesh

# Pad of the meshes in km
PAD = 1.0


def automatic_mesh(dimension: int) -> AutomaticMesh:
    """Build an `AutomaticMesh` with the pad of the tests.

    Parameters
    ----------
    dimension : `int`
        Dimension of the mesh, 2 or 3.

    Returns
    -------
    `AutomaticMesh`
        Meshing object whose only parameters used here are the dimension and
        the pad.
    """
    mesh_parameters = SimpleNamespace(
        dimension=dimension,
        length_z=1.0,
        length_x=1.0,
        length_y=1.0 if dimension == 3 else 0.0,
        quadrilateral=False,
        comm=None,
        mesh_type="firedrake_mesh",
        edge_length=0.25,
        abc_pad_length=PAD,
        periodic=False,
        cells_per_wavelength=None,
        source_frequency=None,
        minimum_velocity=None,
        velocity_model=None,
        output_filename=None,
    )
    return AutomaticMesh(mesh_parameters=mesh_parameters)


def long_mesh(dimension: int, shift: float) -> tuple[fire.MeshGeometry, int]:
    """Build a mesh elongated along one horizontal axis.

    The mesh is 1 km deep (-1 <= z <= 0) and 8 km long along x in 2D or along
    y in 3D, so that two ranks split it along that axis. The minimum of every
    horizontal coordinate is ``-PAD + shift``.

    Parameters
    ----------
    dimension : `int`
        Dimension of the mesh, 2 or 3.
    shift : `float`
        Distance in km from ``-PAD`` to the minimum of the horizontal
        coordinates.

    Returns
    -------
    mesh : `Firedrake.MeshGeometry`
        Mesh with coordinates (z, x) or (z, x, y).
    axis : `int`
        Index of the elongated coordinate: 1 (x) in 2D and 2 (y) in 3D.
    """
    if dimension == 2:
        mesh = fire.RectangleMesh(2, 16, 1.0, 8.0)
        axis = 1
    else:
        mesh = fire.BoxMesh(1, 1, 8, 1.0, 1.0, 8.0)
        axis = 2

    coords = mesh.coordinates.dat.data_with_halos
    coords[:, 0] -= 1.0
    coords[:, 1:] += -PAD + shift
    return mesh, axis


def global_range(mesh: fire.MeshGeometry, axis: int) -> tuple[float, float]:
    """Return the range of a coordinate over every rank.

    Parameters
    ----------
    mesh : `Firedrake.MeshGeometry`
        Mesh, possibly distributed over several ranks.
    axis : `int`
        Index of the coordinate.

    Returns
    -------
    `tuple`
        Minimum and maximum of the coordinate, identical on every rank.
    """
    values = mesh.coordinates.dat.data_ro_with_halos[:, axis]
    comm = mesh.comm
    return (
        comm.allreduce(values.min(initial=np.inf), op=MPI.MIN),
        comm.allreduce(values.max(initial=-np.inf), op=MPI.MAX),
    )


def check_shifted_mesh(
    mesh: fire.MeshGeometry, before: np.ndarray, axis: int, shift: float
) -> None:
    """Check that the horizontal coordinates moved by ``-shift`` to ``-PAD``.

    Parameters
    ----------
    mesh : `Firedrake.MeshGeometry`
        Mesh after ``ensure_common_origin``.
    before : `array`
        Coordinates of the nodes of the rank, halos included, before it.
    axis : `int`
        Index of the elongated coordinate, 8 km long.
    shift : `float`
        Distance in km from ``-PAD`` to the minimum horizontal coordinates
        before ``ensure_common_origin``.

    Returns
    -------
    None
    """
    expected = before.copy()
    expected[:, 1:] -= shift
    np.testing.assert_allclose(
        mesh.coordinates.dat.data_ro_with_halos, expected, rtol=0.0, atol=1e-12
    )
    for horizontal in range(1, before.shape[1]):
        np.testing.assert_allclose(
            global_range(mesh, horizontal),
            (-PAD, 8.0 - PAD if horizontal == axis else 1.0 - PAD),
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize("shift", [0.25, -0.25])
@pytest.mark.parametrize("dimension", [2, 3])
def test_ensure_common_origin_shifts_mesh(dimension: int, shift: float) -> None:
    """Check that the mesh moves to the common origin from either side.

    The mesh starts ``shift`` to the right of ``-PAD`` (to the left if
    negative). After ``ensure_common_origin``, the minimum of each horizontal
    coordinate must be ``-PAD``, with every node moved by ``-shift`` and the
    depth unchanged.

    Parameters
    ----------
    dimension : `int`
        Dimension of the mesh, 2 or 3.
    shift : `float`
        Distance in km from ``-PAD`` to the minimum horizontal coordinates.

    Returns
    -------
    None
    """
    mesh, axis = long_mesh(dimension, shift)
    before = mesh.coordinates.dat.data_ro_with_halos.copy()

    automatic_mesh(dimension).ensure_common_origin(mesh, pad=PAD)

    check_shifted_mesh(mesh, before, axis, shift)


@pytest.mark.parametrize("dimension", [2, 3])
def test_ensure_common_origin_keeps_aligned_mesh(dimension: int) -> None:
    """Check that a mesh already at the common origin is not moved.

    Parameters
    ----------
    dimension : `int`
        Dimension of the mesh, 2 or 3.

    Returns
    -------
    None
    """
    mesh, _ = long_mesh(dimension, 0.0)
    before = mesh.coordinates.dat.data_ro_with_halos.copy()

    automatic_mesh(dimension).ensure_common_origin(mesh, pad=PAD)

    np.testing.assert_array_equal(mesh.coordinates.dat.data_ro_with_halos, before)
