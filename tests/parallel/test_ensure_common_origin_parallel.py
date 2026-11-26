"""Common origin of the meshes with a pad under spatial parallelism.

Run with ``mpiexec -n 2 pytest tests/parallel/test_ensure_common_origin_parallel.py``.
``AutomaticMesh.ensure_common_origin`` moves a mesh with a pad so that its
minimum x (and y in 3D) is ``-pad``, in either direction. Each rank only holds
its part of the mesh, so the minimum must be taken over every rank and every
rank must apply the same shift. The serial cases are in
``tests/on_one_core/test_ensure_common_origin.py``, whose helpers are used here.
"""

import numpy as np
import pytest

from tests.on_one_core.test_ensure_common_origin import (
    PAD,
    automatic_mesh,
    check_shifted_mesh,
    long_mesh,
)

pytestmark = pytest.mark.parallel(2)


@pytest.mark.parametrize("shift", [0.25, -0.25])
@pytest.mark.parametrize("dimension", [2, 3])
def test_ensure_common_origin_shifts_every_rank(dimension: int, shift: float) -> None:
    """Check that every rank applies the same shift to its part of the mesh.

    The mesh starts ``shift`` to the right of ``-PAD`` (to the left if negative)
    and the partition splits the elongated axis, so the ranks hold different
    minimum coordinates. After ``ensure_common_origin``, the global minimum of
    each horizontal coordinate must be ``-PAD``, and every rank must have moved
    its nodes by the same ``-shift``, the depth being unchanged.

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

    # The partition must split the elongated axis for the test to be meaningful
    local_min = mesh.coordinates.dat.data_ro_with_halos[:, axis].min(initial=np.inf)
    assert len(set(mesh.comm.allgather(local_min))) > 1, "The axis is not split"

    before = mesh.coordinates.dat.data_ro_with_halos.copy()
    automatic_mesh(dimension).ensure_common_origin(mesh, pad=PAD)

    check_shifted_mesh(mesh, before, axis, shift)


@pytest.mark.parametrize("dimension", [2, 3])
def test_ensure_common_origin_keeps_aligned_mesh(dimension: int) -> None:
    """Check that a mesh already at the common origin is not moved on any rank.

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
