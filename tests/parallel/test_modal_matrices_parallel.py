"""Modal matrices of a mesh distributed over two ranks.

Run with ``mpiexec -n 2 pytest tests/parallel/test_modal_matrices_parallel.py``.
The degree-of-freedom numbering of a distributed mesh differs from the
serial one, so the parallel matrices are compared against a serial copy of
the same mesh through quantities that do not depend on the numbering.
"""
import numpy as np
import pytest
from firedrake import COMM_SELF, COMM_WORLD, Function, FunctionSpace, RectangleMesh

from spyro.solvers.modal.modal_forms_and_matrices import (
    assemble_sparse_matrices,
    maximum_diagonal_ratio,
    weak_forms,
)


def _forms(comm) -> tuple:
    """Build the modal forms of a heterogeneous velocity on a small mesh.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator the mesh is distributed over.

    Returns
    -------
    tuple
        Stiffness and mass forms ``(a, m)``.
    """
    mesh = RectangleMesh(8, 6, 1.0, 0.75, quadrilateral=True, comm=comm)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = mesh.coordinates
    c = Function(V).interpolate(1.5 + x + 2.0 * y)
    return weak_forms(c, V)


def _invariants(matrix) -> tuple:
    """Return numbering-independent quantities of a sparse matrix.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Matrix to summarize.

    Returns
    -------
    tuple
        Shape, number of nonzeros, sorted diagonal, sorted row sums and
        Frobenius norm.
    """
    return (
        matrix.shape,
        matrix.nnz,
        np.sort(matrix.diagonal()),
        np.sort(np.asarray(matrix.sum(axis=1)).ravel()),
        np.sqrt((matrix.data**2).sum()),
    )


@pytest.mark.parallel(2)
def test_modal_matrices_match_serial():
    """Gathered matrices and the diagonal ratio match a serial assembly."""
    assert COMM_WORLD.size == 2
    a, m = _forms(COMM_WORLD)
    a_serial, m_serial = _forms(COMM_SELF)

    Asp, Msp = assemble_sparse_matrices(a, m)
    Asp_serial, Msp_serial = assemble_sparse_matrices(a_serial, m_serial)
    for parallel, serial in ((Asp, Asp_serial), (Msp, Msp_serial)):
        shape, nnz, diagonal, row_sums, norm = _invariants(parallel)
        shape_s, nnz_s, diagonal_s, row_sums_s, norm_s = _invariants(serial)
        assert shape == shape_s
        assert nnz == nnz_s
        assert np.allclose(diagonal, diagonal_s)
        assert np.allclose(row_sums, row_sums_s)
        assert np.isclose(norm, norm_s)

    # Every rank holds the whole, identical matrix
    gathered = COMM_WORLD.allgather(Asp.diagonal())
    assert np.array_equal(gathered[0], gathered[1])

    _, Msp_inv = assemble_sparse_matrices(a, m, return_M_inv=True)
    expected = np.abs(Msp_inv.multiply(Asp).diagonal()).max()
    ratio = maximum_diagonal_ratio(a, m)
    assert np.isclose(ratio, expected)
    assert np.isclose(ratio, maximum_diagonal_ratio(a_serial, m_serial))
    assert COMM_WORLD.allgather(ratio) == [ratio, ratio]
