"""Generate weak forms used in the modal solver."""

import firedrake as fire
from firedrake import grad, inner
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy.sparse import csr_matrix


def acoustic_weak_forms_c_in_stiffness(
    c, V, quad_rule=None, source=False, user_load=None
):
    """Generate the acoustic weak forms with the velocity in the stiffness.

    The stiffness is ``c**2 * grad(u) . grad(v)`` and the mass is
    ``u * v``. This differs from the forward acoustic problem, whose
    stiffness is ``grad(u) . grad(v)`` and whose mass is ``u * v / c**2``.
    Both share the same eigenvalues for a homogeneous velocity, but not for
    a heterogeneous one. For a time-step estimate consistent with the
    forward problem use ``AcousticWave.modal_weak_forms``.

    Also, it can generate a source term in weak form.

    Parameters
    ----------
    c : `Firedrake.Function`
        Velocity model.
    V : `Firedrake.FunctionSpace`
        Function space for the modal problem.
    quad_rule : `dict`, optional
        Quadrature rule to use for the integration.
        Default is `None`, which uses the default quadrature rule.
    source : `bool`, optional
        Option to get a source term in weak form. Default is `False`
    user_load : `Firedrake.Function`, optional
        User-defined load for the source term. Default is `None`, in
        which a small constant load is applied over the entire domain.

    Returns
    -------
    a : `Firedrake.Form`
        Weak form representing the stiffness matrix.
    m : `Firedrake.Form`
        Weak form  representing the mass matrix.
    L : `Firedrake.Form`, optional
        Weak form representing a source term. Returned only if 'source' is `True`.
    """

    # Functions for the problem
    u, v = fire.TrialFunction(V), fire.TestFunction(V)
    dx = fire.dx(**quad_rule) if quad_rule else fire.dx

    # Bilinear forms
    a = c * c * inner(grad(u), grad(v)) * dx

    if source:  # Source term
        if user_load is None:
            q = 1e-3
        else:
            q = user_load

        L = q * v * dx
        return a, L

    m = inner(u, v) * dx

    return a, m


def gather_csr_matrix(petsc_matrix: PETSc.Mat) -> csr_matrix:
    """Gather a distributed PETSc matrix into a SciPy matrix on every rank.

    Each rank only holds its own rows of a parallel matrix, so the rows of
    every rank are gathered, in rank order, which is the global row order.

    Parameters
    ----------
    petsc_matrix : PETSc.Mat
        AIJ matrix, possibly distributed over its communicator.

    Returns
    -------
    scipy.sparse.csr_matrix
        The whole matrix, identical on every rank of the communicator.
    """
    ptr, ind, val = petsc_matrix.getValuesCSR()
    comm = petsc_matrix.comm.tompi4py()
    # Each rank's row pointer starts at zero, so it is offset by the
    # number of nonzeros on the lower ranks
    nonzeros_per_row = np.concatenate(
        [[0]] + [rank_ptr[1:] - rank_ptr[:-1] for rank_ptr in comm.allgather(ptr)]
    )
    return csr_matrix(
        (
            np.concatenate(comm.allgather(val)),
            np.concatenate(comm.allgather(ind)),
            np.cumsum(nonzeros_per_row),
        ),
        petsc_matrix.size,
    )


def assemble_sparse_matrices(a, m, return_M_inv=False):
    """Assemble the sparse matrices for numerical modal solvers (e.g., SciPy solvers).

    In parallel, every rank gets the whole matrices.

    Parameters
    ----------
    a : `Firedrake.Form`
        Weak form representing the stiffness matrix.
    m : `Firedrake.Form`
        Weak form  representing the mass matrix.
    return_M_inv : `bool`, optional
        Option to return the inverse mass matrix instead of the mass.

    Returns
    -------
    Asp : `csr matrix`
        Sparse matrix representing the stiffness matrix.
    Msp : `csr matrix`
        Sparse matrix representing the mass matrix.
    Msp_inv : `csr matrix`, optional
        Sparse matrix representing the inverse mass matrix.
        Returned only if 'return_M_inv' is `True`
    """

    # Assemble the stiffness matrix
    Asp = gather_csr_matrix(fire.assemble(a).petscmat)

    # Assemble the mass matrix
    Msp = gather_csr_matrix(fire.assemble(m).petscmat)

    if return_M_inv:
        # Assemble the inverse mass matrix
        m_val_inv = np.array(Msp.data)
        m_val_inv[m_val_inv != 0.] = 1. / m_val_inv[m_val_inv != 0.]
        Msp_inv = csr_matrix(
            (m_val_inv, Msp.indices, Msp.indptr), Msp.shape,
        )
        return Asp, Msp_inv

    return Asp, Msp


def maximum_diagonal_ratio(a: fire.Form, m: fire.Form) -> float:
    """Return the largest ratio of the stiffness and mass matrix diagonals.

    This is ``max |A_ii / M_ii|`` over the rows with a nonzero mass
    diagonal, the largest eigenvalue of a lumped mass system. Each rank
    only reads the diagonal entries of its own rows, so nothing is gathered.

    Parameters
    ----------
    a : firedrake.Form
        Weak form representing the stiffness matrix.
    m : firedrake.Form
        Weak form representing the mass matrix.

    Returns
    -------
    float
        The largest diagonal ratio, identical on every rank.

    Raises
    ------
    ValueError
        If no row has both a nonzero mass and a nonzero stiffness diagonal,
        since the ratio would then give an infinite time step.
    """
    a_diagonal = fire.assemble(a).petscmat.getDiagonal()
    m_diagonal = fire.assemble(m).petscmat.getDiagonal()
    a_values = a_diagonal.getArray(readonly=True)
    m_values = m_diagonal.getArray(readonly=True)
    nonzero = m_values != 0.0
    ratios = np.abs(a_values[nonzero] / m_values[nonzero])
    local_maximum = ratios.max() if ratios.size else 0.0
    maximum = a_diagonal.comm.tompi4py().allreduce(local_maximum, op=MPI.MAX)
    if maximum == 0.0:
        raise ValueError(
            "The stiffness and mass diagonals have no nonzero ratio, so no "
            "time step can be estimated from them."
        )
    return maximum
