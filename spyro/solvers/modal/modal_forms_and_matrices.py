"""Generate weak forms used in the modal solver."""

from firedrake import assemble, dx as fire_dx, grad, inner, TestFunction, TrialFunction
from numpy import array
from scipy.sparse import csr_matrix


def weak_forms(c, V, quad_rule=None, source=False, user_load=None):
    """Generate the weak forms for the modal problem.

    Also, it can generate a source term in weak form.

    Parameters
    ----------
    c : `Firedrake.Function`
        Velocity model.
    V : `Firedrake.FunctionSpace`
        Function space for the modal problem.
    quad_rule : `str`, optional
        Quadrature rule to use for the integration.
        Default is None, which uses the default quadrature rule.
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
        Weak form representing a source term. Returned only if 'source' is `True`
    """

    # Functions for the problem
    u, v = TrialFunction(V), TestFunction(V)
    dx = fire_dx(**quad_rule) if quad_rule else fire_dx

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


def assemble_sparse_matrices(a, m, return_M_inv=False):
    """Assemble the sparse matrices for numerical modal solvers (e.g., SciPy solvers).

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
    A = assemble(a)
    a_ptr, a_ind, a_val = A.petscmat.getValuesCSR()
    Asp = csr_matrix((a_val, a_ind, a_ptr), A.petscmat.size)

    # Assemble the mass matrix
    M = assemble(m)
    m_ptr, m_ind, m_val = M.petscmat.getValuesCSR()
    Msp = csr_matrix((m_val, m_ind, m_ptr), M.petscmat.size)

    if return_M_inv:
        # Assemble the inverse mass matrix
        m_val_inv = array(m_val)
        m_val_inv[m_val_inv != 0.] = 1. / m_val_inv[m_val_inv != 0.]
        Msp_inv = csr_matrix((m_val_inv, m_ind, m_ptr), M.petscmat.size)
        return Asp, Msp_inv

    return Asp, Msp
