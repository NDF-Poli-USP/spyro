import scipy
import numpy as np

import firedrake as fd
from firedrake import dot, grad
import finat


def estimate_timestep(mesh, V, c, estimate_max_eigenvalue=True):
    """Estimate the maximum stable timestep based on the spectral radius
    using optionally the Gershgorin Circle Theorem to estimate the
    maximum generalized eigenvalue. Otherwise computes the maximum
    generalized eigenvalue exactly

    ONLY WORKS WITH KMV ELEMENTS
    """

    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    quad_rule = finat.quadrature.make_quadrature(
        V.finat_element.cell, V.ufl_element().degree(), "KMV"
    )
    dxlump = fd.dx(rule=quad_rule)
    A = fd.assemble(u * v * dxlump)
    ai, aj, av = A.petscmat.getValuesCSR()
    av_inv = []
    for value in av:
        if value == 0:
            av_inv.append(0.0)
        else:
            av_inv.append(1 / value)
    Asp = scipy.sparse.csr_matrix((av, aj, ai))
    Asp_inv = scipy.sparse.csr_matrix((av_inv, aj, ai))

    K = fd.assemble(c*c*dot(grad(u), grad(v)) * dxlump)
    ai, aj, av = K.petscmat.getValuesCSR()
    Ksp = scipy.sparse.csr_matrix((av, aj, ai))

    # operator
    Lsp = Asp_inv.multiply(Ksp)
    if estimate_max_eigenvalue:
        # absolute maximum of diagonals
        max_eigval = np.amax(np.abs(Lsp.diagonal()))
    else:
        print(
            "Computing exact eigenvalues is extremely computationally demanding!",
            flush=True,
        )
        max_eigval = scipy.sparse.linalg.eigs(
            Ksp, M=Asp, k=1, which="LM", return_eigenvectors=False
        )[0]

    # print(max_eigval)
    max_dt = np.float(2 / np.sqrt(max_eigval))
    print(
        f"Maximum stable timestep should be about: {np.float(2 / np.sqrt(max_eigval))} seconds",
        flush=True,
    )
    return max_dt
