import scipy
import numpy as np
from scipy.sparse.linalg import inv

import firedrake as fd
from firedrake import dot, grad
import finat


def estimate_timestep(mesh, V, c):
    """Estimate the maximum stable timestep based on the spectral radius"""

    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    quad_rule = finat.quadrature.make_quadrature(
        V.finat_element.cell, V.ufl_element().degree(), "KMV"
    )
    dxlump = fd.dx(rule=quad_rule)
    A = fd.assemble(u * v * dxlump)
    ai, aj, av = A.petscmat.getValuesCSR()
    Asp = scipy.sparse.csr_matrix((av, aj, ai))
    Asp_inv = inv(Asp)

    K = fd.assemble(c * c * dot(grad(u), grad(v)) * dxlump)
    ai, aj, av = K.petscmat.getValuesCSR()
    Ksp = scipy.sparse.csr_matrix((av, aj, ai))

    # operator
    Lsp = Asp_inv.multiply(Ksp)

    max_eigval = scipy.sparse.linalg.eigs(
        Lsp, k=1, which="LM", return_eigenvectors=False
    )
    # print(max_eigval)
    max_dt = np.float(2 / np.sqrt(max_eigval[0]))
    print(
        f"Maximum stable timestep should be about: {np.float(2 / np.sqrt(max_eigval[0]))} seconds"
    )
    return max_dt
