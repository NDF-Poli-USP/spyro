import numpy as np
from scipy.sparse import csc_matrix

def update_bfgs(Binv, s, y):
    """update Hessian according to BFGS method using Sherman-Morrison formula"""

    s = csc_matrix(s).T
    y = csc_matrix(y).T
    Binvp = csc_matrix(Binv)
    c1 = np.float((s.T @ y + y.T @ Binv @ y) / (s.T @ y) ** 2)
    c2 = np.float(1 / (s.T @ y).toarray())
    Binvp += c1 * s @ s.T - c2 * (Binv @ y @ s.T + s @ y.T @ Binv) 

    return Binvp

def sparse_csc_matrix_to_cplex(A):
    """Transform csc_matrix to list format that cplex likes"""

    indices = A.indices
    data = A.data
    idx = A.indptr
    H = []
    for i in range(idx.size - 1):
        since, until = idx[i], idx[i+1]
        H.append([indices[since:until].tolist(), data[since:until].tolist()])

    return H

