import firedrake as fire
import FIAT
import finat


# Function space for the problem
mesh = fire.UnitTetrahedronMesh()
V = fire.FunctionSpace(mesh, 'KMV', 3)
u, v = fire.TrialFunction(V), fire.TestFunction(V)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

c = fire.Constant(1.5)
a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * fire.dx(scheme=quad_rule)
A = fire.assemble(a)
m = fire.inner(u, v) * fire.dx(scheme=quad_rule)
M = fire.assemble(m)
m_val = M.M.handle
print(m_val)

import ipdb
ipdb.set_trace()


# print("\nSolving Eigenvalue Problem")

# if method[:-3] == 'KRYLOVSCH':
#     # Modal solver
#     Lsp = self.solve_eigenproblem(A, M, method, shift=1e-8)

# else:
#     # Assembling the matrices for Scipy solvers
#     m_ptr, m_ind, m_val = M.petscmat.getValuesCSR()
#     Msp = ss.csr_matrix((m_val, m_ind, m_ptr), M.petscmat.size)
#     a_ptr, a_ind, a_val = A.petscmat.getValuesCSR()
#     Asp = ss.csr_matrix((a_val, a_ind, a_ptr), A.petscmat.size)

#     # Modal solver
#     Lsp = self.solve_eigenproblem(Asp, Msp, method, shift=1e-8)

# if monitor:
#     for n_eig, eigval in enumerate(np.unique(Lsp)):
#         f_eig = np.sqrt(abs(eigval)) / (2 * np.pi)
#         print(f"Frequency {n_eig} (Hz): {f_eig:.5f}")
