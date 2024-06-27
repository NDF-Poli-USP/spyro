import finat
from firedrake import *
import numpy as np


def isDiag(M):
    i, j = np.nonzero(M)
    return np.all(i == j)

degree = 4
mesh = RectangleMesh(3, 2, 2.0, 1.0)
element = FiniteElement(  # noqa: F405
    "KMV", mesh.ufl_cell(), degree=degree, variant="KMV"
)
V = FunctionSpace(mesh, element)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

u = TrialFunction(V)
v = TestFunction(V)

form = u*v*dx(scheme=quad_rule)
A = assemble(form)
M = A.M.values
Mdiag = M.diagonal()

print(f"Matrix is diagonal:{isDiag(M)}")
np.save("new_diag", Mdiag)
print("END")