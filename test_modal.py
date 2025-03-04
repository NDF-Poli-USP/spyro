from firedrake import *
from slepc4py import SLEPc

# Create mesh and function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# Define the bilinear and linear forms for the scalar wave equation
u = TrialFunction(V)
v = TestFunction(V)
c = Constant(1.0)  # Wave speed
a = c**2 * inner(grad(u), grad(v)) * dx
m = u * v * dx

# Assemble the matrices
A = assemble(a)
M = assemble(m)

# Set up the SLEPc eigensolver
eigensolver = SLEPc.EPS().create()
A_petsc = A.M.handle
M_petsc = M.M.handle
eigensolver.setOperators(A_petsc, M_petsc)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eigensolver.setFromOptions()

# Solve the eigenvalue problem
eigensolver.solve()

# Extract the eigenvalues
nconv = eigensolver.getConverged()
eigenvalues = []
for i in range(nconv):
    eigenvalue = eigensolver.getEigenvalue(i)
    eigenvalues.append(eigenvalue)

print("Eigenvalues:", eigenvalues)


https://github.com/firedrakeproject/slepc
https://github.com/firedrakeproject/firedrake/blob/master/docs/source/install.rst
https://slepc.upv.es/documentation/instal.htm
https://pypi.org/project/slepc4py/