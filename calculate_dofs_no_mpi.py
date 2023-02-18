from firedrake import *
import numpy as np

scales = [1,2]
for scale in scales:
    mesh = Mesh("meshes/homogeneous_3D_scale"+str(scale)+".msh")
    element = FiniteElement('KMV', mesh.ufl_cell(), degree=3, variant = 'KMV')

    V = FunctionSpace(mesh, element)
    V.dim()

    print(V.dim(), flush = True)
    u = Function(V)
    udat = u.dat.data[:]
    print(np.shape(udat), flush = True)
