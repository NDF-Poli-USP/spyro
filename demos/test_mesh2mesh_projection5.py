# projection vs at
from firedrake import *
import weakref
import spyro
import time
import sys

distribution_parameters = {"partition": True,
                           "overlap_type": (DistributedMeshOverlapType.VERTEX, 50)}

m1 = RectangleMesh(200, 200, 1, 1, quadrilateral=True, distribution_parameters=distribution_parameters)
m2 = RectangleMesh(175, 175, 1, 1, quadrilateral=True, distribution_parameters=distribution_parameters)

p=2
element = FiniteElement('CG', m1.ufl_cell(), degree=p)
V1 = FunctionSpace(m1, element)
V2 = FunctionSpace(m2, element)

m2._parallel_compatible = {weakref.ref(m1)}

x, y = SpatialCoordinate(m1)
f1 = Function(V1).interpolate(sin(5*x)*cos(5*y))
f1_integrated = assemble(f1*dx)

f2 = Function(V2)

m = V2.ufl_domain()
W = VectorFunctionSpace(m, V2.ufl_element())
X = interpolate(m.coordinates, W)

a=X.vector().gather() # global vector

x = a[0:len(a):2]
y = a[1:len(a):2]

xy = []
for i in range(len(x)):
    xy.append([x[i], y[i]])

f2.dat.data[:] = f1.at(xy, dont_raise=True, tolerance=0.001) # FIXME how to set the space (on one partition) with the global values?    
f2_integrated = assemble(f2*dx)

print(f1_integrated)
print(f2_integrated)

File("f1.pvd").write(f1)
File("f2.pvd").write(f2)
