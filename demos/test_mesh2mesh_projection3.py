# quads
from firedrake import *
import weakref
import spyro

distribution_parameters = {"partition": True,
                           "overlap_type": (DistributedMeshOverlapType.VERTEX, 20)}


m1 = RectangleMesh(15, 15, 1, 1, quadrilateral=True, distribution_parameters=distribution_parameters)
m2 = RectangleMesh(10, 10, 1, 1, quadrilateral=True, distribution_parameters=distribution_parameters)

p=2
element = FiniteElement('CG', m1.ufl_cell(), degree=p, variant='spectral')
V1 = FunctionSpace(m1, element)
V2 = FunctionSpace(m2, element)

m2._parallel_compatible = {weakref.ref(m1)}

#P = 2
#V1 = FunctionSpace(m1, 'CG', P, variant='spectral')
#V2 = FunctionSpace(m2, 'CG', P, variant='spectral')

x, y = SpatialCoordinate(m1)
f1 = Function(V1).interpolate(sin(5*x)*cos(5*y))
f1_integrated = assemble(f1*dx)

use_inhouse_projection = 0
f2 = Function(V2)
if use_inhouse_projection:
    print("using an in-house projection")
    degree=None
    if V2.ufl_element().family() == "Kong-Mulder-Veldhuizen" and P == 2:
        degree = 6
    spyro.mesh_to_mesh_projection(f1, f2, degree=degree)
else:
    f2.project(f1)

f2_integrated = assemble(f2*dx)

print(f1_integrated)
print(f2_integrated)

File("f1.pvd").write(f1)
File("f2.pvd").write(f2)
