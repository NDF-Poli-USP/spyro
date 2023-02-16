from firedrake import *
import weakref

m1 = RectangleMesh(15, 15, 1, 1,diagonal="crossed")
m2 = RectangleMesh(10, 10, 1, 1)

# to run project in parallel
m2._parallel_compatible = {weakref.ref(m1)}

# the projection works for P1, P3, P4, P5 (max degree of KMV elements) but fails for P2
P = 2
V1 = FunctionSpace(m1, 'KMV', P)
V2 = FunctionSpace(m2, 'KMV', P)

x, y = SpatialCoordinate(m1)
#f1 = Function(V1).interpolate(sin(5*x)*cos(5*y))
f1 = Function(V1).interpolate(Constant(1))

f2 = Function(V2)
#f2 = Projector(f1, V2).project()
f2.project(f1)

# f2 should be numerically equal to f1
File("f1.pvd").write(f1)
File("f2.pvd").write(f2)

f1_integrated = assemble(f1*dx)
f2_integrated = assemble(f2*dx)
print("<f1> = " + str(f1_integrated) )
print("<f2> = " + str(f2_integrated) )
