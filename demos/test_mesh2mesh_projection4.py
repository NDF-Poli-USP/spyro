# projection vs at
from firedrake import *
import weakref
import spyro
import time
import sys

distribution_parameters = {"partition": True,
                           "overlap_type": (DistributedMeshOverlapType.VERTEX, 50)}

quad=0 # use interpolation_scheme = 3
if quad:
    m1 = RectangleMesh(200, 200, 1, 1, quadrilateral=True, distribution_parameters=distribution_parameters)
    m2 = RectangleMesh(175, 175, 1, 1, quadrilateral=True, distribution_parameters=distribution_parameters)
else:
    m1 = RectangleMesh(200, 200, 1, 1, quadrilateral=False, distribution_parameters=distribution_parameters)
    m2 = RectangleMesh(175, 175, 1, 1, quadrilateral=False, distribution_parameters=distribution_parameters)

p=2
element = FiniteElement('CG', m1.ufl_cell(), degree=p)
V1 = FunctionSpace(m1, element)
V2 = FunctionSpace(m2, element)

m2._parallel_compatible = {weakref.ref(m1)}

x, y = SpatialCoordinate(m1)
f1 = Function(V1).interpolate(sin(5*x)*cos(5*y))
f1_integrated = assemble(f1*dx)

f2 = Function(V2)

interpolation_scheme = 1
if interpolation_scheme==1: # inhouse projection
    degree=None
    if V2.ufl_element().family() == "Kong-Mulder-Veldhuizen" and P == 2:
        degree = 6
    ti = time.time()
    spyro.mesh_to_mesh_projection(f1, f2, degree=degree)
    tf = time.time()
    print("inhouse projection, time="+str(tf-ti),flush=True)
elif interpolation_scheme==2: # Firedrake projection:
    ti = time.time()
    f2.project(f1)
    tf = time.time()
    print("Firedrake projection, time="+str(tf-ti),flush=True)
elif interpolation_scheme==3: # at scheme
    ti = time.time()
    m = V2.ufl_domain()
    W = VectorFunctionSpace(m, V2.ufl_element())
    X = interpolate(m.coordinates, W)

    a=X.vector().gather()
    #x=[]
    #y=[]
    #for i in range(len(a)):
    #    x.append(a[2*i])
    #    y.append(a[2*i-1])

    x = a[0:len(a):2]
    y = a[1:len(a):2]

    print(len(x),flush=True)
    #print(len(y),flush=True)
    #print(x,flush=True)
    #print(y,flush=True)
    print([x,y],flush=True)
    print(X.dat.data,flush=True)
    
    #print(X.dat.data.shape,flush=True)
    xy = []
    for i in range(len(x)):
        xy.append([x[i], y[i]])
    
    #print(xy)


    #f2.dat.data[:] = f1.at([x[0] , y[0]], dont_raise=True, tolerance=0.001)
    f2.dat.data[:] = f1.at(xy, dont_raise=True, tolerance=0.001) # FIXME and now? how to set the space (on one partition) with the global values?
    tf = time.time()
    print("at scheme, time="+str(tf-ti),flush=True)
else:
    sys.exit("interpolation scheme not implemented!")
    
f2_integrated = assemble(f2*dx)

print(f1_integrated)
print(f2_integrated)

File("f1.pvd").write(f1)
File("f2.pvd").write(f2)
