from firedrake import *
from movement import *
import ufl
import spyro
import sys
import time

distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 20)}
n = 40
mesh = RectangleMesh(n, n, 1, 1, diagonal="crossed", distribution_parameters=distribution_parameters)
# ring, from cao etal 2002 p. 131 {{{
den_ring = lambda t, x, y: 1 + t * 5 * exp(-50 * abs( (x-0.5)**2 + (y-0.5)**2 - (0.25)**2 ) )
#}}}
# ring2, from mcrae etal 2018, Eq. 5.3 {{{
den_ring2 = lambda t, x, y: 1 + t * 10 * (1./cosh(200 * ( (x-0.5)**2 + (y-0.5)**2 - (0.25)**2 ) ) )** 2
#}}}
# circle, from Spyro run_fwi_acoustic_moving_mesh_circle_case.py {{{
den_circle = lambda t, x, y: 1.5 / (2.5 + t * tanh(200 * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2))))
#}}}

def monitor_function_ring(mesh, t=1):
    x,y = SpatialCoordinate(mesh)
    return den_ring(t,x,y)

def monitor_function_ring2(mesh, t=1):
    x,y = SpatialCoordinate(mesh)
    return den_ring2(t,x,y)

def monitor_function_circle(mesh, t=1):
    x,y = SpatialCoordinate(mesh)
    return den_circle(t,x,y)

def mask_receivers(mesh):
    x,y = mesh.coordinates
    g = conditional(y > 0.9, 0, (1/0.1)*0.9 - (1/0.1)*y) # 0 apply BC
    g = conditional(y <= 0.8, 1, g)
    return g

def mask_dumb(mesh):
    return Constant(1.)

V = FunctionSpace(mesh, "CG", 3)
f = Function(V)
f.interpolate( Constant(1.0) ) # to print the mesh 

m = Function(V)
m.interpolate(monitor_function_ring2(mesh))
File("monitor.pvd").write(m)

File("mesh_before_amr.pvd").write(f)
if 1:
    p = 1
    tol = 2.0e-03

    ti=time.time()
    if 1:
        #spyro.monge_ampere_solver(mesh, monitor_function_ring, p=p, rtol=tol)
        #spyro.monge_ampere_solver(mesh, monitor_function_ring2, p=p, rtol=tol, mask=None)
        spyro.monge_ampere_solver(mesh, monitor_function_ring2, p=p, rtol=tol, mask=mask_dumb)
        #spyro.monge_ampere_solver(mesh, monitor_function_ring2, p=p, rtol=tol, mask=mask_receivers)
    
    tf=time.time()
    print("time in-house solver="+str(tf-ti))
    File("mesh_after_amr_inhouse_solver.pvd").write(f)
else:
    tol = 2.0e-03
    method = "quasi_newton"
    ti=time.time()
    #mover = MongeAmpereMover(mesh, monitor_function_ring, method=method, rtol=tol)
    mover = MongeAmpereMover(mesh, monitor_function_ring2, method=method, rtol=tol)
    mover.move()
    tf=time.time()
    print("time Joe's solver="+str(tf-ti))
    File("mesh_after_amr_movement_solver.pvd").write(f)

