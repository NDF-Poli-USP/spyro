from firedrake import *
from movement import *
import numpy as np

def vp(mesh, A, initial_guess=False): # from Spyro run_fwi_acoustic_moving_mesh_circle_case.py

    x, y = SpatialCoordinate(mesh)
    if initial_guess:
        return 1.5 + 0.*x 
    else:
        return 2.5 + tanh(A * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2))) 

def monitor(mesh):# based on Spyro run_fwi_acoustic_moving_mesh_circle_case.py
    a = 2 # scaling parameter
    return ( vp(mesh, 20.0, initial_guess=True) /vp(mesh, 20.0) ) ** a 

def run_moving_mesh(mesh, monitor_function, method, tol, file_monitor, time=1):
    
    P1 = FunctionSpace(mesh, "CG", 1)
    monitor = Function(P1, name="Monitor function")

    mover = MongeAmpereMover(mesh, monitor_function, method=method, rtol=tol)

    mover.move()

    monitor.interpolate(monitor_function(mesh))

    if True:
        File("phi.pvd").write(mover.phi, mover.sigma)
        file_monitor.write(monitor, time=time)

n = 20
tol = 1.0e-03
#method = "relaxation"
method = "quasi_newton"

mesh = UnitSquareMesh(n, n, diagonal="crossed")

if True:
    P1 = FunctionSpace(mesh, "CG", 1)
    vpf = Function(P1, name="cp")
    vpf.interpolate( vp(mesh, 20.0, initial_guess=True) ) # based on Spyro run_fwi_acoustic_moving_mesh_circle_case.py
    #vpf.interpolate( vp(mesh, 20.0) ) # based on Spyro run_fwi_acoustic_moving_mesh_circle_case.py
    File("vp_before_amr.pvd").write(vpf)

file_monitor=File("monitor.pvd")

run_moving_mesh(mesh, monitor, method, tol, file_monitor=file_monitor)

if True:
    P1 = FunctionSpace(mesh, "CG", 1)
    vpf = Function(P1, name="cp")
    vpf.interpolate( vp(mesh, 20.0) )
    File("vp_after.pvd").write(vpf)
