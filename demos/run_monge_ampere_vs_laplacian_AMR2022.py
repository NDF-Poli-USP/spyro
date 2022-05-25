from firedrake import *
from movement import *
import numpy as np
import spyro
import sys
import time

# ring, from cao etal 2002 p. 131 {{{
den_ring = lambda t, x, y: 1 + t * 5 * exp(-50 * abs( (x-0.5)**2 + (y-0.5)**2 - (0.25)**2 ) )
#}}}
# circle, from Spyro run_fwi_acoustic_moving_mesh_circle_case.py {{{
den_circle = lambda t, x, y: 2.5 + t * tanh(20 * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2)))
#}}}
# circle 2, from Spyro run_fwi_acoustic_moving_mesh_circle_case.py {{{
den_circle2 = lambda t, x, y: 1.5 / (2.5 + t * tanh(20 * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2))))
#}}}
# circle 3, from Spyro run_fwi_acoustic_moving_mesh_circle_case.py {{{
den_circle3 = lambda t, x, y: 1.5 / (2.5 + t * tanh(200 * (0.125 - sqrt(( x - 0.5) ** 2 + (y - 0.5) ** 2))))
#}}}
# from Marmousi model {{{
model = {}
model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG",  # Equi or KMV
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    #"type": "automatic",
    "type": "spatial",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "None",
    "initmodel": "None",
    "truemodel": "None",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "gaussian-taper",  # none, gaussian-taper, or alid
    "lz": 0.9,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.10, 5.0)], # Z and X # with water layer (for wave propagation comparison)
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.05, 0.0), (-0.05, 17.0), 100),
}
comm = spyro.utils.mpi_init(model)

model["mesh"]["meshfile"] = "meshes/marmousi_elastic_with_water_layer_adapted_using_vp_smoothed_sigma=300.msh"

mesh_0, _ = spyro.io.read_mesh(model, comm) # the monitor is computed over this mesh
_V = FunctionSpace(mesh_0, "CG", 4) # mesh_0 is fixed ATTENTION check the order in model

model_path="./velocity_models/elastic-marmousi-model/model/"
model["mesh"]["truemodel"] = model_path+"MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"# m/s
_vp0 = spyro.io.interpolate(model, mesh_0, _V, guess=False) # vp0 is computed over mesh_0

model["mesh"]["truemodel"] = model_path+"MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s
_vp1 = spyro.io.interpolate(model, mesh_0, _V, guess=False) # vp1 is computed over mesh_0

_M = Function(_V) # create monitor function computed over mesh_0
_alpha=1.0
_M.dat.data[:] = (_vp0.dat.data[:] / _vp1.dat.data[:])**_alpha # over mesh_0
File("monitor_mongeampere_mesh_0.pvd").write(_M)

if False:
    model["mesh"]["meshfile"] = "meshes/marmousi_elastic_with_water_layer_adapted_using_vp.msh"
    mesh_original, V_original = spyro.io.read_mesh(model, comm) # the monitor is computed over this mesh
    vp_original=spyro.io.interpolate(model, mesh_original, V_original, guess=False)
    File("mesh_original.pvd").write(vp_original)
    sys.exit("exit")
#}}}

def monitor_function_ring(mesh, t=1):
    x,y = SpatialCoordinate(mesh)
    return den_ring(t,x,y) 

def monitor_function_circle(mesh, t=1):
    x,y = SpatialCoordinate(mesh)
    return den_circle(t,x,y)

def monitor_function_circle2(mesh, t=1):
    x,y = SpatialCoordinate(mesh)
    return den_circle2(t,x,y)

def monitor_function_circle3(mesh, t=1):
    x,y = SpatialCoordinate(mesh)
    return den_circle3(t,x,y)

def monitor_function_marmousi(mesh, t=1):
    # project onto "mesh" that is being adapted
    P1 = FunctionSpace(mesh, "CG", 1)
    M = Function(P1)
    M.project(_M) # _M is defined over mesh_0
    return M

def run_moving_mesh_mongeampere(mesh, monitor_function, method, tol, file_monitor):#{{{
    
    P1 = FunctionSpace(mesh, "CG", 1)
    monitor = Function(P1, name="Monitor function")

    mover = MongeAmpereMover(mesh, monitor_function, method=method, rtol=tol)

    start = time.time()
    mover.move()
    end = time.time()
    print("Monge-Ampere time="+str(end-start))

    monitor.interpolate(monitor_function(mesh))

    if True:
        File("phi.pvd").write(mover.phi, mover.sigma)
        file_monitor.write(monitor, time=1)
#}}}
def run_moving_mesh_laplacian(mesh, monitor_function, dt, num_timesteps, file_monitor): # {{{
    
    P1 = FunctionSpace(mesh, "CG", 1)
    monitor = Function(P1, name="Monitor function")

    mover = LaplacianSmoother(mesh, timestep=dt, )
    
    def update_forcings(t):
        dummy = Function(P1).interpolate(monitor_function(mesh, t=t))
        mover.f.dat.data[:]=dummy.dat.data[:]
        sys.exit("here")

    time = 0.0
    for j in range(num_timesteps):
        print("time="+str(time))
        mover.move(time, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3, 4])
        time += dt

    monitor.interpolate(monitor_function(mesh))

    if True:
        file_monitor.write(monitor, time=time)
#}}}
def run_moving_mesh_laplacian_2(mesh, den, dt, num_timesteps, file_monitor): #{{{
    V = FunctionSpace(mesh, "CG", 1)
    G = VectorFunctionSpace(mesh,"CG", 1)
    coord_space = mesh.coordinates.function_space()

    v = TestFunction(V)
    u = TrialFunction(V)
    sol = Function(V)
    rho = Function(V)
    
    xm = Function(mesh.coordinates, name="Physical coordinates")
    xi = Function(mesh.coordinates, name="Computational coordinates")

    uvec = TrialFunction(coord_space)
    vvec = TestFunction(coord_space)
    vel = Function(coord_space)

    dx = Measure('dx', mesh)
    t = 0
    start = time.time()
    for i in range(num_timesteps):
        print("Time = "+str(t))
        x,y = SpatialCoordinate(mesh)
        rho_t = den(t,x,y) / assemble(den(t,x,y) * dx)
        rho_tp1 = den(t+dt,x,y) / assemble(den(t+dt,x,y) * dx)
        a = inner( rho_t * grad(u), grad(v)) * dx
        L = inner((rho_tp1-rho_t)/dt, v) * dx
        bcs = DirichletBC(V, 0, (1,2,3,4))
        problem = LinearVariationalProblem(a, L, sol, bcs=bcs)
        solver = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        #mesh.coordinates.assign(xi)
        solver.solve()
    
        if True:
            rho.interpolate(rho_t)
            file_monitor.write(rho, time=i)

        # compute the velocity
        #vel = Function(G).interpolate( grad(sol) )
        a = inner(uvec, vvec) * dx
        L = inner(grad(sol), vvec) * dx
        bcs = DirichletBC(coord_space, 0, (1,2,3,4))
        problem = LinearVariationalProblem(a, L, vel, bcs=bcs)
        solver = LinearVariationalSolver(problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        solver.solve()

        # move the vertices
        xm.dat.data_with_halos[:,0] += vel.dat.data_with_halos[:,0] * dt
        xm.dat.data_with_halos[:,1] += vel.dat.data_with_halos[:,1] * dt
        mesh.coordinates.assign(xm)

        t += dt

    end = time.time()
    print("Laplacian time="+str(end-start))
#}}}

# data for Monge-Ampere solver 
n = 20
tol = 1.0e-03
#method = "relaxation"
method = "quasi_newton"

# data for Laplacian solver
num_timesteps = 10
dt = 1.0/num_timesteps

# prepare the meshes
mesh_mongeampere = UnitSquareMesh(n, n, diagonal="crossed")
mesh_laplacian = UnitSquareMesh(n, n, diagonal="crossed")

# prepare the files
file_monitor_mongeampere = File("monitor_mongeampere.pvd")
file_monitor_laplacian   = File("monitor_laplacian.pvd")

monitor_type = 4
# call the moving mesh solvers
if monitor_type==1: # ring (analytic)
    run_moving_mesh_mongeampere(mesh_mongeampere, monitor_function_ring, method, tol, file_monitor_mongeampere)
    run_moving_mesh_laplacian_2(mesh_laplacian, den_ring, dt, num_timesteps, file_monitor_laplacian)
elif monitor_type==2: # circle (analytic)
    run_moving_mesh_mongeampere(mesh_mongeampere, monitor_function_circle, method, tol, file_monitor_mongeampere)
    run_moving_mesh_laplacian_2(mesh_laplacian, den_circle, dt, num_timesteps, file_monitor_laplacian)
elif monitor_type==3: # circle 2 (analytic)
    run_moving_mesh_mongeampere(mesh_mongeampere, monitor_function_circle2, method, tol, file_monitor_mongeampere)
    run_moving_mesh_laplacian_2(mesh_laplacian, den_circle2, dt, num_timesteps, file_monitor_laplacian)
elif monitor_type==4: # circle 3 (analytic)
    run_moving_mesh_mongeampere(mesh_mongeampere, monitor_function_circle3, method, tol, file_monitor_mongeampere)
    run_moving_mesh_laplacian_2(mesh_laplacian, den_circle3, dt, num_timesteps, file_monitor_laplacian)
elif monitor_type==5: # marmousi (discrete)
    model["mesh"]["meshfile"] = "meshes/marmousi_elastic_with_water_layer_adapted_using_vp_smoothed_sigma=300.msh"
    model["mesh"]["truemodel"] = model_path+"MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"# m/s
    # read meshes for marmousi model
    mesh_mongeampere, _ = spyro.io.read_mesh(model, comm) # the mesh that will be adapted following the discrete monitor

    run_moving_mesh_mongeampere(mesh_mongeampere, monitor_function_marmousi, method, tol, file_monitor_mongeampere)
    #run_moving_mesh_laplacian_2(mesh_laplacian, den_marmousi, dt, num_timesteps, file_monitor_laplacian)
else:
    pass        

