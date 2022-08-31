from firedrake import *
from scipy.optimize import * 
from movement import *
import spyro
import time
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import h5py
import meshio
#import SeismicMesh
import weakref
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from spyro.io import write_function_to_grid, create_segy
#from ..domains import quadrature, space

# define the model parameters {{{
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV", # Equi or KMV #FIXME it will be removed
    "degree": 2,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    "type": "automatic", # options: automatic (same number of cores for evey processor) or spatial
    #"type": "spatial",  # 
    #"custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    #"num_cores_per_shot": 1 #FIXME this is not used
}

model["mesh"] = {
    #"Lz": 1.5,  # depth in km - always positive
    "Lz": 2.0,  # depth in km - always positive 
    "Lx": 2.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "method": "damping", # damping, pml
    "exponent": 2,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the pml in the z-direction (km) - always positive
    "lx":0.5,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "frequency": 3.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    #"frequency": 5.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    #"frequency": 7.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    "delay": 1.0, # FIXME check this
    "num_sources": 1, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((1, -0.01), (1, -0.01), 1), 
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 100, #FIXME not used (remove it, and update an example script)
    #"receiver_locations": spyro.create_transect((1, -1.99), (1, -1.99), 2), # receivers at the bottom of the domain (z=-1.9 km) 
    #"receiver_locations": spyro.create_transect((1, -0.5), (1, -1.99), 2), # receivers at two positions:z=-0.5 and km z=-1.9 km 
    #"receiver_locations": spyro.create_transect((0.1, -0.5), (1.9, -0.5), 100), # receivers at z=-0.5 km REC1
    "receiver_locations": spyro.create_transect((0.1, -1.99), (1.9, -1.99), 100), # receivers at z=-1.99 km REC2
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 1.0, # Final time for event 
    "tf": 2.0, # Final time for event 
    "dt": 0.00025,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
}
#}}}
# make vp {{{
def _make_vp(mesh, V):
    
    x, y = SpatialCoordinate(mesh)
    
    velocity = conditional(y > -1, 1.5, 4.5)

    vp = Function(V, name="velocity").interpolate(velocity)

    File("vp.pvd").write(vp)
    
    return vp
#}}}
comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 20)}
REF = 1
# run reference model {{{
if REF:
    nx = 100
    ny = 100
    mesh_ref = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], quadrilateral=False, diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_ref.coordinates.dat.data[:, 0] -= 0.0 
    mesh_ref.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]

    # define space of the solution and the velocity field
    element_ref = spyro.domains.space.FE_method(mesh_ref, model["opts"]["method"], model["opts"]["degree"])
    V_ref = FunctionSpace(mesh_ref, element_ref)

    element_DG = spyro.domains.space.FE_method(mesh_ref, "DG", 0) # here, it could be 2 too
    V_DG = FunctionSpace(mesh_ref, element_DG)

    sources = spyro.Sources(model, mesh_ref, V_ref, comm)
    receivers = spyro.Receivers(model, mesh_ref, V_ref, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

    vp_ref = _make_vp(mesh_ref, V_DG)
    #sys.exit("exit")
    print("Starting forward computation of the exact model",flush=True) 
    start = time.time()
    p_ref, p_ref_recv = spyro.solvers.forward(
        model, mesh_ref, comm, vp_ref, sources, wavelet, receivers, output=False
    )
    end = time.time()
    print(round(end - start,2),flush=True)
    File("p_ref.pvd").write(p_ref[-1])

    if 0:
        pr = []
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            pr.append(p_ref_recv[ti][rn])
        plt.title("p at receiver")
        plt.plot(pr,label='p')
        plt.legend()
        plt.savefig('/home/santos/Desktop/p_recv.png')
        plt.close()


    # ok, reset to the original order
    #model["opts"]["degree"] = p
    #print(model["opts"]["degree"])
    #sys.exit("exit")
#}}}
#sys.exit("exit")

# now, prepare to run with different mesh resolutions
FIREMESH = 0
# generate or read a mesh {{{
if FIREMESH: 

    mesh = RectangleMesh(25, 25, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh.coordinates.dat.data[:, 0] -= 0.0 
    mesh.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] 

    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = FunctionSpace(mesh, element)
else:
    #model["mesh"]["meshfile"] = "./meshes/double_material_2_km_x_2_km_h_150m.msh"
    model["mesh"]["meshfile"] = "./meshes/double_material_2_km_x_2_km_h_100m.msh"
    #model["mesh"]["meshfile"] = "./meshes/double_material_2_km_x_2_km_h_50m.msh"
    mesh, V = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)
#}}}

AMR = 1
# adapt the mesh using the exact vp, if requested {{{
if AMR:
    nx = 100
    ny = 100 
    mesh_grid = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_grid.coordinates.dat.data[:, 0] -= 0.0 
    mesh_grid.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] 
    V_grid = FunctionSpace(mesh_grid, "CG", 2)
    V_grid_DG = FunctionSpace(mesh_grid, "DG", 2)
    V_vec_grid = VectorFunctionSpace(mesh_grid, "CG", 2)
  
    vp_grid = _make_vp(mesh_grid, V_grid_DG)
    grad_vp_grid = Function(V_vec_grid)

    if 1:
        u_cts = TrialFunction(V_vec_grid)
        v_cts = TestFunction(V_vec_grid)
        a = inner(v_cts, u_cts)*dx
        L = inner(v_cts, grad(vp_grid))*dx
        _cg = {
            "ksp_type": "cg",
            "pc_type": "bjacobi",
            "pc_sub_type": "ilu",
        }
        _problem = LinearVariationalProblem(a, L, grad_vp_grid, bcs=None)
        _solver = LinearVariationalSolver(_problem, solver_parameters=_cg) 
        _solver.solve()

    Linf_grad_vp_grid = norm(grad_vp_grid, norm_type="L100")
    L2_grad_vp_grid = norm(grad_vp_grid, norm_type="L2")
    L1_grad_vp_grid = norm(grad_vp_grid, norm_type="L1")
    
    Linf_vp_grid = norm(vp_grid, norm_type="L100")
    L2_vp_grid = norm(vp_grid, norm_type="L2")
    L1_vp_grid = norm(vp_grid, norm_type="L1")

    File("grad_vp_grid.pvd").write(grad_vp_grid)
    File("vp_grid.pvd").write(vp_grid)

    # M0: Monitor function based the gradient of Vp (arc-length-based metric)
    M0 = sqrt( 1 + inner( grad_vp_grid, grad_vp_grid ) ) 
    M0_L1 = sqrt( 1 + inner( grad_vp_grid, grad_vp_grid )/L1_grad_vp_grid ) 
    M0_L2 = sqrt( 1 + inner( grad_vp_grid, grad_vp_grid )/L2_grad_vp_grid ) 
    M0_Linf = sqrt( 1 + 0.1*inner( grad_vp_grid, grad_vp_grid )/Linf_grad_vp_grid )  # FIXME remove 0.1
  
    # M1: Monitor function based on the inverse of Vp (a priori estimate based metric)
    M1_L1 = ( L1_vp_grid / vp_grid**2 )
    M1_L2 = ( L2_vp_grid / vp_grid**2 )
    M1_Linf = ( Linf_vp_grid / vp_grid**2 )# use Linf such that M1 >~1
    
    # M2: Monitor function based on the inverse of Vp (a priori estimate based metric)
    M2_L1 = ( L1_vp_grid / vp_grid )
    M2_L2 = ( L2_vp_grid / vp_grid )
    M2_Linf = ( Linf_vp_grid / vp_grid ) # use Linf such that M1 >~1
   
    # Huang type monitor function
    E1 = sqrt( inner( grad_vp_grid, grad_vp_grid ) ) # gradient based estimate
    E2 = vp_grid.dat.data.max() / vp_grid - 1 # a priori error estimate (it starts on 1, so it could be better)
    #E3 = (assemble(vp_grid*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid)) ) / vp_grid - 1 # a priori error estimatie

    #E = E1
    E = E2

    beta = 0.2 # (0, 1) # for E1
    beta = 0.3 # (0, 1) # for E2
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid)) 
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M = 1 + alpha * phi

    # Define the monitor function to be projected onto the adapted mesh
    Mfunc = Function(V_grid)
    Mfunc.interpolate(M)
   
    # smooth the monitor function
    if 1: # {{{
        lamb = 0.01

        u = TrialFunction(V_grid)
        v = TestFunction(V_grid)
        u_n = Function(V_grid)

        m = inner((u - u_n), v) * dx 
        a = lamb * inner(grad(u), grad(v)) * dx # implicit
        F = m + a

        lhs_ = lhs(F)
        rhs_ = rhs(F)

        X = Function(V_grid)
        B = Function(V_grid)
        A = assemble(lhs_)

        params = {"ksp_type": "preonly", "pc_type": "lu"}
        solver = LinearSolver(A, solver_parameters=params)

        u_n.assign(Mfunc)
        nt = 1
        for step in range(nt):
            B = assemble(rhs_, tensor=B)
            solver.solve(X, B)
            u_n.assign(X)

        Mfunc.assign(u_n)
    #}}}

    File("Mfunc.pvd").write(Mfunc)
    #sys.exit("exit")

    def monitor_function(mesh): # here, mesh is the physical doman, i.e., x (=xi+Grad phi)
        # project onto "mesh" that is being adapted (i.e., mesh_x)
        _P1 = FunctionSpace(mesh, "CG", 1) 
        _M = Function(_P1)
        spyro.mesh_to_mesh_projection(Mfunc, _M) #degree=6)
        File("Mfunc_x.pvd").write(_M)
        return _M
 
    V_DG = FunctionSpace(mesh, "DG", 2)
    _vp = _make_vp(mesh, V_DG)
    File("vp_before_amr.pvd").write(_vp)
    #sys.exit("exit")
        
    def mask_receivers(mesh):
        _x,_y = mesh.coordinates
        g = conditional(_y < -1.99, 0, 1) # 0 apply BC
        g = conditional(_y > -0.01, 0, g)
        g = conditional(_x < 0.01, 0, g)
        g = conditional(_x > 3.99, 0, g)
        return g
    
    def mask_dumb(mesh):
        return Constant(1.)

    #fix_boundary_nodes = False
    mask = mask_dumb
    #if FIREMESH==0:
    #    fix_boundary_nodes = True
    #    mask = mask_receivers

    mesh._parallel_compatible = {weakref.ref(mesh_grid)}
    step = spyro.monge_ampere_solver(mesh, monitor_function, p=2, mask=mask) #fix_boundary_nodes=fix_boundary_nodes) 

    _vp = _make_vp(mesh, V_DG)
    File("vp_after_amr.pvd").write(_vp)

#}}}
#sys.exit("exit")

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

V_DG = FunctionSpace(mesh, "DG", 2)
vp = _make_vp(mesh, V_DG)
#vp = _make_vp(mesh, V)

J_total = np.zeros((1))
_, p_recv = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output=True)
        
misfit = spyro.utils.evaluate_misfit(model, p_recv, p_ref_recv) # ds_exact[:ll] - guess
J_total[0] += spyro.utils.compute_functional(model, misfit, vp=vp) # J += residual[ti][rn] ** 2 (and J *= 0.5)
J_total[0] /= spyro.utils.compute_functional(model, p_ref_recv, vp=vp) # J += p_ref_recv[ti][rn] ** 2 (and J *= 0.5)
J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
J_total[0] /= comm.ensemble_comm.size # ensemble parallelism (sources)
if comm.comm.size > 1: 
    J_total[0] /= comm.comm.size # spatial parallelism

E = sqrt(J_total[0]) # relative error as defined in Spyro paper
print(E, flush=True)
    
if 0: # used for the configuration with two receivers, one at z=-0.5 km and another at z=-1.99 km
    pr_ref = []
    pr = []
    nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
    rn = 0
    for ti in range(nt):
        pr_ref.append(p_ref_recv[ti][rn])
        pr.append(p_recv[ti][rn])
    plt.title("p at receiver (z=-0.5 km)")
    plt.plot(pr_ref,label='p ref')
    plt.plot(pr,label='p')
    plt.legend()
    plt.savefig('/home/santos/Desktop/p_recv_1.png')
    plt.close()
    
    pr_ref = []
    pr = []
    nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
    rn = 1
    for ti in range(nt):
        pr_ref.append(p_ref_recv[ti][rn])
        pr.append(p_recv[ti][rn])
    plt.title("p at receiver (z=-1.99 km)")
    plt.plot(pr_ref,label='p ref')
    plt.plot(pr,label='p')
    plt.legend()
    plt.savefig('/home/santos/Desktop/p_recv_2.png')
    plt.close()



  
