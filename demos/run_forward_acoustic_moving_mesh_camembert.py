# run_forward_acoustic_moving_mesh_camembert.py
from firedrake import *
from scipy.optimize import * 
import spyro
import time
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import h5py
import meshio
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
import platform

# define the model parameters {{{
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV or spectral
    "quadrature": "KMV", # Equi or KMV #FIXME it will be removed
    "degree": 2,  # p order
    #"degree": 3,  # p order
    #"degree": 4,  # p order
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
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    #"outer_bc": "none", # FIXME testing it none or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "method": "damping", # damping, pml
    "exponent": 2,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the pml in the z-direction (km) - always positive
    "lx":0.5,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

# Receiver locations
rec1=spyro.create_transect((0.15, 0.85), (0.85, 0.85), 20) # receivers at the top of the domain (REC1)
rec2=spyro.create_transect((0.15, 0.15), (0.85, 0.15), 20) # receivers at the bottom of the domain (REC2)
rec = np.concatenate((rec1,rec2))

#print(spyro.create_transect((0.1, 0.9), (0.9, 0.9), 4))
#print(rec)
#sys.exit("exit")

model["acquisition"] = {
    "source_type": "Ricker",
    #"frequency": 6.0, # freq peak = 6 Hz, max freq = 15 Hz (see Jaquet's  Thesis) 
    #"frequency": 10.0, # freq peak = 10 Hz, max freq = ? Hz 
    "frequency": 15.0, # freq peak = 15 Hz, max freq = ? Hz 
    "delay": 1.0, # FIXME check this
    "num_sources": 4, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.1, 0.9), (0.9, 0.9), 4), # waterbottom at z=-0.45 km # out of domain, only to run with some source
    "amplitude": 1.0, #FIXME check this
    "num_receivers": len(rec), #FIXME not used (remove it, and update an example script)
    "receiver_locations": rec, 
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.8, # Final time for event  
    #"dt": 0.00025, # timestep size  
    "dt": 0.00025/6,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10000000,  # how frequently to save solution to RAM
}
#}}}
# make vp {{{
def _make_vp(V):
  
    m = V.ufl_domain()
    W = VectorFunctionSpace(m, V.ufl_element())
    coords = interpolate(m.coordinates, W)
    xq, zq = coords.dat.data[:, 0], coords.dat.data[:, 1]

    x, y = SpatialCoordinate(m)

    v0 = 2.5 # background vp (km/s)
    dv = 0.3*v0 # 30% of perturbation
    
    vp_cond = conditional((x-0.5)**2 + (y-0.5)**2 <= 0.250**2, v0+dv, v0)
    vp = Function(V).interpolate(vp_cond)

    File("exact_vp.pvd").write(vp)

    return vp
#}}}

# controls
FIREMESH = 1    # keep it 1
AMR = 1         # should adapt the mesh?
GUESS = 1       # if 1, run the guess model; otherwise (=0), read results
REF = 0         # if 1, run the reference model; otherwise (=0), read results
QUAD = 1        # if 1, run with quadrilateral elements; otherwise (=0), run with triangles
DG_VP = 1       # if 1, vp is defined on a Discontinuous space (L2 instead of an H1 space)
CONST_VP = 0    # if 1, run with a uniform vp = 2 km/s (it is employed to check convergence rate and wheter adapted mesh introduces errors)
PLOT_AT_REC = 1 # if 1, plot the pressure over time at one receiver
MFUNC = 1       # if 1, M1; if 2, M2; if 3, M3 (default is M3, therefore MFUNC = 3)
print_vtk = False
use_Neumann_BC_as_source = False 

comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 60)} # FIXME if "at" will be the default scheme, then we could remove overlap

# set the reference file name
file_name = "p_ref_p5_recv_freq_"+str(model["acquisition"]["frequency"]) # with P=5
if platform.node()=='recruta':
    path = ""
    sys.exit("path not defined")
else:
    #path = "/share/tdsantos/shots/acoustic_forward_camembert_6Hz/" 
    #path = "/share/tdsantos/shots/acoustic_forward_camembert_10Hz/" 
    #path = "/share/tdsantos/shots/acoustic_forward_camembert_15Hz/" 
    path = "/share/tdsantos/shots/acoustic_forward_camembert_15Hz_4_sources/" 

# run reference model {{{
if REF:
    sys.exit("exit")
    _nx = 100  # nx=100  => dx = dz = 10 m
    _ny = math.ceil( _nx*model["mesh"]["Lz"]/model["mesh"]["Lx"] ) # nx * Lz/Lx, Delta x = Delta z

    # here, we do not need overlaping vertices
    distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}

    mesh_ref = RectangleMesh(_nx, _ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)

    # for the exact model, use a higher-order element
    model["opts"]["degree"] = 5 # it was 5 before
    element = spyro.domains.space.FE_method(mesh_ref, model["opts"]["method"], model["opts"]["degree"])
    V_ref = FunctionSpace(mesh_ref, element) 

    if DG_VP==1:# DG space is better to represent data, but it is not currently possible to use it during FWI
        element_DG = spyro.domains.space.FE_method(mesh_ref, "DG", 0) # model["opts"]["degree"]) 
        V_ref_DG = FunctionSpace(mesh_ref, element_DG)
        vp_ref = _make_vp(V_ref_DG) 
    else:
        vp_ref = _make_vp(V_ref)
    
    #sys.exit("exit")

    sources = spyro.Sources(model, mesh_ref, V_ref, comm)
    receivers = spyro.Receivers(model, mesh_ref, V_ref, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])
    
    if CONST_VP:
        vp_ref.dat.data_with_halos[:] = 2.0 

    #sys.exit("exit")
    print("Starting forward computation of the exact model",flush=True) 
    start = time.time()
    p_ref, p_ref_recv = spyro.solvers.forward(
        model, mesh_ref, comm, vp_ref, sources, wavelet, receivers, output=print_vtk, use_Neumann_BC_as_source=use_Neumann_BC_as_source
    )
    end = time.time()
    print(round(end - start,2),flush=True)
    #sys.exit("exit") # FIXME
    
    if 0:
        File("p_ref.pvd").write(p_ref[-1])

    spyro.io.save_shots(model, comm, p_ref_recv, file_name=path+file_name) 

    _h = round(1000*model["mesh"]["Lx"]/_nx)
    if comm.ensemble_comm.rank == 1:
        print("Reference model:", flush=True)
        print("p = " + str(model["opts"]["degree"]))
        print("h = " + str(_h) + " m")
        print("DOF = " + str(V_ref.dof_count), flush=True)
        print("Nelem = " + str(mesh_ref.num_cells()), flush=True) 

    sys.exit("Reference model finished!")

elif REF==0: 
    print("reading reference model",flush=True)
    p_ref_recv = spyro.io.load_shots(model, comm, file_name=path+file_name)
#}}}

# now, prepare to run with different mesh resolutions
def switch(iii): # switch definition (iii comes from sys.argv) {{{
    if iii == 0:
        return 80  # nx=80  => dx = dz = 12.5 m 
    elif iii == 1:
        return 50  # nx=50  => dx = dz = 20 m
    elif iii == 2:
        return 40  # nx=40  => dx = dz = 25 m
    elif iii == 3:
        return 25  # nx=25  => dx = dz = 40 m
    elif iii == 4:
        return 20  # nx=20  => dx = dz = 50 m
    elif iii == 5:
        return 14  # nx=14  => dx = dz = 71.43 m
    elif iii == 6:
        return 10  # nx=10  => dx = dz = 100 m
    else:
        sys.exit("iii not found! Exiting...")
#}}}

# read or define degree and nx/ny
if len(sys.argv)==6:
    ppp=int(sys.argv[1])
    iii=int(sys.argv[2])
    AMR=int(sys.argv[3])
    DG_VP=int(sys.argv[4])
    QUAD=int(sys.argv[5])
    print("ok, it worked")
    if QUAD==1 and ppp!=4:
        sys.exit("QUAD=1, but degree not equal to 4. Skipping run...")
else:
    ppp=4
    iii=1

if QUAD==1:
    model["opts"]["method"] = "CG"
    model["opts"]["quadrature"] = "GLL"
    model["opts"]["degree"] = 4
    #model["opts"]["degree"] = 8

nx = switch(iii)
ny = math.ceil( nx*model["mesh"]["Lz"]/model["mesh"]["Lx"] ) # nx * Lz/Lx, Delta x = Delta z
model["opts"]["degree"] = ppp

print("\n Degree = " + str(ppp) + ", nx = " + str(nx) + ", AMR = " + str(AMR) +  ", DG_VP = " + str(DG_VP) + ", QUAD = " + str(QUAD) + "\n", flush=True)
#sys.exit("exit")

# generate or read a mesh, and create space V {{{
if FIREMESH: 
    if QUAD==0:
        mesh = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                                distribution_parameters=distribution_parameters)
    elif QUAD==1:
        mesh = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], quadrilateral=True, comm=comm.comm,
                                distribution_parameters=distribution_parameters)
    
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = FunctionSpace(mesh, element)
else:
    sys.exit("Running with unstructured mesh? make sure you set up the path correctly")
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_150m.msh"
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_100m.msh"
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_50m.msh"
    mesh, V = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)
#}}}

#print("DOF = " + str(V.dof_count), flush=True)
#print("Nelem = " + str(mesh.num_cells()), flush=True) 
#sys.exit("exit")

# adapt the mesh using the exact vp, if requested {{{
if AMR==1 and GUESS==1:
    # This mesh-grid is used to compute the monitor function
    # Alternatively, a point cloud scheme could be used instead (see Jaquet's thesis)
    _nx = 200
    _ny = math.ceil( _nx*model["mesh"]["Lz"]/model["mesh"]["Lx"] ) # nx * Lz/Lx, Delta x = Delta z
    mesh_grid = RectangleMesh(_nx, _ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    
    V_grid = FunctionSpace(mesh_grid, "CG", 2) 
    #V_grid_DG = FunctionSpace(mesh_grid, "DG", 2) # DG will be similar to CG
    V_vec_grid = VectorFunctionSpace(mesh_grid, "CG", 2)
  
    vp_grid = _make_vp(V_grid)
    grad_vp_grid = Function(V_vec_grid)

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

    File("grad_vp_grid.pvd").write(grad_vp_grid)
    File("vp_grid.pvd").write(vp_grid)

    # Huang type monitor function
    E1 = sqrt( inner( grad_vp_grid, grad_vp_grid ) ) # gradient based estimate
    E2 = vp_grid.vector().gather().max() / vp_grid - 1 # a priori error estimate (it starts on 1, so it is better)

    E = E1
    #beta = 0.5 # (0, 1) # for E2 + smooth
    #beta = 0.10 # (0, 1) # for E2 w/n smooth
    beta = 0.20 # (0, 1) # for E2 w/n smooth
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M1 = 1 + alpha * phi
   
    E = E2
    beta = 0.5 # (0, 1) # for E2 + smooth
    #beta = 0.3 # (0, 1) # for E2 w/n smooth
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M2 = 1 + alpha * phi

    if MFUNC==1:
        M = M1
    elif MFUNC==2:
        M = M2
    elif MFUNC==3:
        M = max_value(M1,M2)
    else:
        sys.exit("MFUNC not defined!")

    # Define the monitor function to be projected onto the adapted mesh
    Mfunc = Function(V_grid)
    Mfunc.interpolate(M)

    # smooth the monitor function
    if 1: # {{{
        lamb = 0.001

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
        _P1 = FunctionSpace(mesh, "CG", 1) # P1 works better here (even p>1 does not improve when coarse meshes are employed) 
        _M = Function(_P1)
        
        #FIXME use projection for spatial parallel until we solve the issue with "at"
        #spyro.mesh_to_mesh_projection(Mfunc, _M, degree=6)
        
        # it works for serial so far
        _m = _P1.ufl_domain() # quads
        _W = VectorFunctionSpace(_m, _P1.ufl_element())
        _X = interpolate(_m.coordinates, _W)
        _M.dat.data[:] = Mfunc.at(_X.dat.data_ro, dont_raise=True, tolerance=0.001)
        
        File("Mfunc_x.pvd").write(_M)
        return _M
   
    if DG_VP==1:
        element_DG = spyro.domains.space.FE_method(mesh, "DG", 0) #FIXME model["opts"]["degree"])
        V_DG = FunctionSpace(mesh, element_DG)
        _vp = _make_vp(V_DG)
    else: 
        _vp = _make_vp(V) # V is the original space of mesh

    File("vp_before_amr.pvd").write(_vp)
    #sys.exit("exit")
        
    def mask_receivers(mesh):
        _x,_y = mesh.coordinates
        g = conditional(_y < -1.99, 0, 1) # 0 apply BC
        g = conditional(_y > -0.01, 0, g)
        g = conditional(_x < 0.01, 0, g)
        g = conditional(_x > 3.99, 0, g)
        return g
    
    def mask_dummy(mesh):
        return Constant(1.)

    #fix_boundary_nodes = False
    mask = mask_dummy
    #if FIREMESH==0:
    #    fix_boundary_nodes = True
    #    mask = mask_receivers

    mesh._parallel_compatible = {weakref.ref(mesh_grid)}
    start = time.time()
    step = spyro.monge_ampere_solver(mesh, monitor_function, p=2, mask=mask) #fix_boundary_nodes=fix_boundary_nodes) 
    
    # since coordinates were chanegd, clear spatial index
    mesh.clear_spatial_index()

    end = time.time()
    print(round(end - start,2),flush=True)
    
    if DG_VP==1:
        element_DG = spyro.domains.space.FE_method(mesh, "DG", 0) #FIXME model["opts"]["degree"])
        V_DG = FunctionSpace(mesh, element_DG)
        _vp = _make_vp(V_DG)
    else: 
        _vp = _make_vp(V) # V is the original space of mesh
    File("vp_after_amr.pvd").write(_vp)
#}}}
#sys.exit("exit")

# set the file name
h = round(1000*model["mesh"]["Lx"]/nx)
if MFUNC==3:
    sys.exit("exit")
    if QUAD==1:
        file_name = "p_recv_QUAD_AMR_" + str(AMR) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])
    else:
        file_name = "p_recv_AMR_" + str(AMR) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])
else:
    if QUAD==1:
        file_name = "p_recv_QUAD_AMR_" + str(AMR) + "_MFUNC_" + str(MFUNC) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])
    else:
        file_name = "p_recv_AMR_" + str(AMR) + "_MFUNC_" + str(MFUNC) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])


# run the guess model with a given mesh {{{
if GUESS==1:
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

    if DG_VP==1:
        element_DG = spyro.domains.space.FE_method(mesh, "DG", 0) #FIXME model["opts"]["degree"])
        V_DG = FunctionSpace(mesh, element_DG)
        vp = _make_vp(V_DG)
    else: 
        vp = _make_vp(V) 

    if CONST_VP:
        vp.dat.data_with_halos[:] = 2.0 

    start = time.time()
    _, p_recv = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output=print_vtk, use_Neumann_BC_as_source=use_Neumann_BC_as_source)
    end = time.time()
    print(round(end - start,2),flush=True)

    # save shots
    spyro.io.save_shots(model, comm, p_recv, file_name=path+file_name)

elif GUESS==0:
    print("reading guess model",flush=True)
    p_recv = spyro.io.load_shots(model, comm, file_name=path+file_name)
#}}}

def compute_relative_error(p_recv, p_ref_recv): #{{{

    misfit = spyro.utils.evaluate_misfit(model, p_recv, p_ref_recv) # ds_exact[:ll] - guess
    
    J_total = np.zeros((1))
    J_total[0] += spyro.utils.compute_functional(model, misfit) # J += residual[ti][rn] ** 2 (and J *= 0.5)
    J_total[0] /= spyro.utils.compute_functional(model, p_ref_recv) # J += p_ref_recv[ti][rn] ** 2 (and J *= 0.5)
    J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
    J_total[0] /= comm.ensemble_comm.size # ensemble parallelism (sources)
    if comm.comm.size > 1: 
        J_total[0] /= comm.comm.size # spatial parallelism

    E = sqrt(J_total[0]) # relative error as defined in Spyro paper

    return E
#}}}

# retrieve P on the receivers
p_rec1 = p_recv[:,0:20]
p_rec2 = p_recv[:,20:40]
p_ref_rec1 = p_ref_recv[:,0:20]
p_ref_rec2 = p_ref_recv[:,20:40]

# compute the relative errors on each set of receivers 
# FIXME maybe modify it on utils.compute_functional
model["acquisition"]["receiver_locations"] = rec1
E_rec1 = compute_relative_error(p_rec1, p_ref_rec1)

model["acquisition"]["receiver_locations"] = rec2
E_rec2 = compute_relative_error(p_rec2, p_ref_rec2)

model["acquisition"]["receiver_locations"] = rec
E_total = compute_relative_error(p_recv, p_ref_recv)

if comm.ensemble_comm.rank == 0:
    print("E rec1 (%) = "  + str(round(E_rec1*100,4)), flush=True)
    print("E rec2 (%) = "  + str(round(E_rec2*100,4)), flush=True)
    print("E total (%) = " + str(round(E_total*100,4)), flush=True)
    print("p = " + str(model["opts"]["degree"]))
    print("h = " + str(h) + " m")
    print("DOF = " + str(V.dof_count), flush=True) # only works for CG not DG (since the pressure is defined on CG)
    print("Nelem = " + str(mesh.num_cells()), flush=True) 

if comm.ensemble_comm.rank == 0 and PLOT_AT_REC:
    nrec = 10 # middle
    #pe = p_ref_rec1[:,nrec]
    #pg = p_rec1[:,nrec]
    pe = p_ref_rec2[:,nrec]
    pg = p_rec2[:,nrec]
    plt.title("p")
    plt.plot(pe,label='exact')
    plt.plot(pg,label='guess') 
    plt.legend()
    if AMR==1:
        plt.savefig('/home/tdsantos/p_at_rec_wi_amr.png')
    else:
        plt.savefig('/home/tdsantos/p_at_rec_no_amr.png')
    plt.close()

