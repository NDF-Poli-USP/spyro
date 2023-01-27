# run_forward_elastic_moving_mesh_generic_polygon_with_layers.py 
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
    "status": True,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
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
    "num_sources": 1, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.1, 1.9), (0.9, 1.9), 1), # waterbottom at z=-0.45 km # out of domain, only to run with some source
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
    vu = 2.5 - 0.5 # upper layer (km/s)
    vl = 2.5 + 0.5 # lower layer (km/s)
    dv = 0.3*v0 # 30% of perturbation

    n_list = 10
    pol_list = []
    d_theta = 2*pi/n_list
    for i in range(n_list+1):
        theta = 0 + i*d_theta
        r = 0.7*( exp(sin(theta))*sin(2*theta)**2 + exp(cos(theta))*cos(2*theta)**2 ) / 3
        xp = r * cos(theta)
        yp = r * sin(theta)
        pol_list.append((x,y))

    vp_cond = conditional(y >= 0.7, vu, v0)
    vp_cond = conditional(y <= 0.5 - 0.2*x, vl, vp_cond)

    vp_cond = conditional( 300*((x-0.5)*(y-0.5))**2 + ((x-0.5)+(y-0.5))**2 <= 0.300**2, v0+dv, vp_cond)

    vp = Function(V).interpolate(vp_cond)

    vs = Function(V)
    vs.dat.data_with_halos[:] = vp.dat.data_with_halos[:] * 0.6 # 60% is similar to Marmousi elastic model

    rho = Function(V).interpolate(Constant(1.0))

    File("exact_vp.pvd").write(vp)
    File("exact_vs.pvd").write(vs)
    File("exact_rho.pvd").write(rho)

    return vp, vs, rho
#}}}

# controls
FIREMESH = 1    # keep it 1
AMR = 0         # should adapt the mesh?
GUESS = 0       # if 1, run the guess model; otherwise (=0), read results
REF = 0         # if 1, run the reference model; otherwise (=0), read results
QUAD = 0        # if 1, run with quadrilateral elements; otherwise (=0), run with triangles
DG_VP = 0       # if 1, vp is defined on a Discontinuous space (L2 instead of an H1 space)
CONST_VP = 0    # if 1, run with a uniform vp = 2 km/s (it is employed to check convergence rate and wheter adapted mesh introduces errors)
PLOT_AT_REC = 1 # if 1, plot the pressure over time at one receiver
print_vtk = False
use_Neumann_BC_as_source = True # for elastic waves, use Neumann as a source to avoid mesh dependency 

comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 60)} # FIXME if "at" will be the default scheme, then we could remove overlap

# set the reference file name
file_name_uz = "uz_ref_p5_recv_freq_"+str(model["acquisition"]["frequency"]) # with P=5
file_name_ux = "ux_ref_p5_recv_freq_"+str(model["acquisition"]["frequency"]) # with P=5
if platform.node()=='recruta':
    path = ""
    sys.exit("path not defined") 
else:
    path = "/share/tdsantos/shots/elastic_forward_generic_polygon_with_layers_15Hz_Neumann_BC_as_source/" 

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
        sys.exit("do not run the reference model with DGVP")
        element_DG = spyro.domains.space.FE_method(mesh_ref, "DG", 0) # model["opts"]["degree"]) 
        V_ref_DG = FunctionSpace(mesh_ref, element_DG)
        vp_ref = _make_vp(V_ref_DG) 
    else:
        vp_ref, vs_ref, rho_ref = _make_vp(V_ref)
   
    mu_ref   = Function(V_ref).interpolate(rho_ref * vs_ref ** 2.)
    lamb_ref = Function(V_ref).interpolate(rho_ref * (vp_ref ** 2. - 2. * vs_ref ** 2.))

    File("exact_mu.pvd").write(mu_ref)
    File("exact_lamb.pvd").write(lamb_ref)

    #sys.exit("exit")

    sources = spyro.Sources(model, mesh_ref, V_ref, comm)
    receivers = spyro.Receivers(model, mesh_ref, V_ref, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])
    
    if CONST_VP:
        sys.exit("not implemented yet!")
        vp_ref.dat.data_with_halos[:] = 2.0 

    #sys.exit("exit")
    print("Starting forward computation of the exact model",flush=True) 
    start = time.time()
    u_ref, uz_ref, ux_ref, uy_ref = spyro.solvers.forward_elastic_waves(
        model, mesh_ref, comm, rho_ref, lamb_ref, mu_ref, sources, wavelet, receivers, output=print_vtk, use_Neumann_BC_as_source=use_Neumann_BC_as_source
    )
    end = time.time()
    print(round(end - start,2),flush=True)

    spyro.io.save_shots(model, comm, uz_ref, file_name=path+file_name_uz)
    spyro.io.save_shots(model, comm, ux_ref, file_name=path+file_name_ux)

    _h = round(1000*model["mesh"]["Lx"]/_nx)
    if comm.ensemble_comm.rank == 1:
        print("Reference model:", flush=True)
        print("p = " + str(model["opts"]["degree"]))
        print("h = " + str(_h) + " m")
        print("DOF = " + str(V_ref.dof_count*2), flush=True)
        print("Nelem = " + str(mesh_ref.num_cells()), flush=True) 

    sys.exit("Reference model finished!")

elif REF==0: 
    print("reading reference model",flush=True)
    uz_ref = spyro.io.load_shots(model, comm, file_name=path+file_name_uz)
    ux_ref = spyro.io.load_shots(model, comm, file_name=path+file_name_ux)
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
    iii=0

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
  
    _, vs_grid, _ = _make_vp(V_grid) # use vs only
    grad_vs_grid = Function(V_vec_grid)

    u_cts = TrialFunction(V_vec_grid)
    v_cts = TestFunction(V_vec_grid)
    a = inner(v_cts, u_cts)*dx
    L = inner(v_cts, grad(vs_grid))*dx
    _cg = {
        "ksp_type": "cg",
        "pc_type": "bjacobi",
        "pc_sub_type": "ilu",
    }
    _problem = LinearVariationalProblem(a, L, grad_vs_grid, bcs=None)
    _solver = LinearVariationalSolver(_problem, solver_parameters=_cg) 
    _solver.solve()

    File("grad_vs_grid.pvd").write(grad_vs_grid)
    File("vs_grid.pvd").write(vs_grid)

    # Huang type monitor function
    E1 = sqrt( inner( grad_vs_grid, grad_vs_grid ) ) # gradient based estimate
    E2 = vs_grid.vector().gather().max() / vs_grid - 1 # a priori error estimate (it starts on 1, so it is better)

    E = E1
    #beta = 0.5 # (0, 1) # for E2 + smooth
    beta = 0.20 # (0, 1) # for E2 w/n smooth
    #beta = 0.30 # (0, 1) # for E2 w/n smooth
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M1 = 1 + alpha * phi
   
    E = E2
    #beta = 0.5 # (0, 1) # for E2 + smooth
    beta = 0.3 # (0, 1) # for E2 w/n smooth
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M2 = 1 + alpha * phi

    M = max_value(M1,M2)

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
        _, _vs, _ = _make_vp(V_DG)
    else: 
        _, _vs, _ = _make_vp(V) # V is the original space of mesh

    File("vs_before_amr.pvd").write(_vs)
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
        _, _vs, _ = _make_vp(V_DG)
    else: 
        _, _vs, _ = _make_vp(V) # V is the original space of mesh
    File("vs_after_amr.pvd").write(_vs)
#}}}
#sys.exit("exit")

# set the file name
h = round(1000*model["mesh"]["Lx"]/nx)
if QUAD==1:
    file_name_uz = "uz_recv_QUAD_AMR_" + str(AMR) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])
    file_name_ux = "ux_recv_QUAD_AMR_" + str(AMR) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])
else:
    file_name_uz = "uz_recv_AMR_" + str(AMR) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])
    file_name_ux = "ux_recv_AMR_" + str(AMR) + "_DGVP_" + str(DG_VP) + "_p_" + str(model["opts"]["degree"]) + "_h_" + str(h) + "m_freq_" + str(model["acquisition"]["frequency"])

# run the guess model with a given mesh {{{
if GUESS==1:
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

    if DG_VP==1:
        element_DG = spyro.domains.space.FE_method(mesh, "DG", 0) #FIXME model["opts"]["degree"])
        V_DG = FunctionSpace(mesh, element_DG)
        vp, vs, rho = _make_vp(V_DG)
    else: 
        vp, vs, rho = _make_vp(V) 

    mu   = Function(V).interpolate(rho * vs ** 2.)
    lamb = Function(V).interpolate(rho * (vp ** 2. - 2. * vs ** 2.))

    if CONST_VP:
        sys.exit("not implemented yet")
        vp.dat.data_with_halos[:] = 2.0 

    start = time.time()
    u, uz, ux, uy = spyro.solvers.forward_elastic_waves(model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=print_vtk, use_Neumann_BC_as_source=use_Neumann_BC_as_source)
    end = time.time()
    print(round(end - start,2),flush=True)

    # save shots
    spyro.io.save_shots(model, comm, uz, file_name=path+file_name_uz)
    spyro.io.save_shots(model, comm, ux, file_name=path+file_name_ux)

elif GUESS==0:
    print("reading guess model",flush=True)
    uz = spyro.io.load_shots(model, comm, file_name=path+file_name_uz)
    ux = spyro.io.load_shots(model, comm, file_name=path+file_name_ux)
#}}}

def compute_relative_error(uz, ux, uz_ref, ux_ref): # {{{
    J_scale = sqrt(1.e14)

    misfit_uz = spyro.utils.evaluate_misfit(model, uz, uz_ref)# uz_ref[i] - uz[i] (vector)
    misfit_ux = spyro.utils.evaluate_misfit(model, ux, ux_ref)# ux_ref[i] - ux[i] (vector)

    J_total = np.zeros((1))
    J_total[0] += spyro.utils.compute_functional(model, J_scale * misfit_uz) # J += (uz_ref - uz) ** 2 (and J *= 0.5)
    J_total[0] += spyro.utils.compute_functional(model, J_scale * misfit_ux) # J += (ux_ref - ux) ** 2 (and J *= 0.5)

    J_ref   = np.zeros((1))
    J_ref[0]   += spyro.utils.compute_functional(model, J_scale * uz_ref) # J += uz_ref[ti][rn] ** 2 (and J *= 0.5)
    J_ref[0]   += spyro.utils.compute_functional(model, J_scale * ux_ref) # J += ux_ref[ti][rn] ** 2 (and J *= 0.5)
    
    # divide
    J_total[0] /= J_ref[0] # ( (uz_ref - uz)**2 + (ux_ref - ux)**2 ) / (uz_ref**2 + ux_ref**2)

    J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
    J_total[0] /= comm.ensemble_comm.size # paralelismo ensemble (fontes)
    if comm.comm.size > 1:
        J_total[0] /= comm.comm.size # paralelismo espacial

    E = sqrt(J_total[0]) # relative error as defined in Spyro paper

    return E
#}}}

# retrieve UZ on the receivers
uz_rec1 = uz[:,0:20] 
uz_rec2 = uz[:,20:40] 
uz_ref_rec1 = uz_ref[:,0:20]
uz_ref_rec2 = uz_ref[:,20:40] 

# retrieve UX on the receivers
ux_rec1 = ux[:,0:20]
ux_rec2 = ux[:,20:40]
ux_ref_rec1 = ux_ref[:,0:20]
ux_ref_rec2 = ux_ref[:,20:40]

# compute the relative errors on each set of receivers 
# FIXME maybe modify it on utils.compute_functional
model["acquisition"]["receiver_locations"] = rec1
E_rec1 = compute_relative_error(uz_rec1, ux_rec1, uz_ref_rec1, ux_ref_rec1)

model["acquisition"]["receiver_locations"] = rec2
E_rec2 = compute_relative_error(uz_rec2, ux_rec2, uz_ref_rec2, ux_ref_rec2)

model["acquisition"]["receiver_locations"] = rec
E_total = compute_relative_error(uz, ux, uz_ref, ux_ref)

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
    #pe = uz_ref_rec1[:,nrec]
    #pg = uz_rec1[:,nrec]
    pe = uz_ref_rec2[:,nrec]
    pg = uz_rec2[:,nrec]
    plt.title("uz")
    plt.plot(pe,label='exact')
    plt.plot(pg,label='guess') 
    plt.legend()
    if AMR==1:
        plt.savefig('/home/tdsantos/uz_at_rec_wi_amr.png')
    else:
        plt.savefig('/home/tdsantos/uz_at_rec_no_amr.png')
    plt.close()

