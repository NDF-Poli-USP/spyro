from firedrake import *
from movement import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import ufl
import spyro
import math
import h5py
import sys
import time

# make vp {{{
def _make_vp(V, vp_model=1, field="velocity_model"):
    
    # vp_model = 1: Marmousi
    # vp model = 2: SEAM
    # vp model = 3: Gato do Mato
    # vp_model = 4: Marmousi (vs)

    if vp_model==1:
        path = "./velocity_models/elastic-marmousi-model/model/"
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5" # domain 4 x 2 km2 (x, y) 
        Lz = 3.5
        Lx = 17.0
    elif vp_model==2:
        path = "./velocity_models/seam/"
        fname = path + "Vp_at_y=20km.hdf5"
        Lz = 15.0
        Lx = 35.0
    elif vp_model==3:
        path = "./velocity_models/gato_do_mato/"
        fname = path + "c3_2020.npy.hdf5"
        Lx = 17.3120
        Lz = 7.5520
    elif vp_model==4:
        path = "./velocity_models/elastic-marmousi-model/model/"
        fname = path + "MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5" # domain 4 x 2 km2 (x, y) 
        Lz = 3.5
        Lx = 17.0
    else:
        sys.exit("vp model not found!")

    with h5py.File(fname, "r") as f:
        Zo = np.asarray(f.get(field)[()]) # original Marmousi data/domain
        nrow, ncol = Zo.shape
        zo = np.linspace(-Lz, 0.0, nrow) # original Marmousi data/domain
        xo = np.linspace(0.0,  Lx, ncol) # original Marmousi data/domain
        interpolant = RegularGridInterpolator((xo, zo), np.transpose(Zo), bounds_error=False)

        m = V.ufl_domain()
        W = VectorFunctionSpace(m, V.ufl_element())
        coords = interpolate(m.coordinates, W)
        xq, zq = coords.dat.data[:, 0], coords.dat.data[:, 1]

        _vp = interpolant((xq, zq))
        vp = Function(V)
        vp.dat.data[:] = _vp / 1000 # m/s -> km/s

    return vp
#}}}

#case=13 # Marmousi
case=5  # SEAM 
vp_model = 0
Lx = 0
Lz = 0
Tx = 0
Tz = 0
model_name = ""
lamb = 0
adapt_mesh = 0
run_ref = 1
use_Neumann_BC_as_source = False
print_vtk = True
use_DG0 = False 

if case==1: # Marmousi model, structured mesh {{{
    method = "CG" 
    degree = 2
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_structured"
    vp_model = 1
    lamb = 0.003
    #nx = 200
    nx = 100
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, diagonal="crossed")
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz
    mesh.clear_spatial_index()

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)

#}}}
if case==2: # Seam model, structured mesh {{{
    method = "CG" 
    degree = 2
    Lx = 35.0
    Lz = 15.0 
    model_name = "seam_structured"
    vp_model = 2
    lamb = 0.10
    nx = 200
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, diagonal="crossed")
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)

#}}}
if case==3: # Gato do Mato model, structured mesh {{{
    method = "CG" 
    degree = 2
    Lx = 17.3120
    Lz = 7.5520
    model_name = "gato_do_mato_structured"
    vp_model = 3
    lamb = 0.05
    nx = 200
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, diagonal="crossed")
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)

#}}}
if case==4: # Marmousi model, quads {{{
    method = "CG" 
    degree = 4
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_quads"
    vp_model = 1
    lamb = 0.003
    nx = 200
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, quadrilateral=True)
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)

#}}}
if case==5: # Seam model, quads {{{
    method = "CG" 
    degree = 4
    Lx = 35.0
    Lz = 15.0 
    model_name = "seam_quads"
    vp_model = 2
    lamb = 0.10 
    if run_ref:
        nx = 300 # REF
    else:
        nx = 150 # to test amr
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, quadrilateral=True)
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    print("Nelem = " + str(mesh.num_cells()), flush=True)

#}}}
if case==6: # Gato do Mato model, quads {{{
    method = "CG" 
    degree = 4
    Lx = 17.3120
    Lz = 7.5520
    model_name = "gato_do_mato_quads"
    vp_model = 3
    lamb = 0.05
    nx = 200
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, quadrilateral=True)
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)

#}}}
if case==7: # Marmousi model, unstructured mesh (uniform) {{{
    method = "CG" 
    degree = 2
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_unstructured"
    vp_model = 1
    lamb = 0.003

    mesh = Mesh('./meshes/marmousi_Workshop_STMI_Oct_2022.msh')

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)

#}}}
if case==8: # Seam model, unstructured mesh (uniform) {{{
    method = "CG" 
    degree = 2
    Lx = 35.0
    Lz = 15.0 
    model_name = "seam_unstructured"
    vp_model = 2
    lamb = 0.10
    
    mesh = Mesh('./meshes/seam_Workshop_STMI_Oct_2022.msh')

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    print("Nelem = " + str(mesh.num_cells()), flush=True)

#}}}
if case==9: # Gato do Mato model, unstructured mesh (uniform) {{{
    method = "CG" 
    degree = 2
    Lx = 17.3120
    Lz = 7.5520
    model_name = "gato_do_mato_unstructured"
    vp_model = 3
    lamb = 0.05
 
    mesh = Mesh('./meshes/gato_do_mato_Workshop_STMI_Oct_2022.msh')

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)

#}}}
if case==10: # Marmousi model, structured mesh, vs {{{
    method = "CG" 
    degree = 2
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_structured_vs"
    vp_model = 4 # vs
    lamb = 0.003
    nx = 200
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, diagonal="crossed")
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    print("Nelem = " + str(mesh.num_cells()), flush=True)

#}}}
if case==11: # Marmousi model, structured mesh with equi triangles, vs {{{
    method = "CG" 
    degree = 2
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_structured_equi_tria_vs"
    vp_model = 4 # vs
    #lamb = 0.003
    lamb = 0.001
    
    mesh = Mesh('./meshes/marmousi_Workshop_STMI_Oct_2022_2.msh')

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    print("Nelem = " + str(mesh.num_cells()), flush=True)

#}}}
if case==12: # Marmousi model, unstructured (uniform), vs {{{
    method = "CG" 
    degree = 2
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_unstructured_vs"
    vp_model = 4 # vs
    lamb = 0.003
    
    mesh = Mesh('./meshes/marmousi_Workshop_STMI_Oct_2022.msh')

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    print("Nelem = " + str(mesh.num_cells()), flush=True)

#}}}
if case==13: # Marmousi model, unstructured mesh, from gmsh with non-uniform resolution {{{
    method = "KMV" 
    degree = 4
    #Lx = 17.0
    #Lz = 3.5 
    Lx = 7.0 # using the same aspect ratio of SEAM
    Lz = 3.0
    Tx = 6
    Tz = 0.5
    model_name = "marmousi_unstructure_gmsh_nonuniform"
    vp_model = 1
    lamb = 0.001 # original

    #mesh = Mesh('./meshes/marmousi_mesh_for_moving_mesh_paper.msh') for some reason, the Monge-Ampere solver does not converge with this mesh
    # rotate mesh (it comes from SeismicMesh, which has a different coordinate system)
    #cos_theta = math.cos(-math.pi/2.)
    #sin_theta = math.sin(-math.pi/2.)
    #meshx = np.copy(mesh.coordinates.dat.data[:, 0])
    #meshy = np.copy(mesh.coordinates.dat.data[:, 1])
    #mesh.coordinates.dat.data[:, 0] = meshx*cos_theta - meshy*sin_theta
    #mesh.coordinates.dat.data[:, 1] = meshx*sin_theta + meshy*cos_theta 
    #mesh.coordinates.dat.data[:, 1] *= -1

    #pos = np.where(mesh.coordinates.dat.data[:, 0] < 0)
    #mesh.coordinates.dat.data[pos, 0] = 0
    #pos = np.where(mesh.coordinates.dat.data[:, 1] < -3.5)
    #mesh.coordinates.dat.data[pos, 1] = -3.5
    
    if run_ref:
        # REF
        nx = 150 
        ny = math.ceil( nx*Lz/Lx ) 
        mesh = RectangleMesh(nx, ny, Lx, Lz, diagonal="crossed")
        mesh.coordinates.dat.data[:, 0] -= 0.0 - Tx
        mesh.coordinates.dat.data[:, 1] -= Lz + Tz
        mesh.clear_spatial_index()
    else:
        mesh = Mesh('./meshes/marmousi_mesh_for_moving_mesh_paper.msh')

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    
    _f = Function(V)
    File("mesh_"+model_name+".pvd").write(_f)
    #sys.exit("exit")
#}}}
if case==14: # Marmousi model, structured mesh with equi triangles {{{
    method = "CG" 
    degree = 2
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_structured_equi_tria"
    vp_model = 1
    #lamb = 0.003
    lamb = 0.001
    
    mesh = Mesh('./meshes/marmousi_Workshop_STMI_Oct_2022_2.msh')

    element = spyro.domains.space.FE_method(mesh, method, degree)
    V = FunctionSpace(mesh, element)
    #print("Nelem = " + str(mesh.num_cells()), flush=True)

#}}}

# code to adapt the mesh {{{
# This mesh-grid is used to compute the monitor function
# Alternatively, a point cloud scheme could be used instead (see Jaquet's thesis)
_nx = 300
_ny = math.ceil( _nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
mesh_grid = RectangleMesh(_nx, _ny, Lx, Lz, diagonal="crossed")
mesh_grid.coordinates.dat.data[:, 0] -= 0.0 - Tx
mesh_grid.coordinates.dat.data[:, 1] -= Lz + Tz
mesh_grid.clear_spatial_index()

V_grid = FunctionSpace(mesh_grid, "CG", 2)
#V_grid_DG = FunctionSpace(mesh_grid, "DG", 2) # DG will be similar to CG
V_vec_grid = VectorFunctionSpace(mesh_grid, "CG", 2)

vp_grid = _make_vp(V_grid, vp_model=vp_model)
File("vp_grid_"+model_name+".pvd").write(vp_grid)

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

File("grad_vp_grid_"+model_name+".pvd").write(grad_vp_grid)

# Huang type monitor function
E1 = sqrt( inner( grad_vp_grid, grad_vp_grid ) ) # gradient based estimate
E2 = vp_grid.vector().gather().max() / (vp_grid+0.1) - 1 # a priori error estimate (it starts on 1, so it could be better)

E = E1
beta = 0.5 # (0, 1) # for E2 + smooth  #FIXME use this for SEAM quads
phi = sqrt( 1 + E*E ) - 1
phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
alpha = beta / ( phi_hat * ( 1 - beta ) )
M1 = 1 + alpha * phi

E = E2
beta = 0.5 # (0, 1) # for E2 + smooth  #FIXME use this for SEAM quads
phi = sqrt( 1 + E*E ) - 1
phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
alpha = beta / ( phi_hat * ( 1 - beta ) )
M2 = 1 + alpha * phi

#M = min_value(max_value(M1,M2), 4)
M = max_value(M1,M2)

# Define the monitor function to be projected onto the adapted mesh
Mfunc = Function(V_grid)
Mfunc.interpolate(M)
    
# smooth the monitor function
if 1: # {{{
    #lamb = 0.003

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

# FIXME testing limits
#m_max = 10
#inds = [i for (i, val) in enumerate(Mfunc.dat.data) if val > m_max]
#Mfunc.dat.data[inds] = m_max
#print(inds)

#FIXME
#Mfunc.dat.data[:] = 1.0

File("Mfunc_"+model_name+".pvd").write(Mfunc) 
#sys.exit("exit")

count = 0
mf_time = 0

def monitor_function(mesh): # here, mesh is the physical doman, i.e., x (=xi+Grad phi)
    # project onto "mesh" that is being adapted (i.e., mesh_x)
    global count
    global mf_time
    
    _ti=time.time()

    _P1 = FunctionSpace(mesh, "CG", 1) # P1 works better here (even p>1 does not improve when coarse meshes are employed) 
    _M = Function(_P1)
     
    #FIXME use projection for spatial parallelism until we solve the issue with "at"
    #spyro.mesh_to_mesh_projection(Mfunc, _M, degree=6)

    # it works for serial so far
    #_m = _P1.ufl_domain() # quads
    #_W = VectorFunctionSpace(_m, _P1.ufl_element())
    #_X = interpolate(_m.coordinates, _W)
    #_M.dat.data[:] = Mfunc.at(_X.dat.data_ro, dont_raise=True, tolerance=0.001)
    
    # FIXME trying other scheme
    _X = Function(mesh.coordinates)
    _M.dat.data[:] = Mfunc.at(_X.dat.data_ro, dont_raise=True, tolerance=0.001)

    count += 1
    _tf = time.time()
    mf_time += _tf-_ti

    #File("Mfunc_x_"+model_name+".pvd").write(_M)
    return _M

_vp = _make_vp(V, vp_model=vp_model) # V is the original space of mesh
File("vp_before_amr_"+model_name+".pvd").write(_vp)
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

start = time.time()
if adapt_mesh:
    step = spyro.monge_ampere_solver(mesh, monitor_function, p=2, mask=mask, rtol=1.0e-02, print_solution=False) #fix_boundary_nodes=fix_boundary_nodes)
end = time.time()
print(round(end - start,2),flush=True)

print("Monitor function calls: " + str(count),flush=True)
print("monitor function time: " +  str(round(mf_time,2)),flush=True)

_vp = _make_vp(V, vp_model=vp_model) # V is the original space of mesh
File("vp_after_amr_"+model_name+".pvd").write(_vp)
#sys.exit("exit")
#}}}

# code to run the forward model (acoustic waves) {{{

# define the model parameters {{{
model = {}

model["opts"] = {
    ## for Marmousi ##
    #"method": "KMV",  # either CG or KMV or spectral
    #"quadrature": "KMV", # Equi or KMV #FIXME it will be removed
    ## for SEAM ##
    "method": "CG",
    "quadrature": "GLL",
    "degree": 4,
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
#rec = spyro.create_transect((1, -0.5), (34, -0.5), 10 ) # for SEAM
rec = spyro.create_transect((7, -0.6), (12, -0.6), 10 ) # for Marmousi


model["acquisition"] = {
    "source_type": "Ricker",
    "frequency": 12.0, # freq peak = 6 Hz, max freq = 15 Hz (see Jaquet's  Thesis) # for Marmousi
    #"frequency": 3.0, # freq peak = 6 Hz, max freq = 15 Hz (see Jaquet's  Thesis) # for SEAM 
    "delay": 1.0, # FIXME check this
    "num_sources": 1, #FIXME not used (remove it, and update an example script)
    #"source_pos": spyro.create_transect((17.5, -7.5), (17.5, -7.5), 1), # for SEAM
    "source_pos": spyro.create_transect((9.5, -2.0), (9.5, -2.0), 1), # for Marmousi used for 8, 9 and 12 Hz
    #"source_pos": spyro.create_transect((8.0, -2.0), (8.0, -2.0), 1), # for Marmousi used for 12 Hz
    "amplitude": 1.0, #FIXME check this
    "num_receivers": len(rec), #FIXME not used (remove it, and update an example script)
    "receiver_locations": rec,
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 0.00025, # Final time for event  
    "tf": 1.3, # Final time for event  for Marmousi
    #"tf": 4.0, # Final time for event  for SEAM
    "dt": 0.00025, # timestep size for Marmousi 
    #"dt": 0.00025*4, # timestep size for SEAM 
    #"nspool":  50,  # for SEAM (20 for dt=0.00050) how frequently to output solution to pvds
    "nspool":  80,  # for Marmousi (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10000000,  # how frequently to save solution to RAM
}
#}}}

# spyro wants the mesh to start at (0,0)
#mesh.coordinates.dat.data[:, 0] -= Tx
#mesh.coordinates.dat.data[:, 1] += Tz
#mesh.clear_spatial_index()

comm = spyro.utils.mpi_init(model)
#distribution_parameters={"partition": True,
#                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 60)} # FIXME if "at" will be the default scheme, then we could remove overlap

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

method = "DG"
degree = 0
element_DG = spyro.domains.space.FE_method(mesh, method, degree)
V_DG = FunctionSpace(mesh, element_DG)
_vp_DG = _make_vp(V_DG, vp_model=vp_model) # V is the original space of mesh

if use_DG0:
    _vp2run = _vp_DG
else:
    _vp2run = _vp
File("vp_to_run_"+model_name+".pvd").write(_vp2run)

sys.exit("exit")


start = time.time()
_, p_recv = spyro.solvers.forward(model, mesh, comm, _vp2run, sources, wavelet, receivers, output=print_vtk, use_Neumann_BC_as_source=use_Neumann_BC_as_source)
end = time.time()
print(round(end - start,2),flush=True)

# save shots
#spyro.io.save_shots(model, comm, p_recv, file_name=path+file_name)


#}}}
