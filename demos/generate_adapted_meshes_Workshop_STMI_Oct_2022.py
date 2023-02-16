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
        interpolant = RegularGridInterpolator((xo, zo), np.transpose(Zo))

        m = V.ufl_domain()
        W = VectorFunctionSpace(m, V.ufl_element())
        coords = interpolate(m.coordinates, W)
        xq, zq = coords.dat.data[:, 0], coords.dat.data[:, 1]

        _vp = interpolant((xq, zq))
        vp = Function(V)
        vp.dat.data[:] = _vp / 1000 # m/s -> km/s

    return vp
#}}}

case=1
vp_model = 0
Lx = 0
Lz = 0
model_name = ""
lamb = 0

if case==1: # Marmousi model, structured mesh {{{
    method = "CG" 
    degree = 2
    Lx = 17.0
    Lz = 3.5 
    model_name = "marmousi_structured"
    vp_model = 1
    lamb = 0.003
    nx = 200
    ny = math.ceil( nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
    mesh = RectangleMesh(nx, ny, Lx, Lz, diagonal="crossed")
    
    mesh.coordinates.dat.data[:, 0] -= 0.0
    mesh.coordinates.dat.data[:, 1] -= Lz

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
    nx = 200
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
if case==7: # Marmousi model, unstructured mesh {{{
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
if case==8: # Seam model, unstructured mesh {{{
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
if case==9: # Gato do Mato model, unstructured mesh {{{
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
if case==12: # Marmousi model, unstructured, vs {{{
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

# This mesh-grid is used to compute the monitor function
# Alternatively, a point cloud scheme could be used instead (see Jaquet's thesis)
_nx = 300
_ny = math.ceil( _nx*Lz/Lx ) # nx * Lz/Lx, Delta x = Delta z
mesh_grid = RectangleMesh(_nx, _ny, Lx, Lz, diagonal="crossed")
mesh_grid.coordinates.dat.data[:, 0] -= 0.0
mesh_grid.coordinates.dat.data[:, 1] -= Lz
V_grid = FunctionSpace(mesh_grid, "CG", 2)
#V_grid_DG = FunctionSpace(mesh_grid, "DG", 2) # DG will be similar to CG
V_vec_grid = VectorFunctionSpace(mesh_grid, "CG", 2)

vp_grid = _make_vp(V_grid, vp_model=vp_model)
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
File("vp_grid_"+model_name+".pvd").write(vp_grid)

# Huang type monitor function
E1 = sqrt( inner( grad_vp_grid, grad_vp_grid ) ) # gradient based estimate
E2 = vp_grid.vector().gather().max() / (vp_grid+0.1) - 1 # a priori error estimate (it starts on 1, so it could be better)

E = E1
beta = 0.5 # (0, 1) # for E2 + smooth
#beta = 0.10 # (0, 1) # for E2 w/n smooth
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
step = spyro.monge_ampere_solver(mesh, monitor_function, p=2, mask=mask, rtol=1.0e-02) #fix_boundary_nodes=fix_boundary_nodes)
end = time.time()
print(round(end - start,2),flush=True)

print("Monitor function calls: " + str(count),flush=True)
print("monitor function time: " +  str(round(mf_time,2)),flush=True)

_vp = _make_vp(V, vp_model=vp_model) # V is the original space of mesh
File("vp_after_amr_"+model_name+".pvd").write(_vp)
