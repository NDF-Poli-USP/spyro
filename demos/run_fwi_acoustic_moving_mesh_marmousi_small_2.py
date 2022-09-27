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
    "Lz": 2.00-.45,  # depth in km - always positive (waterbottom at z=-0.45 km)
    #"Lx": 2.0,  # width in km - always positive
    "Lx": 4.0,  # width in km - always positive
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
    "num_sources": 4, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.5, -0.01-0.45), (3.5, -0.01-0.45), 4), # waterbottom at z=-0.45 km
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 100, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((0.1, -0.10-0.45), (3.9, -0.10-0.45), 100), # waterbottom at z=-0.45 km REC1
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 2.5, # Final time for event 
    "dt": 0.00025,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
}
#}}}
# make vp {{{
def _make_vp(V, vp_guess=False, field="velocity_model"):
    
    path = "./velocity_models/elastic-marmousi-model/model/"
    if vp_guess: # interpolate from a smoothed field
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy.hdf5" # domain 4 x 2 km2 (x, y) 
    else: # interpolate from the exact field
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy.hdf5" # domain 4 x 2 km2 (x, y) 
        
    with h5py.File(fname, "r") as f:
        Zo = np.asarray(f.get(field)[()]) # original Marmousi data/domain
        nrow, ncol = Zo.shape
        Lz = 2 # Value defined by the velocity model
        Lx = 4 # Value defined by the velocity model
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

        if vp_guess:
            File("guess_vp.pvd").write(vp)
        else:
            File("exact_vp.pvd").write(vp)
    
    return vp
#}}}
comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 60)}

REF = 0
# run reference model {{{
if REF:
    nx = 200
    ny = math.ceil( 100*(model["mesh"]["Lz"]-0.45)/model["mesh"]["Lz"] ) # (Lz-0.45)/Lz
    
    mesh_ref = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_ref.coordinates.dat.data[:, 0] -= 0.0 
    mesh_ref.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.450 km

    # for the exact model, use a higher-order element
    #p = model["opts"]["degree"] # to keep the order defined by the user
    #model["opts"]["degree"] = 4
    element = spyro.domains.space.FE_method(mesh_ref, model["opts"]["method"], model["opts"]["degree"])
    V_ref = FunctionSpace(mesh_ref, element)

    element_DG = spyro.domains.space.FE_method(mesh_ref, "DG", 2) # here, it could be 2 too
    V_DG = FunctionSpace(mesh_ref, element_DG)

    sources = spyro.Sources(model, mesh_ref, V_ref, comm)
    receivers = spyro.Receivers(model, mesh_ref, V_ref, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

    vp_ref = _make_vp(V_DG, vp_guess=False)
    #sys.exit("exit")
    print("Starting forward computation of the exact model",flush=True) 
    start = time.time()
    p_ref, p_ref_recv = spyro.solvers.forward(
        model, mesh_ref, comm, vp_ref, sources, wavelet, receivers, output=False
    )
    end = time.time()
    print(round(end - start,2),flush=True)
    File("p_ref.pvd").write(p_ref[-1])

    spyro.io.save_shots(model, comm, p_ref_recv, file_name="./shots/acoustic_fwi_marmousi_small/p_ref_recv")
#}}}
#sys.exit("exit")
if REF==0:
    p_ref_recv = spyro.io.load_shots(model, comm, file_name="./shots/acoustic_fwi_marmousi_small/p_ref_recv")

# now, prepare to run with different mesh resolutions
FIREMESH = 0
# generate or read a mesh {{{
if FIREMESH: 

    mesh = RectangleMesh(25, 12, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh.coordinates.dat.data[:, 0] -= 0.0 
    mesh.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.450 km

    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = FunctionSpace(mesh, element)
else:
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_150m.msh"
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_100m.msh"
    model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_50m.msh"
    mesh, V = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)
#}}}

AMR = 1
# adapt the mesh using the exact vp, if requested {{{
if AMR:
    nx = 200
    ny = math.ceil( 100*(model["mesh"]["Lz"]-0.45)/model["mesh"]["Lz"] ) # (Lz-0.45)/Lz
    mesh_grid = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_grid.coordinates.dat.data[:, 0] -= 0.0 
    mesh_grid.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.45 km
    V_grid = FunctionSpace(mesh_grid, "CG", 2)
    V_grid_DG = FunctionSpace(mesh_grid, "DG", 2)
    V_vec_grid = VectorFunctionSpace(mesh_grid, "CG", 2)
  
    vp_grid = _make_vp(V_grid_DG, vp_guess=False)
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
    E2 = vp_grid.vector().gather().max() / vp_grid - 1 # a priori error estimate (it starts on 1, so it could be better)

    E = E1
    beta = 0.4 # (0, 1) # for E2 + smooth
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M1 = 1 + alpha * phi
   
    E = E2
    beta = 0.5 # (0, 1) # for E2 + smooth
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
        _P1 = FunctionSpace(mesh, "CG", 1) 
        _M = Function(_P1)
        spyro.mesh_to_mesh_projection(Mfunc, _M, degree=5)
        File("Mfunc_x.pvd").write(_M)
        return _M
    
    V_DG = FunctionSpace(mesh, "DG", 2)
    _vp = _make_vp(V_DG, vp_guess=False)
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
    step = spyro.monge_ampere_solver(mesh, monitor_function, p=1, mask=mask) #fix_boundary_nodes=fix_boundary_nodes) 

    _vp = _make_vp(V_DG, vp_guess=False)
    File("vp_after_amr.pvd").write(_vp)
#}}}
#sys.exit("exit")

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

V_DG = FunctionSpace(mesh, "DG", 2) # FIXME test the forward model with vp in CG
vp_guess = _make_vp(V, vp_guess=True) 
vp_exact = _make_vp(V, vp_guess=False)

# now, starts the FWI computation
m = V.ufl_domain()
W = VectorFunctionSpace(m, V.ufl_element())
X = interpolate(m.coordinates, W)
source_mask = np.where(X.sub(1).dat.data[:] > -0.50)

#ft = Function(V).interpolate(Constant(1.0))
#ft.dat.data[source_mask] = 0.0
#File("ft.pvd").write(ft)
#print(source_mask)
#sys.exit("exit")

class L2Inner(object): #{{{
    # used in the gradient norm computation
    def __init__(self):
        self.A = assemble( TrialFunction(V) * TestFunction(V) * dx, mat_type="matfree")
        self.Ap = as_backend_type(self.A).mat()

    def eval(self, u, v):
        upet = as_backend_type(u).vec()
        vpet = as_backend_type(v).vec()
        Au = self.Ap.createVecLeft()
        self.Ap.mult(upet, Au)
        return vpet.dot(Au)
#}}}
# Objective {{{
class Objective(ROL.Objective):
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.p_guess = None
        self.vp = Function(V)
        self.misfit = 0.0
        self.J = 0.0
        self.p_exact_recv = p_ref_recv
        self.sources = sources
        self.receivers = receivers

    def value(self, x, tol):
        """Compute the functional"""
        J_total = np.zeros((1))
        self.p_guess, p_guess_recv = spyro.solvers.forward(model,
                                                           mesh,
                                                           comm,
                                                           self.vp,
                                                           self.sources, 
                                                           wavelet,
                                                           self.receivers,
                                                           output=False)

        self.misfit = spyro.utils.evaluate_misfit(model, p_guess_recv, self.p_exact_recv)
        J_total[0] += spyro.utils.compute_functional(model, self.misfit, vp=self.vp)
        J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size # paralelismo ensemble (fontes)
        if comm.comm.size > 1:
            J_total[0] /= comm.comm.size # paralelismo espacial

        pe=[]
        pg=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 5
        for ti in range(nt):
            pe.append(self.p_exact_recv[ti][rn])
            pg.append(p_guess_recv[ti][rn])
        plt.title("p")
        plt.plot(pe,label='exact')
        plt.plot(pg,label='guess')
        plt.legend()
        plt.savefig('/home/santos/Desktop/FWI_acoustic.png')
        plt.close()

        self.J = J_total[0]

        return J_total[0]

    def gradient(self, g, x, tol):
        """Compute the gradient of the functional"""
        dJ = Function(V, name="gradient")
        dJ_local = spyro.solvers.gradient(model,
                                          mesh,
                                          comm,
                                          self.vp,
                                          self.receivers,
                                          self.p_guess,
                                          self.misfit)
        if comm.ensemble_comm.size > 1:
            comm.allreduce(dJ_local, dJ)
        else:
            dJ = dJ_local
        dJ /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            dJ /= comm.comm.size

        # mask the source layer
        dJ.dat.data[source_mask] = 0.0
        File("dJ_acoustic_original.pvd").write(dJ)
        
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        self.vp.assign(Function(V, x.vec, name="vp"))

#}}}
# ROL parameters definition {{{
paramsDict = {
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS', # Line Search Quasi-Newton Method 
            'Maximum Storage': 10, #(20==10)
        }
    },
    'Step': {
        'Type': 'Line Search', # the only one that works
        'Line Search': {
            'Function Evaluation Limit': 20,
            'Sufficient Decrease Tolerance': 1e-4,
            'Initial Step Size': 0.01,
            'Accept Last Alpha': False, # allows to go further even if Function Evaluation Limit is reached (not good)
            'Accept Linesearch Minimizer': False, #allows to go further if Function Evaluation Limit is reached (not good)
            'Use Previous Step Length as Initial Guess': False,
            'Line-Search Method':{
                'Type': 'Cubic Interpolation', # default
            },
            'Descent Method': {
                'Type': 'Quasi-Newton Method' # works fine, but it could be better (must use L-BFGS)
                #'Type': 'Steepest descent' # it worked better than Quasi-Newton Method (it ignores L-BFGS)
            },
            'Curvature Condition': {
                'Type': 'Strong Wolfe Conditions', # works fine
            }
        },
    },
    "Status Test": {
        "Gradient Tolerance": 1e-15,
        'Relative Gradient Tolerance': 1e-10,
        "Step Tolerance": 1.0e-15,
        'Relative Step Tolerance': 1e-15,
        "Iteration Limit": 15
    },  
}       
#}}}
# prepare to run FWI, set guess add control bounds to the problem (uses more RAM) {{{
params = ROL.ParameterList(paramsDict, "Parameters")
inner_product = L2Inner()

xig = Function(V) # initial guess
xlo = Function(V) # lower bound
xup = Function(V) # upper bound

xig.dat.data[:] = vp_guess.dat.data[:]
xlo.dat.data[:] = min(vp_exact.dat.data[:])
xup.dat.data[:] = max(vp_exact.dat.data[:])

x_lo = FeVector(xlo.vector(), inner_product)
x_up = FeVector(xup.vector(), inner_product)
    
bnd = ROL.Bounds(x_lo, x_up, 1.0)
obj = Objective(inner_product)
#}}}

Ji=[]
ii=[]
outfile = File("final_vp.pvd")
max_loop_it = 1 # the number of iteration here depends on the max iteration of ROL
max_rol_it = 15

for i in range(max_loop_it):
    print("###### Loop iteration ="+str(i), flush=True)
    print("###### FWI iteration performed ="+str(i*max_rol_it), flush=True)

    # at this moment, mesh_x=mesh_xi
    obj.vp.dat.data_with_halos[:] = xig.dat.data_ro_with_halos[:] # initial guess (mesh_x)
    outfile.write(obj.vp,time=i)

    print("   Minimize: starting ROL solver...", flush=True)
    problem = ROL.OptimizationProblem(obj, FeVector(xig.vector(), inner_product), bnd=bnd)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    mesh_moved = False
    xig.dat.data_with_halos[:] = obj.vp.dat.data_ro_with_halos[:] # no mesh movement, therefore mesh_x=mesh_xi

    ii.append(i)
    Ji.append(obj.J)

outfile.write(obj.vp,time=i)
if COMM_WORLD.rank == 0:
    with open(r'J.txt', 'w') as fp:
        for j in Ji:
            # write each item on a new line
            fp.write("%s\n" % str(j))
        print('Done')

