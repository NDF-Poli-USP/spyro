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
#import SeismicMesh
import weakref
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from spyro.io import write_function_to_grid, create_segy
#from ..domains import quadrature, space
import platform

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
    "dt": 0.00050,  # timestep size for guess model
    #"dt": 0.00050/4,  # timestep size 0.00050/4 for reference model with nx = 200 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
}
#}}}
# make vp {{{
def _make_vp(V, vp_guess=False, field="velocity_model", apply_exact_vp_over_source_line=False):
    
    #path = "./velocity_models/elastic-marmousi-model/model/"
    path = "/share/tdsantos/velocity_models/elastic-marmousi-model/model/"
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
    
        if vp_guess==True and apply_exact_vp_over_source_line==True: # use exact vp for z>-0.53 (see source_mask definition) 
            _vp_exact = _make_vp(V, vp_guess=False)
            _m = V.ufl_domain()
            _W = VectorFunctionSpace(_m, V.ufl_element())
            _X = interpolate(_m.coordinates, _W)
            _source_mask = np.where(_X.sub(1).dat.data[:] > -0.53)
            vp.dat.data[_source_mask] = _vp_exact.dat.data[_source_mask] 

        if vp_guess:
            File("guess_vp.pvd").write(vp)
        else:
            File("exact_vp.pvd").write(vp)
    
    return vp
#}}}

# controls
AMR = 0
REF = 0
QUAD = 1
FIREMESH = 1
ADAPT_MESH_FOR_GUESS_VP = 1 # adapt the mesh using the initial guess (vp)

if QUAD==1:
    model["opts"]["method"] = "CG"
    model["opts"]["quadrature"] = "GLL"
    model["opts"]["degree"] = 4
    #model["opts"]["degree"] = 8

comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 60)}

# set the reference file name
file_name = "p_ref_p4_recv_freq_"+str(model["acquisition"]["frequency"]) # with P=4
if platform.node()=='recruta':
    path = "./shots/acoustic_fwi_moving_mesh_marmousi_small/"
else:
    #path = "/share/tdsantos/shots/acoustic_fwi_moving_mesh_marmousi_small/"
    path = "/share/tdsantos/shots/acoustic_fwi_moving_mesh_marmousi_small_h20_m_TOO_SLOW/" # nx=200 (ref model)

# run reference model {{{
if REF:
    _nx = 200 # 200 is the original, 50 is to speed-up
    _ny = math.ceil( _nx*model["mesh"]["Lz"]/model["mesh"]["Lx"] )
   
    # here, we do not need overlaping vertices
    distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}

    mesh_ref = RectangleMesh(_nx, _ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_ref.coordinates.dat.data[:, 0] -= 0.0 
    mesh_ref.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.450 km
    mesh_ref.clear_spatial_index()

    # for the exact model, use a higher-order element
    model["opts"]["degree"] = 4
    element = spyro.domains.space.FE_method(mesh_ref, model["opts"]["method"], model["opts"]["degree"])
    V_ref = FunctionSpace(mesh_ref, element)

    sources = spyro.Sources(model, mesh_ref, V_ref, comm)
    receivers = spyro.Receivers(model, mesh_ref, V_ref, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

    vp_ref = _make_vp(V_ref, vp_guess=False)
    #sys.exit("exit")
    print("Starting forward computation of the exact model",flush=True) 
    start = time.time()
    p_ref, p_ref_recv = spyro.solvers.forward(
        model, mesh_ref, comm, vp_ref, sources, wavelet, receivers, output=False
    )
    end = time.time()
    print(round(end - start,2),flush=True)
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

if REF==0:
    print("reading reference model",flush=True)
    p_ref_recv_original = spyro.io.load_shots(model, comm, file_name=path+file_name)
    
    print("interpolating p_ref_recv on the time points of the guess model",flush=True)
    n_t, n_rec = p_ref_recv_original.shape
    dt = model["timeaxis"]['tf']/n_t

    time_vector_ref = np.zeros((1,n_t)) 
    for ite in range(n_t):
        time_vector_ref[0,ite] = dt*ite

    dt = model["timeaxis"]['dt'] # guess
    n_t = int(model["timeaxis"]['tf']/dt) # guess

    time_vector_guess = np.zeros((1,n_t))
    for ite in range(n_t):
        time_vector_guess[0,ite] = dt*ite

    p_ref_recv = np.zeros((n_t, n_rec)) # size of the guess model
    for rec in range(n_rec):
        f = interp1d(time_vector_ref[0,:], p_ref_recv_original[:,rec] )
        p_ref_recv[:,rec] = f(time_vector_guess[0,:])

    if 0:
        plt.title("p")
        rec = 99
        plt.plot(time_vector_ref[0,:],p_ref_recv_original[:,rec],label='exact')
        plt.plot(time_vector_guess[0,:],p_ref_recv[:,rec],label='guess')
        plt.legend()
        plt.savefig('/home/tdsantos/test_acoustic.png')
        plt.close()

    #sys.exit("exit")
#}}}

# now, prepare to run with different mesh resolutions
# generate or read the initial mesh and space V on which vp is defined {{{
if FIREMESH: 
    #_nx = 25 # too coarse
    _nx = 50 # h=80m
    _ny = math.ceil( _nx*model["mesh"]["Lz"]/model["mesh"]["Lx"] )

    if QUAD==0:
        mesh = RectangleMesh(_nx, _ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    elif QUAD==1:
        mesh = RectangleMesh(_nx, _ny, model["mesh"]["Lx"], model["mesh"]["Lz"], quadrilateral=True, comm=comm.comm,
                            distribution_parameters=distribution_parameters)

    mesh.coordinates.dat.data[:, 0] -= 0.0 
    mesh.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.450 km
    mesh.clear_spatial_index()

    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = FunctionSpace(mesh, element)
else:
    if QUAD==1:
        sys.exit("QUAD=1, but FIREMESH=0, exiting")
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_150m.msh"
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_100m.msh"
    #model["mesh"]["meshfile"] = "./meshes/marmousi_small_no_water_h_50m.msh"
    model["mesh"]["meshfile"] = "/share/tdsantos/meshes/marmousi_small_no_water_h_50m.msh"
    mesh, V = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)
#}}}

# define the source wave function
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

# define max and min values of vp (used in ROL)
vp_exact = _make_vp(V, vp_guess=False)
#vp_guess = _make_vp(V, vp_guess=True)  
min_vp = vp_exact.vector().gather().min()
max_vp = vp_exact.vector().gather().max()

# keep the original coordinates (computational mesh) to be used during the mesh adaption, if requested
xi = Function(mesh.coordinates, name="Computational coordinates")

# define mask to be applied over the source region (it will avoid high gradient values over there)
#m = V.ufl_domain()
#W = VectorFunctionSpace(m, V.ufl_element())
#X = interpolate(m.coordinates, W)
#source_mask = np.where(X.sub(1).dat.data[:] > -0.53)
#if 0:
#    ft = Function(V).interpolate(Constant(1.0))
#    ft.dat.data[source_mask] = 0.0
#    File("ft.pvd").write(ft)
#    print(source_mask)
#    sys.exit("exit")

# controls
Ji=[]
outfile = File("final_vp.pvd")
max_loop_it = 4 # the number of iteration here depends on the max iteration of ROL (IT CAN NOT BE SMALLER THAN 'Maximum Storage')
#max_rol_it = 20 
max_rol_it = 0 # FIXME to debug

class L2Inner(object): #{{{
    # used in the gradient norm computation
    def __init__(self, V):
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
    def __init__(self, V, inner_product, p_ref_recv, sources, receivers, source_mask):
        ROL.Objective.__init__(self)
        self.V = V
        self.inner_product = inner_product
        self.p_guess = None
        self.vp = Function(self.V)
        self.misfit = 0.0
        self.J = 0.0
        self.p_exact_recv = p_ref_recv
        self.sources = sources
        self.receivers = receivers
        self.source_mask = source_mask

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
        if 1:
            num_receivers = len(model["acquisition"]["receiver_locations"])
            dt = model["timeaxis"]["dt"]
            tf = model["timeaxis"]["tf"]
            nt = int(tf / dt)  # number of timesteps
            for ti in range(nt):
                for rn in range(num_receivers):
                    if self.misfit[ti][rn] > 1:
                        print("receiver with error: "+str(rn),flush=True)
                        pg=[]
                        for ti in range(nt):
                            pg.append(p_guess_recv[ti][rn])
                        plt.title("p")
                        plt.plot(pg,label='guess')
                        plt.legend()
                        plt.savefig('/home/tdsantos/receiver_with_error.png')
                        plt.close()
                        #sys.exit("close")



        J_total[0] += spyro.utils.compute_functional(model, self.misfit, vp=self.vp)
        J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size # paralelismo ensemble (fontes)
        if comm.comm.size > 1:
            J_total[0] /= comm.comm.size # paralelismo espacial

        if 0:
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
        dJ = Function(self.V, name="gradient")
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

        # mask the source layer to avoid higher gradient values over there
        dJ.dat.data[self.source_mask] = 0.0
        File("dJ_acoustic_original.pvd").write(dJ)
        
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        self.vp.assign(Function(self.V, x.vec, name="vp"))

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
        "Iteration Limit": max_rol_it
    },  
}       
#}}}
# create grid/mesh to interpolate inverted field {{{
nx = 200
ny = math.ceil( nx*model["mesh"]["Lz"]/model["mesh"]["Lx"] ) # nx * Lz/Lx, Delta x = Delta z

if QUAD==0:
    mesh_grid = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                        distribution_parameters=distribution_parameters)
elif QUAD==1:
    pad = 1 #FIXME testing a larger grid domain COME BACK HERE fix the issue with at and quad
    mesh_grid = RectangleMesh(nx, ny, model["mesh"]["Lx"]+pad, model["mesh"]["Lz"]+pad, quadrilateral=True, comm=comm.comm,
                        distribution_parameters=distribution_parameters)

mesh_grid.coordinates.dat.data[:, 0] -= 0.0 + pad/2 
mesh_grid.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 + pad/2 # waterbottom at z=-0.45 km
mesh_grid.clear_spatial_index()

V_grid = FunctionSpace(mesh_grid, "CG", 2)
V_vec_grid = VectorFunctionSpace(mesh_grid, "CG", 2)
#}}}
# function to transfer data from one mesh to another {{{
def _mesh_to_mesh_interpolation(data_in, data_out): 
    if QUAD==0: # ok, use mesh projection
        spyro.mesh_to_mesh_projection(data_in, data_out, degree=5) # from data_in to data_out
    elif QUAD==1: # quadrilateral elements
        # it works for serial so far
        _m = data_out.ufl_domain() # quads
        _W = VectorFunctionSpace(_m, data_out.ufl_element())
        _X = interpolate(_m.coordinates, _W)
        data_out.dat.data[:] = data_in.at(_X.dat.data_ro, dont_raise=True, tolerance=0.001) 
#}}}
# function to adapt the mesh {{{
def adapt_mesh(vp_grid, grad_vp_grid, mesh):
    print("   Adapt mesh: starting Monge-Ampere solver...", flush=True)

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
    #sys.exit("exit")

    # Huang type monitor function
    E1 = sqrt( inner( grad_vp_grid, grad_vp_grid ) ) # gradient based estimate
    E2 = vp_grid.vector().gather().max() / vp_grid - 1 # a priori error estimate (it starts on 1, so it could be better)

    E = E1
    beta = 0.5 # (0, 1) # for E2 + smooth
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
    Mfunc = Function(V_grid).interpolate(M)

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
        # project onto "mesh" that is being adapted (i.e., mesh(x))
        _P1 = FunctionSpace(mesh, "CG", 1) 
        _M = Function(_P1)
        #spyro.mesh_to_mesh_projection(Mfunc, _M, degree=5)
        _mesh_to_mesh_interpolation(Mfunc, _M) 
        File("Mfunc_x.pvd").write(_M)
        return _M
        
    def mask_dummy(mesh):
        return Constant(1.)

    # change the mesh coordinates to the original ones (computational mesh)
    mesh.coordinates.assign(xi)

    # make mesh and mesh_grid parallel compatible (to perform mesh-to-mesh interpolation via projection)
    mesh._parallel_compatible = {weakref.ref(mesh_grid)}
    
    # ok, adapt the mesh according to the monitor function
    step = spyro.monge_ampere_solver(mesh, monitor_function, p=2, mask=mask_dummy) #fix_boundary_nodes=fix_boundary_nodes) 
    
    # now, clear spatial index
    mesh.clear_spatial_index()
# }}}

# adapt the mesh using initial vp, if requested {{{
if AMR and ADAPT_MESH_FOR_GUESS_VP: # testing the mesh adaptation before starting the FWI
    # interpolate the inverted field on the grid/mesh
    _vp_guess = _make_vp(V, vp_guess=True) # source line is not filled with exact vp (the idea is to generate smooth monitor function)
    
    _vp_guess.dat.data[np.where(_vp_guess.dat.data[:] < 1.7)] = 1.7

    _vp_grid = Function(V_grid)
    
    #spyro.mesh_to_mesh_projection(_vp_guess, _vp_grid, degree=5) # from _vp_guess to _vp_grid
    _mesh_to_mesh_interpolation(_vp_guess, _vp_grid)

    _grad_vp_grid = Function(V_vec_grid) 
    adapt_mesh(_vp_grid, _grad_vp_grid, mesh) 
    
    #spyro.mesh_to_mesh_projection(_vp_grid, _vp_guess, degree=5)
    #File("vp_guess_after_inital_adaptation.pvd").write(_vp_guess)
    #sys.exit("exit")
#}}}

# set the initial guess (vp)
vp_guess = _make_vp(V, vp_guess=True, apply_exact_vp_over_source_line=True)  
#sys.exit("exit")

# FWI loop with or without mesh adaptation important: (if mesh changes, V changes too)
for i in range(max_loop_it): # it should be thought as re-mesh
 
    print("###### Loop iteration ="+str(i), flush=True)
    print("###### FWI iteration performed ="+str(i*max_rol_it), flush=True)

    # create sources and receivers for a given mesh and space V
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)

    # update source mask
    m = V.ufl_domain()
    W = VectorFunctionSpace(m, V.ufl_element())
    X = interpolate(m.coordinates, W)
    source_mask = np.where(X.sub(1).dat.data[:] > -0.53)
    if 0: # {{{
        ft = Function(V).interpolate(Constant(1.0))
        ft.dat.data[source_mask] = 0.0
        File("ft.pvd").write(ft)
        print(source_mask)
    #    sys.exit("exit")
    #}}}

# prepare to run FWI, set guess add control bounds to the problem {{{
    params = ROL.ParameterList(paramsDict, "Parameters")
    inner_product = L2Inner(V) # if mesh changes, V changes also

    xlo = Function(V) # lower bound
    xup = Function(V) # upper bound

    xlo.dat.data_with_halos[:] = min_vp
    xup.dat.data_with_halos[:] = max_vp

    x_lo = FeVector(xlo.vector(), inner_product)
    x_up = FeVector(xup.vector(), inner_product)
        
    bnd = ROL.Bounds(x_lo, x_up, 1.0)
    obj = Objective(V, inner_product, p_ref_recv, sources, receivers, source_mask)
#}}}

    obj.vp.dat.data_with_halos[:] = vp_guess.dat.data_ro_with_halos[:] # initial guess 
    outfile.write(vp_guess,time=i)

    print("   Minimize: starting ROL solver...", flush=True)
    
    problem = ROL.OptimizationProblem(obj, FeVector(vp_guess.vector(), inner_product), bnd=bnd)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
    
    #FIXME remove it
    File("inverted_vp.pvd").write(obj.vp)
    #sys.exit("exit")

    # interpolate the inverted field on the grid/mesh
    vp_grid = Function(V_grid)
    #spyro.mesh_to_mesh_projection(obj.vp, vp_grid, degree=5)
    _mesh_to_mesh_interpolation(obj.vp, vp_grid)
    grad_vp_grid = Function(V_vec_grid)
    
    #FIXME remove it
    #File("vp_grid_after_interp.pvd").write(vp_grid) ok 

# adapt the mesh using inverted vp, if requested {{{
    if AMR and i<max_loop_it-1: # no need to adapt the last FWI loop
       adapt_mesh(vp_grid, grad_vp_grid, mesh) 
#}}}

    # now, interpolate the inverted field on the initial guess fuction (defined now in the adpated or non-adapted mesh)
    #spyro.mesh_to_mesh_projection(vp_grid, vp_guess, degree=5)
    _mesh_to_mesh_interpolation(vp_grid, vp_guess)
   
    #FIXME remove it
    File("vp_grid_after_interp.pvd").write(vp_grid) 
    File("vp_guess_after_interp.pvd").write(vp_guess)
    sys.exit("exit")

    Ji.append(obj.J)

# write final solution and J
outfile.write(vp_guess,time=i)
if COMM_WORLD.rank == 0:
    if AMR:
        with open(r'J_with_amr.txt', 'w') as fp:
            for j in Ji:
                # write each item on a new line
                fp.write("%s\n" % str(j))
        print('Done')
    else:
        with open(r'J_no_amr.txt', 'w') as fp:
            for j in Ji:
                # write each item on a new line
                fp.write("%s\n" % str(j))
        print('Done')

