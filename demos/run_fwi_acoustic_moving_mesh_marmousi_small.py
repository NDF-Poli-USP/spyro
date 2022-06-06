from firedrake import *
from scipy.optimize import * 
from movement import *
import spyro
import time
import sys
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
    "degree": 4,  # p order
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
    "Lz": 1.5,  # depth in km - always positive
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
    "delay": 1.0, # FIXME check this
    "num_sources": 1, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.6, -0.1), (1.4, -0.1), 1),
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 10, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((0.6, -0.2), (1.4, -0.2), 10),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.0, # Final time for event 
    "dt": 0.0005,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}
#}}}
# make vp {{{
def _make_vp(V, mesh, vp_guess=False):
    """creating velocity models"""
    x,z = SpatialCoordinate(mesh)
    if vp_guess: # interpolate from a smoothed field
        vp   = Function(V).interpolate(1.5 + 0.0 * x) # original one (constant)
        #vp   = Function(V).interpolate(
        #    2.5
        #    + 1 * tanh(2 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2))) # initial guess
        #)
        File("guess_vp.pvd").write(vp)
    else: # interpolate from the exact field
        vp  = Function(V).interpolate(
            2.5
            #+ 1 * tanh(20 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2))) # original one (smoother)
            + 1 * tanh(100 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2))) # sharper
        )
        File("exact_vp.pvd").write(vp)

    return vp
#}}}
# cut a small domain from the original Marmousi model {{{
def _cut_marmousi(minz, maxz, minx, maxx, smooth=False, field="velocity_model"):
   
    from SeismicMesh.sizing.mesh_size_function import write_velocity_model
    import segyio
    import math

    path = "./velocity_models/elastic-marmousi-model/model/"
    if smooth:
        fname_marmousi = path + "MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"
    else:
        fname_marmousi = path + "MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"

    with h5py.File(fname_marmousi, "r") as f:
        Zo = np.asarray(f.get(field)[()]) # original Marmousi data/domain
        nrow, ncol = Zo.shape
        Lz = 3.5  # original depth of Marmousi model
        Lx = 17.0 # original length of Marmousi model
        zo = np.linspace(-Lz, 0.0, nrow) # original Marmousi data/domain
        xo = np.linspace(0.0,  Lx, ncol) # original Marmousi data/domain
        interpolant = RegularGridInterpolator((xo, zo), np.transpose(Zo))
        #interpolant = RegularGridInterpolator((zo, xo), Zo)

        nrowq = math.ceil( nrow * (maxz-minz) / Lz )
        ncolq = math.ceil( ncol * (maxx-minx) / Lx )
        assert nrowq > 0
        assert ncolq > 0
        
        zq = np.linspace(minz, maxz, nrowq)
        xq = np.linspace(minx, maxx, ncolq)
        #zq, xq = np.meshgrid(zq, xq)
        xq, zq = np.meshgrid(xq, zq)

        #Zq = interpolant((zq, xq))
        Zq = interpolant((xq, zq))

        if smooth:
            fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy"
        else:
            fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy"
        
        # save to segy format
        create_segy(Zq, fname)
        # save to hdf5 format
        write_velocity_model(fname)
    
    if True: # plot vg? {{{
        with segyio.open(fname, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            show_vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                show_vp[:, index] = trace
    
        fig, ax = plt.subplots()
        plt.pcolormesh(show_vp, shading="auto")
        plt.title("Guess model")
        plt.colorbar(label="P-wave velocity (km/s)")
        plt.xlabel("x-direction (m)")
        plt.ylabel("z-direction (m)")
        ax.axis("equal")
        plt.show()
    #}}}

if True:
    #minz = -3.0 
    #maxz = -1.5
    #minx = 1.0
    #maxx = 3.0
    minz = -1.95 
    maxz = -0.45
    minx = 8.5
    maxx = 10.5
    _cut_marmousi(minz, maxz, minx, maxx, smooth=False)
    _cut_marmousi(minz, maxz, minx, maxx, smooth=True)
    #sys.exit("exit")

#}}}
comm = spyro.utils.mpi_init(model)
# generate mesh with SeismicMesh, if requested {{{
if False:
    from SeismicMesh import get_sizing_function_from_segy, generate_mesh, Rectangle, plot_sizing_function
    from scipy.ndimage import gaussian_filter
    import segyio

    # number of cells per wavelenght, from Roberts et al., 2021 (GMD)
    # Homegeneous case x 1.20 (see Table 3 from Roberts et al., 2021 (GMD))
    # multiplying by 4 because we want the cells show the features
    if model['opts']['degree']   == 2:
        M = 7.02 * 4
    elif model['opts']['degree'] == 3:
        M = 3.96 * 4
    elif model['opts']['degree'] == 4:
        M = 2.67 * 4
    elif model['opts']['degree'] == 5:
        M = 2.03 * 4 
    
    if model["acquisition"]["frequency"] != 3:
        sys.exit("mesh not generated, check the .segy file for 5 Hz")
    
    smooth = True # keep it True for initial model
    path = "./velocity_models/elastic-marmousi-model/model/"

    if smooth:
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy"
    else:
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy"

    bbox = (0.0, 2000.0, -1500.0, 0.0)
    rectangle = Rectangle(bbox)
    
    hmin = 15.0 
    freq = model["acquisition"]["frequency"]
    dt = model["timeaxis"]["dt"]
    ef = get_sizing_function_from_segy(
        fname,
        bbox=bbox,
        hmin=hmin,          # minimum edge length in the domain 
        units="m-s",        # the units of the seismic velocity model (forcing m/s because of a <1000 assumption) FIXME 
        wl=M,               # number of cells per wavelength for a given f_max
        freq=freq,           # f_max in hertz for which to estimate wl
        dt=dt,           # theoretical maximum stable timestep in seconds given Courant number Cr
        grade=1.0,         # maximum allowable variation in mesh size in decimal percent
        domain_pad=0.,      # the width of the domain pad in -z, +x, -x, +y, -y directions
        pad_style="edge",   # the method (`edge`, `linear_ramp`, `constant`) to pad velocity in the domain pad region
    )
    #plot_sizing_function(ef, comm)
    points, cells = generate_mesh(domain=rectangle,
                              edge_length=ef,
                              mesh_improvement=False,
                              perform_checks=False,
                              verbose=10,
                              max_iter=50,
                              r0m_is_h0=True,
                             )
    
    if comm.comm.rank == 0:
        p = model["opts"]["degree"]
        vtk_file = "./meshes/fwi_amr_marmousi_small_p=" + str(p) + ".vtk"
        meshio.write_points_cells(
            vtk_file,
            #points[:, [1, 0]] / 1000,
            points[:] / 1000, # do not swap here
            [("triangle", cells)],
            file_format="vtk",
            binary=False
        )
        gmsh_file = "./meshes/fwi_amr_marmousi_small_p=" + str(p) + ".msh"
        meshio.write_points_cells(
            gmsh_file,
            points[:] / 1000, # do not swap here
            [("triangle", cells)],
            file_format="gmsh22",
            binary=False
        )
 
    sys.exit("exit")
#}}}

# run exact model with a finer mesh {{{
if False:
    mesh_x = RectangleMesh(40, 30, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm)
    mesh_x.coordinates.dat.data[:, 0] -= 0.0 
    mesh_x.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]

    # for the exact model, use a higher-order element
    p = model["opts"]["degree"] # to keep the order defined by the user
    model["opts"]["degree"] = 4
    element = spyro.domains.space.FE_method(mesh_x, model["opts"]["method"], model["opts"]["degree"])
    V_x = FunctionSpace(mesh_x, element)

    sources = spyro.Sources(model, mesh_x, V_x, comm)
    receivers = spyro.Receivers(model, mesh_x, V_x, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

    # FIXME COME BACK HERE
    vp_exact = _make_vp(V_x, mesh_x, vp_guess=False)

    print("Starting forward computation of the exact model",flush=True) 
    start = time.time()
    p_exact, p_exact_at_recv = spyro.solvers.forward(
        model, mesh_x, comm, vp_exact, sources, wavelet, receivers, output=False
    )
    end = time.time()
    print(round(end - start,2),flush=True)
    File("p_exact.pvd").write(p_exact[-1])

    # ok, reset to the original order
    model["opts"]["degree"] = p
    print(model["opts"]["degree"])
    #sys.exit("exit")
#}}}

# now, prepare to run the FWI with a coarser mesh

p = model["opts"]["degree"]
model["mesh"]["meshfile"] = "./meshes/marmousi_amr_small_p=" + str(p) + ".msh" 
mesh_x, V_x = spyro.io.read_mesh(model, comm) # mesh that will be adapted
# adapt the mesh using the exact vp, if requested {{{
if False:
    # FIXME come back here 
    _mesh_xi, _V_xi = spyro.io.read_mesh(model, comm) # computational mesh
    _vpi_xi = _make_vp(_V_xi, _mesh_xi, vp_guess=True)
    _vpf_xi = _make_vp(_V_xi, _mesh_xi, vp_guess=False)
    _M_xi   = Function(_V_xi)   # monitor function using vpi_xi and vpf_xi

    alpha = 1.
    _M_xi.dat.data_with_halos[:] = (_vpi_xi.dat.data_ro_with_halos[:] / _vpf_xi.dat.data_ro_with_halos[:])**alpha 
    mesh_x._parallel_compatible = {weakref.ref(_mesh_xi)}
    #print(mesh_x._parallel_compatible)
    #print(_mesh_xi._parallel_compatible)
    #sys.exit("exit")

    def monitor_function(mesh, M_xi=_M_xi):
        # project onto "mesh" that is being adapted (i.e., mesh_x)
        P1 = FunctionSpace(mesh, "CG", 1)
        M_x = Function(P1)
        M_x.project(M_xi) # Project from computational mesh (mesh_xi) onto physical mesh (mesh_x)
        return M_x

    method = "quasi_newton"
    tol = 1.0e-03
    mover = MongeAmpereMover(mesh_x, monitor_function, method=method, rtol=tol)
    step  = mover.move() # mesh_x will be adapted, so space V_x

    print("mesh_x adapted in "+str(step)+" steps")

    _vp_exact = _make_vp(V_x, mesh_x, vp_guess=False)
    File("adapted_mesh.pvd").write(_vp_exact)
    sys.exit("exit")
#}}}
sources = spyro.Sources(model, mesh_x, V_x, comm)
receivers = spyro.Receivers(model, mesh_x, V_x, comm)
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

vp_guess = _make_vp(V_x, mesh_x, vp_guess=True)
#sys.exit("exit")

class L2Inner(object): #{{{
    # used in the gradient norm computation
    def __init__(self):
        self.A = assemble( TrialFunction(V_x) * TestFunction(V_x) * dx, mat_type="matfree")
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
        self.vp = Function(V_x)
        self.misfit = 0.0
        self.J = 0.0
        self.p_exact_recv = p_exact_at_recv 
        self.sources = sources
        self.receivers = receivers

    def value(self, x, tol):
        """Compute the functional"""
        J_total = np.zeros((1))
        self.p_guess, p_guess_recv = spyro.solvers.forward(model, 
                                                           mesh_x, 
                                                           comm, 
                                                           self.vp, 
                                                           self.sources, 
                                                           wavelet, 
                                                           self.receivers)
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
        #plt.savefig('/home/santos/Desktop/FWI_acoustic.png')
        plt.close()
        
        self.J = J_total[0]
        return J_total[0]

    def gradient(self, g, x, tol):
        """Compute the gradient of the functional"""
        dJ = Function(V_x, name="gradient")
        dJ_local = spyro.solvers.gradient(model, 
                                          mesh_x, 
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

        #File("dJ_acoustic_gaussian_all.pvd").write(dJ)
        #File("dJ_acoustic_gaussian.pvd").write(dJ)
        File("dJ_acoustic_original.pvd").write(dJ)
        #sys.exit("exit")
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        self.vp.assign(Function(V_x, x.vec, name="vp"))

#}}}
# ROL parameters definition {{{
paramsDict = {
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS', # Line Search Quasi-Newton Method 
            'Maximum Storage': 5, #(20==10)
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
                #'Type': 'Quasi-Newton Method' # works fine, but it could be better (must use L-BFGS)
                'Type': 'Steepest descent' # it worked better than Quasi-Newton Method (it ignores L-BFGS)
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

xig = Function(V_x) # initial guess
xlo = Function(V_x) # lower bound
xup = Function(V_x) # upper bound

xig.dat.data[:] = vp_guess.dat.data[:] 
xlo.dat.data[:] = min(vp_exact.dat.data[:])
xup.dat.data[:] = max(vp_exact.dat.data[:])

x_lo = FeVector(xlo.vector(), inner_product)
x_up = FeVector(xup.vector(), inner_product)

bnd = ROL.Bounds(x_lo, x_up, 1.0)
obj = Objective(inner_product)
#}}}
# prepare monitor function {{{
mesh_xi, V_xi = spyro.io.read_mesh(model, comm) # computational mesh

vpi_xi = Function(V_xi) # to keep vp before minimizer
vpf_xi = Function(V_xi) # to keep vp after minimizer (inverted vp)
M_xi = Function(V_xi)   # monitor function using vpi_xi and vpf_xi

# parameters for Monge-Ampere solver
method = "quasi_newton"
tol = 1.0e-03
#}}}

Ji=[]
ii=[]
adapt_mesh = False
mesh_moved = True
outfile = File("final_vp.pvd")
for i in range(10): #FIXME define it better
    print("Loop iteration="+str(i), flush=True) 
   
    # at this moment, mesh_x=mesh_xi
    if mesh_moved:
        vpi_xi.dat.data[:] = xig.dat.data[:] # initial guess (mesh_xi) # vpi_xi is updated only if mesh has changed
    obj.vp.dat.data[:] = xig.dat.data[:] # initial guess (mesh_x)
    outfile.write(obj.vp,time=i)
    
    print("   Minimize: starting ROL solver...", flush=True)
    problem = ROL.OptimizationProblem(obj, FeVector(xig.vector(), inner_product), bnd=bnd)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
    
    #FIXME here and elsewhere: use dat.data_ro_with_halos
    vpf_xi.dat.data[:] = obj.vp.dat.data[:] # inverted field (mesh_xi)

    alpha = 1. 
    M_xi.dat.data[:] = (vpi_xi.dat.data[:] / vpf_xi.dat.data[:])**alpha # (mesh_xi) 
    File("monitor.pvd").write(M_xi)
  
    # FIXME maybe do not adapt the last step
    # ok, let's adapt mesh_x
    if adapt_mesh: #and max(abs(M_xi.dat.data[:]-1)) > 0.15 # FIXME define this limit better, or remove it
        print("   Starting moving mesh...", flush=True)
        print("   monitor diff="+str(max(abs(M_xi.dat.data[:]-1))))
        
        def monitor_function(mesh, M_xi=M_xi):
            # project onto "mesh" that is being adapted (i.e., mesh_x)
            P1 = FunctionSpace(mesh, "CG", 1)
            M_x = Function(P1)
            M_x.project(M_xi)
            return M_x
        
        mover = MongeAmpereMover(mesh_x, monitor_function, method=method, rtol=tol)
        step  = mover.move() # mesh_x will be adapted, so space V_x
        if step>0: 
            # ok, we have mesh movement, update vpi_xi
            print("OK, mesh has changed!")
            mesh_moved = True
            xig.project(vpf_xi)# project vp onto the new mesh such this could be used as initial guess for next i 
            mesh_xi.coordinates.dat.data[:] = mesh_x.coordinates.dat.data[:] # update mesh_xi FIXME improve this
            
            # FIXME maybe there is another way to update the tabulations of the source and the receivers
            obj.receivers = spyro.Receivers(model, mesh_x, V_x, comm)
            obj.sources = spyro.Sources(model, mesh_x, V_x, comm) 
        else:
            print("Mesh is the same")
            mesh_moved = False
            xig.dat.data[:] = obj.vp.dat.data[:] # no mesh movement, therefore mesh_x=mesh_xi
       
    else:
        xig.dat.data[:] = obj.vp.dat.data[:] # no mesh movement, therefore they have the same space/mesh (mesh_x=mesh_xi)

    ii.append(i)
    Ji.append(obj.J)

if COMM_WORLD.rank == 0:
    with open(r'J.txt', 'w') as fp:
        for j in Ji:
            # write each item on a new line
            fp.write("%s\n" % str(j))
        print('Done')
  
