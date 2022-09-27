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
    "delay": 1.0, # FIXME check this
    "num_sources": 4, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.5, -0.01), (3.5, -0.01), 4),
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 100, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((0.1, -0.10), (3.9, -0.10), 100),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 2.5, # Final time for event 
    "dt": 0.0005,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
}
#}}}
# make vp {{{
def _make_vp(model, mesh, V, vp_guess=False, field="velocity_model"):
    
    path = "./velocity_models/elastic-marmousi-model/model/"
    if vp_guess: # interpolate from a smoothed field
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy.hdf5" 
    else: # interpolate from the exact field
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy.hdf5" 
        
    with h5py.File(fname, "r") as f:
        Zo = np.asarray(f.get(field)[()]) # original Marmousi data/domain
        nrow, ncol = Zo.shape
        Lz = model["mesh"]["Lz"]
        Lx = model["mesh"]["Lx"] 
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
# cut a small domain from the original Marmousi model {{{
def _cut_marmousi(minz, maxz, minx, maxx, smooth=False, field="velocity_model"):
   
    from SeismicMesh.sizing.mesh_size_function import write_velocity_model
    import segyio
    import math

    path = "./velocity_models/elastic-marmousi-model/model/"
    if smooth:
        #fname_marmousi = path + "MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"
        #fname_marmousi = path + "MODEL_S-WAVE_VELOCITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"
        fname_marmousi = path + "MODEL_DENSITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"
    else:
        #fname_marmousi = path + "MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"
        #fname_marmousi = path + "MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5"
        fname_marmousi = path + "MODEL_DENSITY_1.25m.segy.hdf5"

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
            #fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy"
            #fname = path + "MODEL_S-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy"
            fname = path + "MODEL_DENSITY_1.25m_small_domain_smoothed_sigma=300.segy"
        else:
            #fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy"
            #fname = path + "MODEL_S-WAVE_VELOCITY_1.25m_small_domain.segy"
            fname = path + "MODEL_DENSITY_1.25m_small_domain.segy"
        
        # save to segy format
        create_segy(Zq, fname)
        # save to hdf5 format
        #write_velocity_model(fname)
        hfname = fname +".hdf5"
        print(f"Writing velocity model: {hfname}", flush=True)
        with h5py.File(hfname, "w") as fh:
            #fh.create_dataset("velocity_model", data=Zq, dtype="f")
            fh.create_dataset("density", data=Zq, dtype="f")
            fh.attrs["shape"] = Zq.shape
            #fh.attrs["units"] = "m/s"
            fh.attrs["units"] = "g/cm3"

    
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

if False:
    # 4x2 middle of the domain
    minz = -2.0 
    maxz =  0.0
    minx = 7.5
    maxx = 11.5
    # 2x1.5 middle of the domain
    #minz = -1.95 
    #maxz = -0.45
    #minx = 8.5
    #maxx = 10.5
    _cut_marmousi(minz, maxz, minx, maxx, smooth=False)
    #_cut_marmousi(minz, maxz, minx, maxx, smooth=True)
    sys.exit("exit")

#}}}
comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 20)}
# generate mesh with SeismicMesh, if requested {{{
# number of cells per wavelenght, from Roberts et al., 2021 (GMD)
# Homegeneous case x 1.20 (see Table 3 from Roberts et al., 2021 (GMD))
# multiplying by n because we want the cells show the features
n = 1
if model['opts']['degree']   == 2:
    M = 7.02 * n
elif model['opts']['degree'] == 3:
    M = 3.96 * n
elif model['opts']['degree'] == 4:
    M = 2.67 * n
elif model['opts']['degree'] == 5:
    M = 2.03 * n 

if False:
    from SeismicMesh import get_sizing_function_from_segy, generate_mesh, Rectangle, plot_sizing_function
    from scipy.ndimage import gaussian_filter
    import segyio

    
    if model["acquisition"]["frequency"] != 3:
        sys.exit("mesh not generated, check the .segy file for 5 Hz")
    
    smooth = True # keep it True for initial model
    path = "./velocity_models/elastic-marmousi-model/model/"

    if smooth:
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy"
    else:
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy"

    bbox = (0.0, model["mesh"]["Lx"]*1000, -model["mesh"]["Lz"]*1000.0, 0.0)
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
        vtk_file = "./meshes/fwi_amr_marmousi_small_p=" + str(p) + "_M=" + str(M) + ".vtk"
        meshio.write_points_cells(
            vtk_file,
            #points[:, [1, 0]] / 1000,
            points[:] / 1000, # do not swap here
            [("triangle", cells)],
            file_format="vtk",
            binary=False
        )
        gmsh_file = "./meshes/fwi_amr_marmousi_small_p=" + str(p) + "_M=" + str(M) + ".msh"
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
if True:
    mesh_x = RectangleMesh(50, 25, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
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

    vp_exact = _make_vp(model, mesh_x, V_x, vp_guess=False)
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
model["mesh"]["meshfile"] = "./meshes/fwi_amr_marmousi_small_p=" + str(p) + "_M=" + str(M) + ".msh" 
mesh_x, V_x = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)# mesh that will be adapted

# adapt the mesh using the exact vp, if requested {{{
if False:
    #_mesh_xi, _V_xi = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)# computational mesh
    mesh_x = RectangleMesh(45, 25, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_x.coordinates.dat.data[:, 0] -= 0.0 
    mesh_x.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]
    element = spyro.domains.space.FE_method(mesh_x, model["opts"]["method"], model["opts"]["degree"])
    V_x = FunctionSpace(mesh_x, element)
   

    _mesh_xi = RectangleMesh(45, 25, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    _mesh_xi.coordinates.dat.data[:, 0] -= 0.0 
    _mesh_xi.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]
    element = spyro.domains.space.FE_method(_mesh_xi, model["opts"]["method"], model["opts"]["degree"])
    _V_xi = FunctionSpace(_mesh_xi, element)

    
    _vpi_xi = _make_vp(model, _mesh_xi, _V_xi, vp_guess=True)
    _vpf_xi = _make_vp(model, _mesh_xi, _V_xi, vp_guess=False)
    _M_xi   = Function(_V_xi)   # monitor function using vpi_xi and vpf_xi
    
    mesh_x._parallel_compatible = {weakref.ref(_mesh_xi)}

    mtype = 1
    if mtype==1:
        alpha = 4.
        _M_xi.dat.data_with_halos[:] = (_vpi_xi.dat.data_ro_with_halos[:] / _vpf_xi.dat.data_ro_with_halos[:])**alpha 
    elif mtype==2:
        max_grad_vp = norm(grad(_vpf_xi - _vpi_xi), norm_type="L2") 
        print(max_grad_vp, flush=True)
        #alpha = 1/(max_grad_vp**2)
        alpha = 5/(max_grad_vp**2)
        #_M_xi.interpolate( sqrt( 1 + alpha*inner(grad(_vpf_xi),grad(_vpf_xi)) ) )
        
        _f = _vpf_xi - _vpi_xi
        _M_xi.interpolate( sqrt( 1 + alpha*inner(grad(_f),grad(_f)) ) )
        
        File("monitor.pvd").write(_M_xi)

    def monitor_function(mesh, M_xi=_M_xi):
        # project onto "mesh" that is being adapted (i.e., mesh_x)
        P1 = FunctionSpace(mesh, "CG", 2)
        M_x = Function(P1)
        #M_x.project(M_xi) # Project from computational mesh (mesh_xi) onto physical mesh (mesh_x)
        spyro.mesh_to_mesh_projection(M_xi, M_x)
        return M_x

    receiver_z = 0.44545454545454544
    receiver_x = -0.1
    tolerance = 1e-2
    expect = mesh_x.locate_cell([receiver_z, receiver_x], tolerance=tolerance)

    method = "quasi_newton"
    tol = 1.0e-03
    #mover = MongeAmpereMover(mesh_x, monitor_function, method=method, rtol=tol)
    #step  = mover.move() # mesh_x will be adapted, so space V_x
    step = spyro.monge_ampere_solver(mesh_x, monitor_function, p=2, rtol=tol) 


    print("mesh_x adapted in "+str(step)+" steps")

    actual = mesh_x.locate_cell([receiver_z, receiver_x], tolerance=tolerance)
    
    print("expect="+str(expect)+", actual="+str(actual)) 
    
    #_vp_exact = Function(V_x).interpolate(Constant(1.)) 
    _vp_exact = _make_vp(model, mesh_x, V_x, vp_guess=False)
    File("adapted_mesh.pvd").write(_vp_exact)

    # testing in parallel
    #_mesh_xi.coordinates.dat.data_with_halos[:] = mesh_x.coordinates.dat.data_ro_with_halos[:] # update mesh_xi 
    _vp_exact_2 = _make_vp(model, _mesh_xi, _V_xi, vp_guess=False)
    File("original_mesh.pvd").write(_vp_exact_2)


    mesh_fine = RectangleMesh(60, 35, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_fine.coordinates.dat.data[:, 0] -= 0.0 
    mesh_fine.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]

    # for the exact model, use a higher-order element
    model["opts"]["degree"] = 4
    element = spyro.domains.space.FE_method(mesh_x, model["opts"]["method"], model["opts"]["degree"])
    V_fine = FunctionSpace(mesh_fine, element)

    _vp_exact_3 = _make_vp(model, mesh_fine, V_fine, vp_guess=False)
    File("fine_mesh.pvd").write(_vp_exact_3)

    sys.exit("exit")
#}}}
sources = spyro.Sources(model, mesh_x, V_x, comm)
receivers = spyro.Receivers(model, mesh_x, V_x, comm)
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

vp_guess = _make_vp(model, mesh_x, V_x, vp_guess=True)
water = np.where(vp_guess.dat.data[:] < 1.51)
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

        dJ.dat.data[water] = 0.0
        File("dJ_acoustic_original.pvd").write(dJ)
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
mesh_xi, V_xi = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)# computational mesh
vpi_xi = Function(V_xi) # to keep vp before minimizer
vpf_xi = Function(V_xi) # to keep vp after minimizer (inverted vp)
M_xi = Function(V_xi)   # monitor function using vpi_xi and vpf_xi

# parameters for Monge-Ampere solver
method = "quasi_newton"
tol = 1.0e-03
#}}}

Ji=[]
ii=[]
m_type = 2 # monitor type
adapt_mesh = False 
mesh_moved = True
outfile = File("final_vp.pvd")
max_loop_it = 10 # the number of iteration here depends on the max iteration of ROL
max_rol_it = 15
amr_freq = 75 # 15, 45, 75 
degree = None
if model["opts"]["method"]=="KMV" and model["opts"]["degree"] == 2:
    degree = 6 # alias to solve Firedrake projection

for i in range(max_loop_it):
    print("###### Loop iteration ="+str(i), flush=True) 
    print("###### FWI iteration performed ="+str(i*max_rol_it), flush=True) 
   
    # at this moment, mesh_x=mesh_xi
    if mesh_moved: # vpi_xi is updated only if mesh has changed
        vpi_xi.dat.data_with_halos[:] = xig.dat.data_ro_with_halos[:] # initial guess (mesh_xi) 
    obj.vp.dat.data_with_halos[:] = xig.dat.data_ro_with_halos[:] # initial guess (mesh_x)
    outfile.write(obj.vp,time=i)
    
    print("   Minimize: starting ROL solver...", flush=True)
    problem = ROL.OptimizationProblem(obj, FeVector(xig.vector(), inner_product), bnd=bnd)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
    
    vpf_xi.dat.data_with_halos[:] = obj.vp.dat.data_ro_with_halos[:] # inverted field (mesh_xi)

    if m_type==1:
        alpha = 3. # 4 for P2 and original dt generates numerical instabilities 
        M_xi.dat.data_with_halos[:] = (vpi_xi.dat.data_ro_with_halos[:] / vpf_xi.dat.data_ro_with_halos[:])**alpha # (mesh_xi)
    elif m_type==2:
        #e = vpf_xi - vpi_xi
        e = vpf_xi
        ne = norm(grad(e), norm_type="L2") 
        alpha = 1/(ne**2+1.e-6)
        #alpha = 1/(ne+1.e-6) # too much mesh distortion! L20
        M_xi.interpolate( sqrt( 1 + alpha*inner(grad(e), grad(e)) ) )
    
    File("monitor.pvd").write(M_xi)
  
    # ok, let's adapt mesh_x
    if adapt_mesh and i < (max_loop_it-1) and ((i+1)*max_rol_it)%amr_freq == 0: # do not adapt in the last step
        print("   Starting moving mesh...", flush=True)
        
        def monitor_function(mesh, M_xi=M_xi):
            # project onto "mesh" that is being adapted (i.e., mesh_x)
            P1 = FunctionSpace(mesh, "CG", 1)
            M_x = Function(P1)
            #M_x.project(M_xi) 
            spyro.mesh_to_mesh_projection(M_xi, M_x) # If using higher-order elements here, maybe use higher degree
            return M_x
        
        mover = MongeAmpereMover(mesh_x, monitor_function, method=method, rtol=tol, fix_boundary_nodes=True)
        step  = mover.move() # mesh_x will be adapted, so space V_x
        if step>0: 
            # ok, we have mesh movement, update vpi_xi
            print("OK, mesh has changed!")
            mesh_moved = True
            #xig.project(vpf_xi)# project vp onto the new mesh such this could be used as initial guess for next i 
            spyro.mesh_to_mesh_projection(vpf_xi, xig, degree=degree)
            mesh_xi.coordinates.dat.data_with_halos[:] = mesh_x.coordinates.dat.data_ro_with_halos[:] # update mesh_xi 
        else:
            print("Mesh is the same")
            mesh_moved = False
            xig.dat.data_with_halos[:] = obj.vp.dat.data_ro_with_halos[:] # no mesh movement, therefore mesh_x=mesh_xi
       
    else:
        print("Mesh is the same")
        mesh_moved = False
        xig.dat.data_with_halos[:] = obj.vp.dat.data_ro_with_halos[:] # no mesh movement, therefore mesh_x=mesh_xi

    ii.append(i)
    Ji.append(obj.J)

if COMM_WORLD.rank == 0:
    with open(r'J.txt', 'w') as fp:
        for j in Ji:
            # write each item on a new line
            fp.write("%s\n" % str(j))
        print('Done')
  
