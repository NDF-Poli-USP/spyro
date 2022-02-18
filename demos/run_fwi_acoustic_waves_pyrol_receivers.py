from firedrake import *
from scipy.optimize import * 
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import meshio
import SeismicMesh
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
#from ..domains import quadrature, space

#parameters from Daiane
model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG", # Equi or KMV #FIXME it will be removed
    "degree": 1,  # p order
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
    "Lz": 2.0,  # depth in km - always positive
    "Lx": 2.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/square.msh",
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
    "frequency": 7.0, 
    "delay": 1.0, # FIXME check this
    "num_sources": 4, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.6, -0.1), (1.4, -0.1), 4),
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 10, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((0.6, -0.2), (1.4, -0.2), 10),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.0, # Final time for event 
    "dt": 0.001,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

if 1: 
    mesh = RectangleMesh(45, 45, model["mesh"]["Lx"], model["mesh"]["Lz"]-0.5, diagonal="crossed", comm=comm.comm)
    mesh.coordinates.dat.data[:, 0] -= 0.0 # PML size
    mesh.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]-0.5
    element = spyro.domains.space.FE_method(
        mesh, model["opts"]["method"], model["opts"]["degree"]
    )
    V = FunctionSpace(mesh, element)
else:
    mesh, V = spyro.io.read_mesh(model, comm)

# make vp {{{
def _make_vp(V, mesh, vp_guess=False):
    """creating velocity models"""
    x,z = SpatialCoordinate(mesh)
    if vp_guess:
        vp   = Function(V).interpolate(1.5 + 0.0 * x)
        File("guess_vp.pvd").write(vp)
    else:
        vp  = Function(V).interpolate(
            2.5
            + 1 * tanh(20 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2)))
            # 5.0 + 0.5 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
        )
        File("exact_vp.pvd").write(vp)

    return vp
#}}}

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
                dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
            )

vp_exact = _make_vp(V,mesh,vp_guess=False) #exact
vp_guess =_make_vp(V,mesh,vp_guess=True)
#sys.exit("exit")

print("Starting forward computation",flush=True)
start = time.time()
p_exact, p_exact_at_recv = spyro.solvers.forward(
    model, mesh, comm, vp_exact, sources, wavelet, receivers, output=True
)
end = time.time()
print(round(end - start,2),flush=True)
File("p_exact.pvd").write(p_exact[-1])
#sys.exit("exit")
#if comm.ensemble_comm.rank == 0:
#    print(round(end - start,2),flush=True)
#    File("p_exact.pvd").write(p_exact[-1])
    #sys.exit("exit")
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
        self.p_exact_recv = p_exact_at_recv 

    def value(self, x, tol):
        """Compute the functional"""
        J_total = np.zeros((1))
        self.p_guess, p_guess_recv = spyro.solvers.forward(model,mesh,comm,self.vp,sources,wavelet,receivers)
        self.misfit = spyro.utils.evaluate_misfit(model, p_guess_recv, self.p_exact_recv)
        J_total[0] += spyro.utils.compute_functional(model, self.misfit, vp=self.vp)
        J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size # paralelismo ensemble (fontes)
        if comm.comm.size > 1: 
            J_total[0] /= comm.comm.size # paralelismo espacial

        #if comm.ensemble_comm.rank == 0:
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

        return J_total[0]

    def gradient(self, g, x, tol):
        """Compute the gradient of the functional"""
        dJ = Function(V, name="gradient")
        dJ_local = spyro.solvers.gradient(model,mesh,comm,self.vp,receivers,self.p_guess,self.misfit)
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
        sys.exit("exit")
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
        "Iteration Limit": 100
    },
}
#}}}
# prepare to run FWI, set guess add control bounds to the problem (uses more RAM)
params = ROL.ParameterList(paramsDict, "Parameters")
inner_product = L2Inner()

x0 = Function(V)
xlo = Function(V)
xup = Function(V)

x0.dat.data[:]  = vp_guess.dat.data[:] 
#xlo.dat.data[:] = min(vp_exact.dat.data[:])
#xup.dat.data[:] = max(vp_exact.dat.data[:])
xlo.dat.data[:] = 1.5 
xup.dat.data[:] = 3.5
#if comm.ensemble_comm.rank == 0:
#    File("xlo.pvd").write(xlo)
#    File("xup.pvd").write(xup)
    #sys.exit('exit')

opt = FeVector(x0.vector(), inner_product)
x_lo = FeVector(xlo.vector(), inner_product)
x_up = FeVector(xup.vector(), inner_product)

bnd = ROL.Bounds(x_lo, x_up, 1.0)

obj = Objective(inner_product)

#Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions (must define "line search" in params)
problem = ROL.OptimizationProblem(obj, opt, bnd=bnd)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()

#if comm.ensemble_comm.rank == 0:
#    File("final_vp.pvd").write(obj.vp)
File("final_vp.pvd").write(obj.vp)

