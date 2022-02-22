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
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
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
    "status": True,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.25,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
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

# build or read a mesh {{{
if 0:
    mesh = RectangleMesh(45, 45, model["mesh"]["Lx"], model["mesh"]["Lz"]-0.5, diagonal="crossed", comm=comm.comm)
    mesh.coordinates.dat.data[:, 0] -= 0.0 # PML size
    mesh.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"]-0.5
    element = spyro.domains.space.FE_method(
        mesh, model["opts"]["method"], model["opts"]["degree"]
    )
    V = VectorFunctionSpace(mesh, element)
    P = VectorFunctionSpace(mesh, element, dim=2) # for lambda and mu inside PyROL
    H = FunctionSpace(mesh, element)
else:
    mesh, H = spyro.io.read_mesh(model, comm) #FIXME update io read mesh for elastic
    element = spyro.domains.space.FE_method(
        mesh, model["opts"]["method"], model["opts"]["degree"]
    )
    V = VectorFunctionSpace(mesh, element)
    P = VectorFunctionSpace(mesh, element, dim=2) # for lambda and mu inside PyROL
#}}}
# make lamb and mu {{{
def _make_elastic_parameters(H, mesh, guess=False):
    """creating velocity models"""
    x,z = SpatialCoordinate(mesh)
    _cp = 1.5
    _cs = 1.
    _rho = 1. 
    _mu = (_cs**2)*_rho 
    _lamb = (_cp**2)*_rho-2*_mu
    _cp = 2.5 # cp=3.5 
    _cs = 1.5  # cs=2.0
    _mu_max = (_cs**2)*_rho 
    _lamb_max = (_cp**2)*_rho-2*_mu_max
    if guess:
        lamb = Function(H).interpolate(_lamb + 0.0 * x)
        mu   = Function(H).interpolate(_mu + 0.0 * x)
        File("guess_lamb.pvd").write(lamb)
        File("guess_mu.pvd").write(mu)
    else:
        lamb  = Function(H).interpolate(
            0.5*(_lamb_max+_lamb)
            + 0.5*(_lamb_max-_lamb) * tanh(20 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2)))
        )
        mu  = Function(H).interpolate(
            0.5*(_mu_max+_mu)
            + 0.5*(_mu_max-_mu) * tanh(20 * (0.125 - sqrt(( x - 1) ** 2 + (z + 0.5) ** 2)))
        )
        File("exact_lamb.pvd").write(lamb)
        File("exact_mu.pvd").write(mu)

    rho = Constant(_rho)
    return lamb, mu, rho
#}}}

# FWI better well when sigma_x>500 (e.g., 1000 or 2000) (sigma_x defines the source and receivers)
lamb_guess, mu_guess, rho = _make_elastic_parameters(H, mesh, guess=True) 
lamb_exact, mu_exact, _ = _make_elastic_parameters(H, mesh, guess=False) 
#sys.exit("exit")

sources = spyro.Sources(model, mesh, H, comm)
receivers = spyro.Receivers(model, mesh, H, comm)
wavelet = spyro.full_ricker_wavelet(
                dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
            )

print("Starting forward computation",flush=True)
start = time.time()
u_exact, uz_exact, ux_exact, uy_exact = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, receivers, output=True
)
end = time.time()
print(round(end - start,2), flush=True)
print("FIXME: check why gnorm in the first step (ite=0) is different",flush=True)
sys.exit("exit")
File("u_exact.pvd").write(u_exact[-1])
if 0: # print initial guess {{{
    u_initial_guess, _, _, _ = spyro.solvers.forward_elastic_waves(
        model, mesh, comm, rho, lamb_guess, mu_guess, sources, wavelet, receivers, output=False
    )
    File("u_initial_guess.pvd").write(u_initial_guess[-1])
    sys.exit("exit")
#}}}
class L2Inner(object): #{{{
    def __init__(self):
        self.A = assemble( TrialFunction(H) * TestFunction(H) * dx, mat_type="matfree")
        self.Ap = as_backend_type(self.A).mat()
        self.ulp = Function(H)
        self.ump = Function(H)
        self.vlp = Function(H)
        self.vmp = Function(H)

    def eval(self, u, v):
        self.ulp.dat.data[:] = u.dat.data[:,0] # lambda
        self.vlp.dat.data[:] = v.dat.data[:,0] # lambda
        self.ump.dat.data[:] = u.dat.data[:,1] # mu
        self.vmp.dat.data[:] = v.dat.data[:,1] # mu
        ulpet = as_backend_type(self.ulp.vector()).vec() # lambda
        vlpet = as_backend_type(self.vlp.vector()).vec() # lambda
        umpet = as_backend_type(self.ump.vector()).vec() # mu
        vmpet = as_backend_type(self.vmp.vector()).vec() # mu
        A_ul = self.Ap.createVecLeft() # lambda
        A_um = self.Ap.createVecLeft() # mu
        self.Ap.mult(ulpet, A_ul) # lambda
        self.Ap.mult(umpet, A_um) # mu
        return vlpet.dot(A_ul) + vmpet.dot(A_um)
#}}}
class ObjectiveElastic(ROL.Objective): #{{{
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.u_guess = None # Z,X displacements, entire domain
        self.uz_guess = None
        self.ux_guess = None
        self.uz_exact = uz_exact # Z displacements, receiver
        self.ux_exact = ux_exact # X displacements, receiver
        self.misfit_uz = None
        self.misfit_ux = None
        self.lamb = Function(H) 
        self.mu = Function(H)

    def run_forward(self):
        J_scale = sqrt(1.e14) 
        self.u_guess, self.uz_guess, self.ux_guess, _ = spyro.solvers.forward_elastic_waves(
            model, mesh, comm, rho, self.lamb, self.mu, sources, wavelet, receivers, output=False
        )
        # compute the residual (misfit) 
        self.misfit_uz = J_scale * spyro.utils.evaluate_misfit(model, self.uz_guess, self.uz_exact)
        self.misfit_ux = J_scale * spyro.utils.evaluate_misfit(model, self.ux_guess, self.ux_exact)

    def value(self, x, tol):
        #print("Starting forward computation - elastic waves")
        """Compute the functional"""
        self.run_forward()
        J_total = np.zeros((1))
        J_total[0] += spyro.utils.compute_functional(model, self.misfit_uz)
        J_total[0] += spyro.utils.compute_functional(model, self.misfit_ux)
        J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size # paralelismo ensemble (fontes)
        if comm.comm.size > 1:
            J_total[0] /= comm.comm.size # paralelismo espacial

        ue=[]
        ug=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            ue.append(self.ux_exact[ti][rn])
            ug.append(self.ux_guess[ti][rn])
        plt.title("u_z")
        plt.plot(ue,label='exact')
        plt.plot(ug,label='guess')
        plt.legend()
        plt.savefig('/home/santos/Desktop/FWI.png')
        plt.close()

        File("u_guess.pvd").write(self.u_guess[-1])
        return J_total[0]

    def gradient(self, g, x, tol):
        #print("Starting gradient computation - elastic waves")
        """Compute the gradient of the functional"""
        if not type(self.misfit_uz)==list:
            self.run_forward()
        
        misfit_uy = []
        dJdl = Function(H, name="dJdl")
        dJdm = Function(H, name="dJdm")
        dJdl_local, dJdm_local = spyro.solvers.gradient_elastic_waves(
            model, mesh, comm, rho, self.lamb, self.mu, 
            receivers, self.u_guess, self.misfit_uz, self.misfit_ux, misfit_uy, output=False,
        )
        if comm.ensemble_comm.size > 1:
            comm.allreduce(dJdl_local, dJdl)
            comm.allreduce(dJdm_local, dJdm)
        else:
            dJdl = dJdl_local
            dJdm = dJdm_local
        dJdl /= comm.ensemble_comm.size
        dJdm /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            dJdl /= comm.comm.size
            dJdm /= comm.comm.size

        #File("dJdl_elastic_original.pvd").write(dJdl)
        #File("dJdm_elastic_original.pvd").write(dJdm)
        File("dJdl_elastic_pointsource.pvd").write(dJdl)
        File("dJdm_elastic_pointsource.pvd").write(dJdm)
        #sys.exit("stop")
        g.scale(0)
        g.vec.dat.data[:,0] += dJdl.dat.data[:] # FIXME - is employed for gaussian function
        g.vec.dat.data[:,1] += dJdm.dat.data[:] # FIXME - is emploued for gaussian function

    def update(self, x, flag, iteration):
        self.lamb.dat.data[:] = x.vec.dat.data[:,0]
        self.mu.dat.data[:] = x.vec.dat.data[:,1]
#}}}
# ROL parameters definition {{{
paramsDict = {
    'General': {
        'Projected Gradient Criticality Measure': False, # does not make difference
        'Recompute Objective Function': False, # no need
        'Secant': {
            #'Type': 'Barzilai-Borwein', # Line Search Quasi-Newton Method (works similar to L-BFGS)
            #'Type': 'Limited-Memory DFP', # Line Search Quasi-Newton Method (works similar to L-BFGS)
            #'Type': 'Limited-Memory SR1', # Line Search Quasi-Newton Method (works similar than L-BFGS)
            'Type': 'Limited-Memory BFGS', # Line Search Quasi-Newton Method 
            'Maximum Storage': 5, #(20==10)
            'Barzilai-Borwein Type': 2
        }
    },
    'Step': {
        'Type': 'Line Search', # the only one that works
        #'Type': 'Augmented Lagrangian', # does not work with bound values, so it turns on Trust-Region Solver
        #'Type': 'Bundle', # does not work with bound values, so it turns on Dogleg Trust-Region Solver
        #'Type': 'Interior Point', # does not work
        #'Type': 'Primal Dual Active Set', # too slow!
        #'Type': 'Trust Region', # does not show improvement with Dogleg Solver
        #'Type': 'Moreau-Yosida Penalty', # too slow!
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
                #'Type': 'Nonlinear CG' # Quasi-Newton Method works better
            },
            'Curvature Condition': {
                #'Type': 'Wolfe Conditions' # similar to Strong Wolfe Conditions
                'Type': 'Strong Wolfe Conditions', # works fine
                #'Type': 'Generalized Wolfe Conditions' # similar to Strong Wolfe Conditions
                #'Type': 'Approximate Wolfe Conditions' # similar to Strong Wolfe Conditions
                #'Type': 'Goldstein Conditions' # similar to Strong Wolfe Conditions
            }
        },
        'Trust Region': {
            #'Subproblem Solver': 'Cauchy Point', # does not show improvement
            #'Subproblem Solver': 'Dogleg', # does not show improvement
            'Subproblem Solver': 'Double Dogleg', # improvement is similar to Line Search (Quasi-Newton Method)
            #'Subproblem Solver': 'Truncated CG', # too slow
            #'Subproblem Model': 'Colemanli' # does not work
            'Subproblem Model': 'Kelley Sachs'
        },
        'Moreau-Yosida Penalty': {
            'Initial Penalty Parameter': 1.,
            'Subproblem': {
                'Iteration Limit': 10,
            }
        },
        'Augmented Lagrangian': {
            'Initial Penalty Parameter'               : 1.e2,
            'Penalty Parameter Growth Factor'         : 2,
            'Minimum Penalty Parameter Reciprocal'    : 0.1,
            'Initial Optimality Tolerance'            : 1.0,
            'Optimality Tolerance Update Exponent'    : 1.0,
            'Optimality Tolerance Decrease Exponent'  : 1.0,
            'Initial Feasibility Tolerance'           : 1.0,
            'Feasibility Tolerance Update Exponent'   : 0.1,
            'Feasibility Tolerance Decrease Exponent' : 0.9,
            'Print Intermediate Optimization History' : True,
            'Subproblem Step Type'                    : 'Line Search',
            'Subproblem Iteration Limit'              : 10
        }
    },
    "Status Test": {
        "Gradient Tolerance": 1e-15,
        'Relative Gradient Tolerance': 1e-10,
        "Step Tolerance": 1.0e-15,
        'Relative Step Tolerance': 1e-15,
        "Iteration Limit": 200
    },
}
#}}}
# prepare to run FWI, set guess add control bounds to the problem (uses more RAM)
params = ROL.ParameterList(paramsDict, "Parameters")
inner_product = L2Inner()

x0 = Function(P)
# lamb
x0.dat.data[:,0] = lamb_guess.dat.data[:]
# mu
x0.dat.data[:,1] = mu_guess.dat.data[:]

xlo = Function(P)
# lamb
xlo.dat.data[:,0] = min(lamb_exact.dat.data[:])
# mu
xlo.dat.data[:,1] = min(mu_exact.dat.data[:])

xup = Function(P)
# lamb
xup.dat.data[:,0] = max(lamb_exact.dat.data[:])
# mu
xup.dat.data[:,1] = max(mu_exact.dat.data[:])

opt = FeVector(x0.vector(), inner_product)
x_lo = FeVector(xlo.vector(), inner_product)
x_up = FeVector(xup.vector(), inner_product)

bnd = ROL.Bounds(x_lo, x_up, 1.0)

obj = ObjectiveElastic(inner_product)

#Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions (must define "line search" in params)
problem = ROL.OptimizationProblem(obj, opt, bnd=bnd)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
#algo = ROL.Algorithm("Line Search", params) # step size is defined here 
#algo.run(opt, obj, bnd)

#Augmented Lagrangian solver
#Subproblem Solver: Line Search
#algo = ROL.Algorithm("Augmented Lagrangian", params) # NOT GOOD!
#algo.run(opt, obj, bnd)

# summary of the test:
# Line search: 
#   Function Evaluation Limit = 41
#   Type: Cubic Interpolation with Strong Wolf Condition worked better
#   Descent Method:   
#       Steepest Descent (as known as Gradient Descent): seemed to work better than Quasi-Newton
#       Quasi-Newton with L-BFGS to estimate the Hessian (same used in M1QN3): the default approach

# From ROL documentation (https://trilinos.github.io/rol.html)
# ROL::EStep
#   AUGMENTEDLAGRANGIAN
#   BUNDLE
#   COMPOSITESTEP
#   LINESEARCH
#   MOREAUYOSIDAPENALTY
#   PRIMALDUALACTIVESET
#   TRUSTREGION
#   INTERIORPOINT 
#   FLETCHER 

# ROL::EDescent (Enumeration of descent direction types, used in Line Search, for example)
#   STEEPEST
#   NONLINEARCG
#   SECANT (LBFGS, LDFP, LSR1, BARZILAIBORWEIN)
#   NEWTON
#   NEWTONKRYLOV
#   SECANTPRECOND

# ROL::ELineSearch
#   ITERATIONSCALING 
#   PATHBASEDTARGETLEVEL
#   BACKTRACKING 
#   BISECTION 
#   GOLDENSECTION 
#   CUBICINTERP 
#   BRENTS 


# Problem Type B (bounded):
#   Line Search
#   Trust Region
#   Moreau-Yosida Penalty
#   Primal Dual Active Set
#   Interior Point
   
File("final_lamb_elastic.pvd").write(obj.lamb)
File("final_mu_elastic.pvd").write(obj.mu)

