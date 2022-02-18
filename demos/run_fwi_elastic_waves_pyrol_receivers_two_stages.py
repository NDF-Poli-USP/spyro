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
#from ..domains import quadrature, space

model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG", # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

model["mesh"] = {
    "Lz": 0.8,  # depth in km - always positive
    "Lx": 0.8,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.25,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(0.4, 0.6)],
    "frequency": 5.0, #10.0
    "delay": 0.0,
    "num_receivers": 1,
    "receiver_locations": spyro.create_transect(
        (0.01, 0.55), (0.79, 0.55), 100 # with circle
        #(0.01, 0.25), (0.79, 0.25), 100 for no circle
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.001*700, # Final time for event (for test 7)
    "dt": 0.00100,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

mesh = RectangleMesh(85, 85, model["mesh"]["Lz"], model["mesh"]["Lx"], diagonal="crossed")

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)
#V = FunctionSpace(mesh, element)
V = VectorFunctionSpace(mesh, element)
P = VectorFunctionSpace(mesh, element, dim=2) # for lambda and mu inside PyROL
H = FunctionSpace(mesh, element)

mu_exact    = 1./4.
lamb_exact  = 1./2.
rho_exact   = 1.0

# FWI better well when sigma_x>500 (e.g., 1000 or 2000) (sigma_x defines the source and receivers)

z, x = SpatialCoordinate(mesh)
lamb = Function(H).interpolate(
    lamb_exact + 0.5
    + 0.5 * tanh(100.0 * (0.1 - sqrt((z - 0.4) ** 2 + (x - 0.4) ** 2)))
)
mu = Function(H).interpolate(
    mu_exact + 0.5
    + 0.5 * tanh(100.0 * (0.1 - sqrt((z - 0.4) ** 2 + (x - 0.4) ** 2)))
)
File("exact_lamb.pvd").write(lamb)
File("exact_mu.pvd").write(mu)
#sys.exit('exit')

rho = Constant(rho_exact) 

sources = spyro.Sources(model, mesh, H, comm)
receivers = spyro.Receivers(model, mesh, H, comm)
wavelet = spyro.full_ricker_wavelet(
                dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
            )

print("Starting forward computation")
start = time.time()
u_exact, uz_exact, ux_exact, uy_exact = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=False
)
end = time.time()
print(round(end - start,2))
#sys.exit("exit")
File("u_exact.pvd").write(u_exact[-1])

class L2Inner(object): #{{{
    def __init__(self):
        self.A = assemble( TrialFunction(H) * TestFunction(H) * dx, mat_type="matfree")
        self.Ap = as_backend_type(self.A).mat()

    def eval(self, u, v):
        upet = as_backend_type(u).vec()
        vpet = as_backend_type(v).vec()
        Au = self.Ap.createVecLeft()
        self.Ap.mult(upet, Au)
        return vpet.dot(Au)
#}}}
class ObjectiveElastic(ROL.Objective): #{{{
    def __init__(self, inner_product, var_flag, lamb0, mu0):
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
        #self.lamb.dat.data[:] = lamb0.dat.data[:]
        self.lamb.assign(lamb0)
        self.mu = Function(H)
        #self.mu.dat.data[:] = mu0.dat.data[:]
        self.mu.assign(mu0)
        self.var_flag = var_flag # lambda == 0, mu ==1

    def run_forward(self):
        J_scale = sqrt(1.e14) 
        self.u_guess, self.uz_guess, self.ux_guess, _ = spyro.solvers.forward_elastic_waves(
            model, mesh, comm, rho, self.lamb, self.mu, sources, wavelet, receivers, output=False
        )
        # compute the residual (misfit) 
        self.misfit_uz = J_scale * spyro.utils.evaluate_misfit(model, self.uz_guess, self.uz_exact)
        self.misfit_ux = J_scale * spyro.utils.evaluate_misfit(model, self.ux_guess, self.ux_exact)

    def value(self, x, tol):
        """Compute the functional"""
        self.run_forward()
        # compute functional
        J = 0
        J+= spyro.utils.compute_functional(model, self.misfit_uz)
        J+= spyro.utils.compute_functional(model, self.misfit_ux)

        ue=[]
        ug=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        #rn = 0
        rn = 25
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
        print(J)

        return J

    def gradient(self, g, x, tol):
        """Compute the gradient of the functional"""
        if not type(self.misfit_uz)==list:
            self.run_forward()
        
        misfit_uy = []
        dJdl, dJdm = spyro.solvers.gradient_elastic_waves(
            model, mesh, comm, rho, self.lamb, self.mu, 
            receivers, self.u_guess, self.misfit_uz, self.misfit_ux, misfit_uy, output=False,
        )
        File("dJdl_elastic.pvd").write(dJdl)
        File("dJdm_elastic.pvd").write(dJdm)
        g.scale(0)
        if self.var_flag==0: # lambda
            g.vec.dat.data[:] -= dJdl.dat.data[:]
        else: # mu
            g.vec.dat.data[:] -= dJdm.dat.data[:]

    def update(self, x, flag, iteration):
        if self.var_flag==0: # lambda
            self.lamb.dat.data[:] = x.vec.dat.data[:]
        else: # mu
            self.mu.dat.data[:] = x.vec.dat.data[:]
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
            'Function Evaluation Limit': 10,
            'Sufficient Decrease Tolerance': 1e-4,
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
        "Iteration Limit": 5 
    },
}
#}}}

# prepare to run FWI, set guess add control bounds to the problem (uses more RAM)
lamb_smooth = Function(H).interpolate(
    lamb_exact + 0.5
    + 0.5 * tanh(20.0 * (0.1 - sqrt((z - 0.4) ** 2 + (x - 0.4) ** 2)))
)
mu_smooth = Function(H).interpolate(
    mu_exact + 0.5
    + 0.5 * tanh(20.0 * (0.1 - sqrt((z - 0.4) ** 2 + (x - 0.4) ** 2)))
)
File("smoothed_lamb.pvd").write(lamb_smooth)
File("smoothed_mu.pvd").write(mu_smooth)
#sys.exit('exit')

params = ROL.ParameterList(paramsDict, "Parameters")
inner_product = L2Inner()

lamb0 = Function(H)
lamb1 = Function(H)
mu0 = Function(H)
mu1 = Function(H)

# create bounds for lambda
xlo_lamb = Function(H)
xup_lamb = Function(H)
xlo_lamb.dat.data[:] = min(lamb.dat.data[:])
xup_lamb.dat.data[:] = max(lamb.dat.data[:])

x_lo_lamb = FeVector(xlo_lamb.vector(), inner_product)
x_up_lamb = FeVector(xup_lamb.vector(), inner_product)
bnd_lamb = ROL.Bounds(x_lo_lamb, x_up_lamb, 1.0)

# create bounds for mu
xlo_mu = Function(H)
xup_mu = Function(H)
xlo_mu.dat.data[:] = min(mu.dat.data[:])
xup_mu.dat.data[:] = max(mu.dat.data[:])
x_lo_mu = FeVector(xlo_mu.vector(), inner_product)
x_up_mu = FeVector(xup_mu.vector(), inner_product)
bnd_mu = ROL.Bounds(x_lo_mu, x_up_mu, 1.0)
   
# create inital conditions
lamb0.dat.data[:] = lamb_smooth.dat.data[:]
mu0.dat.data[:] = mu_smooth.dat.data[:]

for i in range(4): 
    # invert for lambda 
    obj_lamb = ObjectiveElastic(inner_product, 0, lamb0, mu0)
    algo_lamb = ROL.Algorithm("Line Search", params)
    x_0_lamb = FeVector(lamb0.vector(), inner_product)
    algo_lamb.run(x_0_lamb, obj_lamb, bnd_lamb)
    lamb1.dat.data[:] = obj_lamb.lamb.dat.data[:] # keep inverted lambda

    # invert for mu 
    obj_mu = ObjectiveElastic(inner_product, 1, lamb0, mu0) 
    algo_mu = ROL.Algorithm("Line Search", params)
    x_0_mu = FeVector(mu0.vector(), inner_product)
    algo_mu.run(x_0_mu, obj_mu, bnd_mu)
    mu1.dat.data[:] = obj_mu.mu.dat.data[:] # keep inverted mu
  
    # set inverted values as initial values of the next iteration
    lamb0.dat.data[:] = lamb1.dat.data[:]
    mu0.dat.data[:] = mu1.dat.data[:]
    
    #File("final_lamb_elastic.pvd").write(obj_mu.lamb)
    #File("final_mu_elastic.pvd").write(obj_mu.mu)
    #sys.exit("exit")
   

File("final_lamb_elastic.pvd").write(lamb0)
File("final_mu_elastic.pvd").write(mu0)

