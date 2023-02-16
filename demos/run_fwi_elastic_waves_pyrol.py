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
    "source_pos": [(0.4, 0.4)],
    "frequency": 10.0,
    "delay": 0.0,
    "num_receivers": 1,
    "receiver_locations": spyro.create_transect(
        (0.5, 0.4), (0.5, 0.4), 1
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.001*400,  # Final time for event (for test 7)
    "dt": 0.0010,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

mesh = RectangleMesh(50, 50, model["mesh"]["Lz"], model["mesh"]["Lx"], diagonal="crossed")

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)
#V = FunctionSpace(mesh, element)
V = VectorFunctionSpace(mesh, element)
P = VectorFunctionSpace(mesh, element, dim=2) # for lambda and mu inside PyROL
H = FunctionSpace(mesh, element)

z, x = SpatialCoordinate(mesh) 

mu_exact    = 1./4.
lamb_exact  = 1./2.
mu_guess    = 0.5 * mu_exact 
lamb_guess  = 0.5 * lamb_exact
rho_exact   = 1.0

lamb = Constant(lamb_exact) # exact
mu = Constant(mu_exact)
rho = Constant(rho_exact) 

sources = spyro.Sources(model, mesh, H, comm)
receivers = spyro.Receivers(model, mesh, H, comm)
wavelet = spyro.full_ricker_wavelet(
                dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
            )

print("Starting forward computation")
start = time.time()
u_exact, _, _, _ = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=False
)
end = time.time()
print(round(end - start,2))
#sys.exit("exit")

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
        self.u_exact = u_exact # Z,X displacements, entire domain
        self.misfit = None  # Z,X misfits, entire domain
        self.misfit_uz = []
        self.misfit_ux = []
        self.misfit_uy = []
        self.lamb = Function(H) 
        self.mu = Function(H)

    def value(self, x, tol):
        #print("Starting forward computation - elastic waves")
        """Compute the functional"""
        J_scale = sqrt(1.e14) 
        self.u_guess, _, _, _ = spyro.solvers.forward_elastic_waves(
            model, mesh, comm, rho, self.lamb, self.mu, sources, wavelet, receivers, output=False
        )
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        residual = [Function(H) for t in range(nt)]
        J=0
        for step in range(nt):
            residual[step] = J_scale * (self.u_exact[step] - self.u_guess[step])
            J+=J_scale*assemble(
                    (0.5*inner(self.u_guess[step]-self.u_exact[step],self.u_guess[step]-self.u_exact[step]))*dx
                    )
        self.misfit = residual
        return J

    def gradient(self, g, x, tol):
        #print("Starting gradient computation - elastic waves")
        """Compute the gradient of the functional"""
        if self.misfit:
            dJdl, dJdm = spyro.solvers.gradient_elastic_waves(
                model, mesh, comm, rho, self.lamb, self.mu, 
                receivers, self.u_guess, self.misfit_uz, self.misfit_ux, self.misfit_uy, output=False,
                residual=self.misfit # entire domain
            )
            File("dJdl_elastic.pvd").write(dJdl)
            File("dJdm_elastic.pvd").write(dJdm)
            g.scale(0)
            g.vec.dat.data[:,0] += dJdl.dat.data[:]
            g.vec.dat.data[:,1] += dJdm.dat.data[:]
        else:
            sys.exit("exit called")

    def update(self, x, flag, iteration):
        self.lamb.dat.data[:] = x.vec.dat.data[:,0]
        self.mu.dat.data[:] = x.vec.dat.data[:,1]
#}}}
# ROL parameters definition {{{
paramsDict = {
    "General": {
        "Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 3}#25}
    },
    "Step": {
        "Type": "Augmented Lagrangian",
        "Line Search": {
            "Descent Method": {
                "Type": "Quasi-Newton Step"
            }
        },
        "Augmented Lagrangian": {
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
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 2,#10,
        },
    },
    "Status Test": {
        "Gradient Tolerance": 1e-15,
        'Relative Gradient Tolerance': 1e-10,
        "Step Tolerance": 1.0e-15,
        'Relative Step Tolerance': 1e-15,
        "Iteration Limit": 30
    },
}
#}}}
# prepare to run FWI, set guess add control bounds to the problem (uses more RAM)
params = ROL.ParameterList(paramsDict, "Parameters")
inner_product = L2Inner()

x0 = Function(P)
xlo = Function(P)
xup = Function(P)

x0.dat.data[:,0] = lamb_guess # lambda
x0.dat.data[:,1] = mu_guess # mu

xlo.dat.data[:,0] = 0.3 * lamb_exact
xlo.dat.data[:,1] = 0.3 * mu_exact

xup.dat.data[:,0] = 2.0 * lamb_exact
xup.dat.data[:,1] = 2.0 * mu_exact

opt = FeVector(x0.vector(), inner_product)
x_lo = FeVector(xlo.vector(), inner_product)
x_up = FeVector(xup.vector(), inner_product)

bnd = ROL.Bounds(x_lo, x_up, 1.0)

obj = ObjectiveElastic(inner_product)
algo = ROL.Algorithm("Line Search", params)
algo.run(opt, obj, bnd)

File("final_lamb_elastic.pvd").write(obj.lamb)
File("final_mu_elastic.pvd").write(obj.mu)


