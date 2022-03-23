# Marmousi model used to test elastic FWI in a real case
from firedrake import *
from scipy.optimize import *
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import meshio
#import SeismicMesh
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    #"type": "automatic",
    "type": "spatial",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/marmousi_elastic_with_water_layer_adapted.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.10, 5.0)], # Z and X # with water layer (for wave propagation comparison)
    #"source_pos": [(-0.50, 5.0)], # Z and X # with water layer (maybe for FWI)
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect((-0.05, 0.0), (-0.05, 17.0), 100),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.6,  # Final time for event (used to reach the bottom of the domain)
    #"tf": 0.100,  # Final time for event (used to measure the time)
    "dt": 0.00025, # default
    #"dt": 0.0001, # needs for P=5
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 999999,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh, H = spyro.io.read_mesh(model, comm)
element = spyro.domains.space.FE_method(
                mesh, model["opts"]["method"], model["opts"]["degree"]
            )
V = VectorFunctionSpace(mesh, element)
P = VectorFunctionSpace(mesh, element, dim=2) # for lambda and mu inside PyROL

# interpolate exact Vs, Vp, and Density onto the mesh 
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s 
vs_exact = spyro.io.interpolate(model, mesh, H, guess=False)
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"# m/s
vp_exact = spyro.io.interpolate(model, mesh, H, guess=False)
model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.hdf5"       # g/cm3
rho = spyro.io.interpolate(model, mesh, H, guess=False, field="velocity_model") #FIXME change it in SeismicMesh

# vs and vp in km/s
# rho in g/cm3 == Gt/km3
# lambda and mu in GPa

mu_exact = Function(H, name="mu").interpolate(rho * vs_exact ** 2.)
lamb_exact = Function(H, name="lamb").interpolate(rho * (vp_exact ** 2. - 2. * vs_exact ** 2.))

write_files=0
if comm.ensemble_comm.rank == 0 and write_files==1: #{{{
    rho.rename("rho")
    vp_exact.rename("p-wave vel")
    vs_exact.rename("s-wave vel")
    lamb_exact.rename("lambda")
    mu_exact.rename("mu")
    File("density.pvd", comm=comm.comm).write(rho)
    File("p-wave_velocity.pvd", comm=comm.comm).write(vp_exact)
    File("s-wave_velocity.pvd", comm=comm.comm).write(vs_exact)
    File("lambda.pvd", comm=comm.comm).write(lamb_exact)
    File("mu.pvd", comm=comm.comm).write(mu_exact)
    sys.exit("Exit without running")
#}}}

sources = spyro.Sources(model, mesh, H, comm)
receivers = spyro.Receivers(model, mesh, H, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)

if 0: # {{{
    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V2 = VectorFunctionSpace(mesh, element)
    print("DOF: "+str(V2.dof_count))
    sys.exit("Exit without running")
#}}}

print("Starting forward computation",flush=True)
start = time.time()
u_exact, uz_exact, ux_exact, uy_exact = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb_exact, mu_exact, sources, wavelet, receivers, output=False
)
end = time.time()
print(round(end - start,2), flush=True)
File("u_exact.pvd").write(u_exact[-1])
J_global = []

model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.smoothed.segy.hdf5"# m/s 
vs_guess = spyro.io.interpolate(model, mesh, H, guess=False)

model["mesh"]["truemodel"] = "velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed.segy.hdf5"# m/s
vp_guess = spyro.io.interpolate(model, mesh, H, guess=False)

mu_guess = Function(H, name="mu").interpolate(rho * vs_guess ** 2.)
lamb_guess = Function(H, name="lamb").interpolate(rho * (vp_guess ** 2. - 2. * vs_guess ** 2.))

write_files=0
if comm.ensemble_comm.rank == 0 and write_files==1: #{{{
    rho.rename("rho")
    vp_guess.rename("p-wave vel")
    vs_guess.rename("s-wave vel")
    lamb_guess.rename("lambda")
    mu_guess.rename("mu")
    File("density.pvd", comm=comm.comm).write(rho)
    File("p-wave_velocity.pvd", comm=comm.comm).write(vp_guess)
    File("s-wave_velocity.pvd", comm=comm.comm).write(vs_guess)
    File("lambda.pvd", comm=comm.comm).write(lamb_guess)
    File("mu.pvd", comm=comm.comm).write(mu_guess)
    sys.exit("Exit without running")
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
        #plt.savefig('/home/tdsantos/FWI.png')
        plt.savefig('/home/santos/Desktop/FWI.png')
        plt.close()

        File("u_guess.pvd").write(self.u_guess[-1])
        
        J_global.append(J_total[0])

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
            'Type': 'Limited-Memory BFGS', # Line Search Quasi-Newton Method 
            'Maximum Storage': 5, #(20==10)
            'Barzilai-Borwein Type': 2
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
                #'Type': 'Nonlinear CG' # Quasi-Newton Method works better
            },
            'Curvature Condition': {
                'Type': 'Strong Wolfe Conditions', # works fine
            }
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

vp_final = Function(H).assign( ( (obj.lamb+2.*obj.mu)/rho )**0.5 )
vs_final = Function(H).assign( ( obj.mu/rho ) ** 0.5 )
File("final_vp.pvd").write(vp_final)
File("final_vs.pvd").write(vs_final)
File("final_lamb_elastic.pvd").write(obj.lamb)
File("final_mu_elastic.pvd").write(obj.mu)

print(J_global)
print("J (initial)="+str(J_global[0]))
print("J (final)="+str(J_global[-1]))
