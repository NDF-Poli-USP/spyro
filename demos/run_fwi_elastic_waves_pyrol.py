# script used to run a simple fwi (elastic waves)

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
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV", # Equi or KMV
    "degree": 3,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 1.5,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
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
    "source_pos": [(-0.75, 0.75)],
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 1,
    "receiver_locations": spyro.create_transect(
       (-0.9, 0.375), (-0.9, 1.125), 1
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 0.0005*4000,  # Final time for event
    "tf": 0.0005*1600,  # Final time for event (for test 7)
    "dt": 0.00050,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

mesher="FM" # either SM (SeismicMesh), FM (Firedrake mesher), or RM (read an existing mesh)
if mesher=="FM":
    #                            (X     Y)  ->  (Z  X) in spyro
    #mesh = RectangleMesh(30, 30, 1.75, 2.0, diagonal="crossed")
    #mesh = RectangleMesh(85, 85, 1.5, 1.5, diagonal="crossed") # to test water layer
    #mesh = RectangleMesh(80, 80, 1.5, 1.5, diagonal="crossed") # to test water layer, mesh aligned with interface
    mesh = RectangleMesh(50, 50, 1.5, 1.5, diagonal="crossed") # to test FWI, mesh aligned with interface
elif mesher=="SM":
    raise ValueError("check this first")
    bbox = (0.0, 1.5, 0.0, 1.5)
    rect = SeismicMesh.Rectangle(bbox)
    points, cells = SeismicMesh.generate_mesh(domain=rect, edge_length=0.025)
    mshname = "meshes/test_mu=0.msh"
    meshio.write_points_cells(
        mshname,
        points[:], # do not swap here
        [("triangle", cells)],
        file_format="gmsh22",
        binary=False
    )
    mesh = Mesh(mshname)
elif mesher=="RM":
    mshname = "meshes/test_mu=0_unstructured.msh" # to test water layer
    mesh = Mesh(mshname)
else:
    raise ValueError("mesher not yet supported")

if model["BCs"]["status"]:
    mesh.coordinates.dat.data[:, 0] -= 1.75 # x -> z
    mesh.coordinates.dat.data[:, 1] -= 0.25 # y -> x
else:
    mesh.coordinates.dat.data[:, 0] -= 1.5
    mesh.coordinates.dat.data[:, 1] -= 0.0

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)
V = FunctionSpace(mesh, element)

z, x = SpatialCoordinate(mesh) 

####### control #######
run_fwi=1 
run_elastic=1
####### control #######

if run_fwi==0: # exact values
    if run_elastic:
        lamb = Constant(1./2.) # exact
        mu = Constant(1./4.)
    else: # acoustic version
        lamb = Constant(1.) # exact

rho = Constant(1.) # for test 3 and 7 (constant cp and cd)

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
                dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
            )

if run_fwi==0: # only to generate the exact value
    print("Starting forward computation")
    start = time.time()
    if run_elastic:
        u_field, uz_at_recv, ux_at_recv, uy_at_recv = spyro.solvers.forward_elastic_waves(
            model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=False
        )
        spyro.io.save_shots(model, comm, uz_at_recv, file_name="./shots/test_fwi/uz_at_recv_exact.dat")
        spyro.io.save_shots(model, comm, ux_at_recv, file_name="./shots/test_fwi/ux_at_recv_exact.dat")
    else:
        vp = (lamb / rho) ** 0.5
        u_field, uz_at_recv = spyro.solvers.forward(
            model, mesh, comm, vp, sources, wavelet, receivers, output=True
        )
        spyro.io.save_shots(model, comm, uz_at_recv, file_name="./shots/test_fwi/p_at_recv_exact.dat")
    end = time.time()
    print(round(end - start,2))

    if 0:
        cmin=-1e-4
        cmax=1e-4
        u_at_recv = (uz_at_recv**2. + ux_at_recv**2.)**0.5
        spyro.plots.plot_shots(model, comm, u_at_recv, show=True, vmin=cmin, vmax=cmax)
   
    if 1:
        ue=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            ue.append(uz_at_recv[ti][rn])
            ue.append(uz_at_recv[ti][rn])
        plt.title("u_z")
        plt.plot(ue,label='exact')
        plt.legend()
        plt.savefig('/home/santos/Desktop/FWI_exact.png')
        plt.show()

    sys.exit("exiting without running gradient")

quad_rule = finat.quadrature.make_quadrature(
    V.finat_element.cell, V.ufl_element().degree(), "KMV"
)
dxlump = dx(rule=quad_rule)

def regularize_gradient(vp, dJ, gamma):#{{{
    """Tikhonov regularization"""
    m_u = TrialFunction(V)
    m_v = TestFunction(V)
    qr_x, qr_s, _ = spyro.domains.quadrature.quadrature_rules(V)
    mgrad = m_u * m_v * dx(rule=qr_x)
    ffG = dot(grad(vp), grad(m_v)) * dx(rule=qr_x)
    G = mgrad - ffG
    lhsG, rhsG = lhs(G), rhs(G)
    gradreg = Function(V)
    grad_prob = LinearVariationalProblem(lhsG, rhsG, gradreg)
    grad_solver = LinearVariationalSolver(
        grad_prob,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "mat_type": "matfree",
        },
    )
    grad_solver.solve()
    dJ += gamma * gradreg
    return dJ
#}}}
class L2Inner(object): #{{{
    def __init__(self):
        self.A = assemble(
            TrialFunction(V) * TestFunction(V) * dxlump, mat_type="matfree"
        )
        self.Ap = as_backend_type(self.A).mat()

    def eval(self, _u, _v):
        upet = as_backend_type(_u).vec()
        vpet = as_backend_type(_v).vec()
        A_u = self.Ap.createVecLeft()
        self.Ap.mult(upet, A_u)
        return vpet.dot(A_u)
#}}}
class ObjectiveElastic(ROL.Objective): #{{{
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.u_guess = None # Z,X displacements
        self.misfit_uz = 0.0 # at receivers
        self.misfit_ux = 0.0 # at receivers
        self.misfit_uy = 0.0 # at receivers
        self.lamb = Function(V) 
        self.mu = Constant(1./4.) # FIXME change here 
        self.uz_exact_recv = spyro.io.load_shots(model, comm, file_name="./shots/test_fwi/uz_at_recv_exact.dat") 
        self.ux_exact_recv = spyro.io.load_shots(model, comm, file_name="./shots/test_fwi/ux_at_recv_exact.dat")

    def value(self, x, tol):
       # print("Starting forward computation - elastic waves")
        """Compute the functional"""
        J_total = np.zeros((1))
        self.lamb.dat.data[:] = x[:]
        self.u_guess, uz_guess_recv, ux_guess_recv, uy_guess_recv = spyro.solvers.forward_elastic_waves(
            model, mesh, comm, rho, self.lamb, self.mu, sources, wavelet, receivers, output=False
        )
        self.misfit_uz = spyro.utils.evaluate_misfit(model, uz_guess_recv, self.uz_exact_recv)
        self.misfit_ux = spyro.utils.evaluate_misfit(model, ux_guess_recv, self.ux_exact_recv)
        self.misfit_uy = []
       
        reg = model["opts"]["regularization"]
        J_total[0] += spyro.utils.compute_functional(model, self.misfit_uz, vp=self.lamb)
        if reg:
            model["opts"]["regularization"] = False
        J_total[0] += spyro.utils.compute_functional(model, self.misfit_ux)
        if reg:
            model["opts"]["regularization"] = True    

        ue=[]
        ug=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            ue.append(self.uz_exact_recv[ti][rn])
            ug.append(uz_guess_recv[ti][rn])
        plt.title("u_z")
        plt.plot(ue,label='exact')
        plt.plot(ug,label='guess')
        plt.legend()
        plt.savefig('/home/santos/Desktop/FWI_elastic.png')
        plt.close()
        return J_total[0]

    def gradient(self, g, x, tol):
        #print("Starting gradient computation - elastic waves")
        """Compute the gradient of the functional"""
        dJ = Function(V, name="gradient")
        self.lamb.dat.data[:] = x[:]
        dJdl_local, dJdm_local = spyro.solvers.gradient_elastic_waves(
            model, mesh, comm, rho, self.lamb, self.mu, 
            receivers, self.u_guess, self.misfit_uz, self.misfit_ux, self.misfit_uy, output=False
        )
       
        dJ = dJdl_local
        # regularize the gradient if asked.
        if model['opts']['regularization']:
            dJ = regularize_gradient(self.u_guess, dJ, model['opts']['gamma'])
        # Visualize
        File("dJdl_elastic.pvd").write(dJdl_local)
        File("dJdm_elastic.pvd").write(dJdm_local)
        sys.exit("exiting without running gradient")
        g.scale(0)
        g.vec += dJ
#}}}
class ObjectiveAcoustic(ROL.Objective): # {{{
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.p_guess = None
        self.misfit = 0.0
        self.vp = Function(V) 
        self.p_exact_recv = spyro.io.load_shots(model, comm, file_name="./shots/test_fwi/p_at_recv_exact.dat") 

    def value(self, x, tol):
        print("Starting forward computation - acoustic waves")
        """Compute the functional"""
        J_total = np.zeros((1))
        self.vp.dat.data[:] = x[:]
        self.p_guess, p_guess_recv = spyro.solvers.forward(model, mesh, comm, self.vp, sources, wavelet, receivers)
        self.misfit = spyro.utils.evaluate_misfit(model, p_guess_recv, self.p_exact_recv)
        J_total[0] += spyro.utils.compute_functional(model, self.misfit, vp=self.vp)
        
        ue=[]
        ug=[]
        nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
        rn = 0
        for ti in range(nt):
            ue.append(self.p_exact_recv[ti][rn])
            ug.append(p_guess_recv[ti][rn])
        plt.title("p")
        plt.plot(ue,label='exact')
        plt.plot(ug,label='guess')
        plt.legend()
        plt.savefig('/home/santos/Desktop/FWI_acoustic.png')
        plt.close()
        return J_total[0]

    def gradient(self, g, x, tol):
        print("Starting gradient computation - acoustic waves")
        """Compute the gradient of the functional"""
        dJ = Function(V, name="gradient")
        self.vp.dat.data[:] = x[:]
        dJ_local = spyro.solvers.gradient(model, mesh, comm, self.vp, receivers, self.p_guess, self.misfit)
        dJ = dJ_local
        # regularize the gradient if asked.
        if model['opts']['regularization']:
            dJ = regularize_gradient(self.vp, dJ, model['opts']['gamma'])
        # Visualize
        File("dJ_acoustic.pvd").write(dJ)
        g.scale(0)
        g.vec += dJ
#}}}

paramsDict = {
    "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
    "Step": {
        "Type": "Augmented Lagrangian",
        "Augmented Lagrangian": {
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 5.0,
        },
        "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
    },
    "Status Test": {
        "Gradient Tolerance": 1e-16,
        "Iteration Limit": 50,
        "Step Tolerance": 1.0e-16,
    },
}

# prepare to run FWI, set guess add control bounds to the problem (uses more RAM)
params = ROL.ParameterList(paramsDict, "Parameters")
inner_product = L2Inner()

x0 = Function(V)
xlo = Function(V)
xup = Function(V)
if run_elastic:
    obj = ObjectiveElastic(inner_product)
    x0.dat.data[:] = 0.25 # lambda 
    xlo.interpolate(Constant(0.1))
    xup.interpolate(Constant(1.0))
else:
    obj = ObjectiveAcoustic(inner_product)
    x0.dat.data[:] = (0.9) ** 0.5 # vp
    xlo.interpolate(Constant(0.7))
    xup.interpolate(Constant(1.3))

opt = FeVector(x0.vector(), inner_product)
x_lo = FeVector(xlo.vector(), inner_product)
x_up = FeVector(xup.vector(), inner_product)
bnd = ROL.Bounds(x_lo, x_up, 1.0)

# run FWI
algo = ROL.Algorithm("Line Search", params)
algo.run(opt, obj, bnd)

if run_elastic:
    File("final_lamb_elastic.pvd").write(obj.lamb)
else:
    File("final_vp_acoustic.pvd").write(obj.vp)


