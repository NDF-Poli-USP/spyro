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
#from ..domains import quadrature, space

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV", # Equi or KMV
    "degree": 3,  # p order
    "dimension": 2,  # dimension
    "regularization": True,  # regularization is on?
    "gamma": 1e-8, # regularization parameter
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
       # (-0.4, 0.375), (-0.4, 1.125), 2
       # (-1.1, 0.375), (-1.1, 1.125), 2
       (-1.1, 0.375), (-1.1, 1.125), 1
       # (-1.25, -0.25), (-1.25, 1.75), 100 for the case with ABL
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
run_elastic=0
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
        spyro.io.save_shots(model, comm, uz_at_recv, file_name="./shots/test_fwi/uz_at_recv_exact.dat")
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

uz_exact = spyro.io.load_shots(model, comm, file_name="./shots/test_fwi/uz_at_recv_exact.dat")
ux_exact = spyro.io.load_shots(model, comm, file_name="./shots/test_fwi/ux_at_recv_exact.dat")

def regularize_gradient(vp, dJ, gamma):
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

# to be used with scipy
def compute_J_dJ(x):
    J_total = np.zeros((1))
    print("Starting forward computation")
    if run_elastic:
        # assign field
        lamb_chg = Function(V)
        lamb_chg.dat.data[:] = x[:] # FIXME include mu
        # compute the guess
        u_guess, uz_guess, ux_guess, uy_guess = spyro.solvers.forward_elastic_waves(
            model, mesh, comm, rho, lamb_chg, mu, sources, wavelet, receivers, output=False
        )
        # compute the residual (misfit) 
        residual_z = spyro.utils.evaluate_misfit(model, uz_guess, uz_exact)
        residual_x = spyro.utils.evaluate_misfit(model, ux_guess, ux_exact)
        residual_y = []
        # compute the functional (J)
        model["opts"]["regularization"] = True
        model["opts"]["gamma"] = 1.e-6
        J_total[0] += spyro.utils.compute_functional(model, residual_z, vp=lamb_chg)
        model["opts"]["regularization"] = False
        J_total[0] += spyro.utils.compute_functional(model, residual_x)
        # compute the gradient (grad J)
        print("Starting gradient computation")
        dJdl, dJdm = spyro.solvers.gradient_elastic_waves(
            model, mesh, comm, rho, lamb_chg, mu, receivers, u_guess, residual_z, residual_x, residual_y, output=True
        )
        # regularize the gradient if requested
        if model["opts"]["regularization"]:
            dJdl = regularize_gradient(lamb_chg, dJdl, model["opts"]["gamma"])
    else:
        # assign field
        vp = Function(V)
        vp.dat.data[:] = x[:]
        # compute the guess
        u_guess, uz_guess = spyro.solvers.forward(
            model, mesh, comm, vp, sources, wavelet, receivers,
        )
        # compute the residual (misfit) 
        residual_z = spyro.utils.evaluate_misfit(model, uz_guess, uz_exact)
        # compute the functional (J)
        model["opts"]["regularization"] = False
        model["opts"]["gamma"] = 1.e-3
        J_total[0] += spyro.utils.compute_functional(model, residual_z, vp=vp)
        # compute the gradient (grad J)
        print("Starting gradient computation")
        dJdl = spyro.solvers.gradient(
            model, mesh, comm, vp, receivers, u_guess, residual_z,
        )
        # regularize the gradient if requested
        if model["opts"]["regularization"]:
            dJdl = regularize_gradient(vp, dJdl, model["opts"]["gamma"])

    ue=[]
    ug=[]
    nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
    rn = 0
    for ti in range(nt):
        ue.append(uz_exact[ti][rn])
        ug.append(uz_guess[ti][rn])
    plt.title("u_z")
    plt.plot(ue,label='exact')
    plt.plot(ug,label='guess')
    plt.legend()
    plt.savefig('/home/santos/Desktop/FWI.png')
    plt.close()
    #plt.show()
    File("dJdl.pvd").write(dJdl)
    #File("dJdm.pvd").write(dJdm)
 
    return J_total[0], dJdl.dat.data[:] # FIXME include dJdm

# set guess
x0 = Function(V)
if run_elastic:
    x0.dat.data[:] = 1./3. # same as obtained by guess
else:
    x0.dat.data[:] = (0.9) ** 0.5 # vp

lmax = 1.2
lmin = 0.5
bounds = [(lmin,lmax) for _ in range(len(x0.dat.data[:]))]
print("Starting minimize computation")
if 1:
    #method = 'CG' it is worse than L-BFGS-B
    method = 'L-BFGS-B'
    res = minimize(compute_J_dJ, 
                x0.dat.data[:], 
                method=method, 
                jac=True, bounds=bounds, tol = 1e-12, 
                options={"disp": True,"maxcor": 20,"eps": 1e-12, "ftol":1e-12 , "gtol": 1e-12,"maxls":3,"maxiter": 30})
else:
    minimizer_kwargs = {"method": "L-BFGS-B","jac": True, "bounds": bounds}
    res = basinhopping(compute_J_dJ, 
                        x0.dat.data[:],
                        niter=4,
                        disp=True,
                        minimizer_kwargs=minimizer_kwargs)

xf = Function(V)
xf.dat.data[:] = res.x[:]
File("final_lamb.pvd").write(xf)

