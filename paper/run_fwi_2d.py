from firedrake import *
import numpy as np
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from mpi4py import MPI

import spyro

# import gc
import psutil
import os


def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2**20)
    return mem


outdir = "fwi_p3_ho/"
if COMM_WORLD.rank == 0:
    mem = open(outdir + "mem.txt", "w")
    func = open(outdir + "func.txt", "w")

model = {}
model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 3,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1.0,  # regularization parameter
}
model["parallelism"] = {
    "type": "automatic",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/marmousi_guess.msh",
    "initmodel": "velocity_models/marmousi_guess.hdf5",
    "truemodel": "not_used.hdf5",
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 40,
    "source_pos": spyro.create_transect((-0.01, 1.0), (-0.01, 15.0), 40),
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 500,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 500),
}
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 5.00,  # Final time for event
    "dt": 0.001,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 1000,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
    "skip": 4,
}
comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)
if COMM_WORLD.rank == 0:
    print(f"The mesh has {V.dim()} degrees of freedom")
vp = spyro.io.interpolate(model, mesh, V, guess=True)
if comm.ensemble_comm.rank == 0:
    File("guess_velocity.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
if comm.ensemble_comm.rank == 0:
    control_file = File(outdir + "control.pvd", comm=comm.comm)
    grad_file = File(outdir + "grad.pvd", comm=comm.comm)
quad_rule = finat.quadrature.make_quadrature(
    V.finat_element.cell, V.ufl_element().degree(), "KMV"
)
dxlump = dx(scheme=quad_rule)

water = np.where(vp.dat.data[:] < 1.51)


class L2Inner(object):
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


def regularize_gradient(vp, dJ):
    """Tikhonov regularization"""
    m_u = TrialFunction(V)
    m_v = TestFunction(V)
    mgrad = m_u * m_v * dx(scheme=quad_rule)
    ffG = dot(grad(vp), grad(m_v)) * dx(scheme=quad_rule)
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
    dJ += gradreg
    return dJ


class Objective(ROL.Objective):
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.p_guess = None
        self.misfit = 0.0
        self.p_exact_recv = spyro.io.load_shots(model, comm)

    def value(self, x, tol):
        """Compute the functional"""
        J_total = np.zeros((1))
        self.p_guess, p_guess_recv = spyro.solvers.forward(
            model,
            mesh,
            comm,
            vp,
            sources,
            wavelet,
            receivers,
        )
        self.misfit = spyro.utils.evaluate_misfit(
            model, p_guess_recv, self.p_exact_recv
        )
        J_total[0] += spyro.utils.compute_functional(model, self.misfit, velocity=vp)
        J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            J_total[0] /= comm.comm.size

        if COMM_WORLD.rank == 0:
            mem.write(str(get_memory_usage()))
            func.write(str(J_total[0]))
            mem.write("\n")
            func.write("\n")

        return J_total[0]

    def gradient(self, g, x, tol):
        """Compute the gradient of the functional"""
        dJ = Function(V, name="gradient")
        dJ_local = spyro.solvers.gradient(
            model,
            mesh,
            comm,
            vp,
            receivers,
            self.p_guess,
            self.misfit,
        )
        if comm.ensemble_comm.size > 1:
            comm.allreduce(dJ_local, dJ)
        else:
            dJ = dJ_local
        dJ /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            dJ /= comm.comm.size
        # regularize the gradient if asked.
        if model["opts"]["regularization"]:
            dJ = regularize_gradient(vp, dJ)
        # mask the water layer
        dJ.dat.data[water] = 0.0
        # Visualize
        if comm.ensemble_comm.rank == 0:
            grad_file.write(dJ)
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        vp.assign(Function(V, x.vec, name="velocity"))
        # If iteration reduces functional, save it.
        if iteration >= 0:
            if comm.ensemble_comm.rank == 0:
                control_file.write(vp)


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
        "Iteration Limit": 100,
        "Step Tolerance": 1.0e-16,
    },
}

params = ROL.ParameterList(paramsDict, "Parameters")

inner_product = L2Inner()

obj = Objective(inner_product)

u = Function(V, name="velocity").assign(vp)
opt = FeVector(u.vector(), inner_product)

# Add control bounds to the problem (uses more RAM)
xlo = Function(V)
xlo.interpolate(Constant(1.0))
x_lo = FeVector(xlo.vector(), inner_product)

xup = Function(V)
xup.interpolate(Constant(5.0))
x_up = FeVector(xup.vector(), inner_product)

bnd = ROL.Bounds(x_lo, x_up, 1.0)

algo = ROL.Algorithm("Line Search", params)

algo.run(opt, obj, bnd)

if comm.ensemble_comm.rank == 0:
    File("res.pvd", comm=comm.comm).write(vp)


if COMM_WORLD.rank == 0:
    func.close()
    mem.close()
