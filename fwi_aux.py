from firedrake import *
import numpy as np
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
from scipy.interpolate import griddata
import ROL
from mpi4py import MPI
import spyro
import psutil
import os

def forward_solver(model, comm, output_pdf = False, guess = False, save_shots=False):
    mesh, V = spyro.io.read_mesh(model, comm)
    vp = spyro.io.interpolate(model, mesh, V, guess=guess)
    sources = spyro.Sources(model, mesh, V, comm)
    receivers = spyro.Receivers(model, mesh, V, comm)
    wavelet = spyro.full_ricker_wavelet(
        dt=model["timeaxis"]["dt"],
        tf=model["timeaxis"]["tf"],
        freq=model["acquisition"]["frequency"],
    )
    p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
    if save_shots:
        spyro.io.save_shots(model, comm, p_r)
    if output_pdf:
        spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
    return True

def get_memory_usage():
        """Return the memory usage in Mo."""
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(2 ** 20)
        return mem

def fwi_solver(model, comm, number_of_iterations = 20):
    ## My parameters:
    frequency = model["acquisition"]["frequency"]

    outdir = "FWI/"+str(int(frequency))+"Hz_"
    if COMM_WORLD.rank == 0:
        mem = open(outdir + "mem.txt", "w")
        func = open(outdir + "func.txt", "w")

    
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
    dxlump = dx(rule=quad_rule)

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


    def regularize_gradient(vp, dJ, gamma):
        """Tikhonov regularization"""
        m_u = TrialFunction(V)
        m_v = TestFunction(V)
        mgrad = m_u * m_v * dxlump
        ffG = dot(grad(vp), grad(m_v)) * dxlump
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
        dJ += gamma*gradreg
        return dJ


    class Objective(ROL.Objective):
        def __init__(self, inner_product):
            ROL.Objective.__init__(self)
            self.inner_product = inner_product
            self.p_guess = None
            self.misfit = 0.0
            self.p_exact_recv = spyro.io.load_shots(model, comm)
            print(np.shape(self.p_exact_recv))
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
            print(np.shape(p_guess_recv))
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
                gamma = model['opts']['gamma']
                dJ = regularize_gradient(vp, dJ, gamma)
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
            "Iteration Limit": number_of_iterations,
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
        File(str(frequency)+"Hz_res.pvd", comm=comm.comm).write(vp)


    if COMM_WORLD.rank == 0:
        func.close()
        mem.close()

    segy_fname = "velocity_models/test0_"+str(frequency) +"Hz_iteration.segy"


    # write a new file to be used in the re-meshing
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        grid_spacing = 10.0 / 1000.0
        m = V.ufl_domain()
        W = VectorFunctionSpace(m, V.ufl_element())
        coordinates = interpolate(m.coordinates, W)
        x, y = coordinates.dat.data[:, 0], coordinates.dat.data[:, 1]    # add buffer to avoid NaN when calling griddata
        min_x = np.amin(x) + 0.05
        max_x = np.amax(x) - 0.05
        min_y = np.amin(y) + 0.05
        max_y = np.amax(y) - 0.05  
        z = vp.dat.data[:]  # convert from km/s to m/s    # target grid to interpolate to
        xi = np.arange(min_x, max_x, grid_spacing)
        yi = np.arange(min_y, max_y, grid_spacing)
        xi, yi = np.meshgrid(xi, yi)    # interpolate
        vp_i = griddata((x, y), z, (xi, yi), method="linear")
        print("creating new velocity model...", flush=True)
        spyro.io.create_segy(vp_i, segy_fname)

    if COMM_WORLD.rank == 0:
        func.close()
        mem.close()
    
    return segy_fname

