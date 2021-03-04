from firedrake import *

import numpy as np
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
import spyro
from mpi4py import MPI

import psutil
import os

# https://github.com/firedrakeproject/firedrake/issues/1617
outdir = "overthrust3d_fwi_v11/take4/"

model = {}

model["opts"] = {
    "method": "KMV",
    "quadrature": "KMV",  # # GLL, GL, Equi
    "element": "tetra",  # tria or tetra
    "degree": 2,  # p order
    "dimension": 3,  # dimension
}
model["mesh"] = {
    "Lz": 4.14,  # depth in km - always positive
    "Lx": 4.0,  # width in km - always positive
    "Ly": 4.0,  # thickness in km - always positive
    "meshfile": "meshes/overthrust_3D_initial_model_reduced_v2.msh",
    "initmodel": "velocity_models/overthrust_3D_initial_model_reduced.hdf5",
    "truemodel": "velocity_models/overthrust_3D_exact_model_reduced.hdf5",
}
model["PML"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 5.0,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.75,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.75,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.75,  # thickness of the pml in the y-direction (km) - always positive
}

sources = spyro.insert_fixed_value(spyro.create_2d_grid(0.25,3.75,0.25,3.75,2),-0.05, 0)

recvs = spyro.insert_fixed_value(spyro.create_2d_grid(0.2,3.8,0.2,3.8,20),-0.15, 0)

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": len(sources),
    "source_pos": sources,
    "amplitude": 100,
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": len(recvs),
    "receiver_locations": recvs,
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 3.0,  # Final time for event
    "dt": 0.0005,  # timestep size
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 20,  # how frequently to save solution to RAM
    "skip": 1,
}  # how frq. to output to files and screen
# Use one core per shot.
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    "num_cores_per_shot": 1,
    # input is a list of integers with the length of the number of shots.
}

model["inversion"] = {"freq_bands": [None]}

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp_guess = spyro.io.interpolate(model, mesh, V, guess=True)


chk = DumbCheckpoint("saving_vp_guess_"+str(comm.ensemble_comm.rank), mode=FILE_UPDATE, comm=comm.comm)
#chk = DumbCheckpoint("saving_vp_guess_"+str(comm.ensemble_comm.rank), mode=FILE_UPDATE, comm=comm.comm)
#chk.load(vp_guess)


if comm.ensemble_comm.rank ==0):
   File("vp_overthrust3d_guess.pvd").write(vp_guess)

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)

print(gc.isenabled(), flush=True)


def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

# Mask the source/receiver/water layer
mask_depth = -0.50
z = mesh.coordinates[0]
water_index = Function(V).interpolate(conditional(z < mask_depth, 1, 0))
water = np.where(water_index.dat.data[:] < 0.55)

for index, freq_band in enumerate(model["inversion"]["freq_bands"]):

    if COMM_WORLD.rank == 0:
        print(
            f"START OF FWI, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo",
            flush=True,
        )

    def _load_exact_shot():
        for sn in range(model["acquisition"]["num_sources"]):
            if spyro.io.is_owner(comm, sn):
                shot = spyro.io.load_shots(
                    "shots/overthrust_3D_exact_"
                    + str(5.0)
                    + "_Hz_source_"
                    + str(sn)
                    + ".dat"
                )
        return shot

    # spool files for writing
    if comm.ensemble_comm.rank == 0:
        control_file = File(outdir + "control5.0_Hz.pvd", comm=comm.comm)
        grad_file = File(outdir + "grad5.0_Hz.pvd", comm=comm.comm)

    class L2Inner(object):
        """How ROL computes the L2 norm"""

        def __init__(self):
            self.A = assemble(
                TrialFunction(V) * TestFunction(V) * dx(rule=qr_x), mat_type="matfree"
            )
            self.Ap = as_backend_type(self.A).mat()

        def eval(self, _u, _v):
            upet = as_backend_type(_u).vec()
            vpet = as_backend_type(_v).vec()
            A_u = self.Ap.createVecLeft()
            self.Ap.mult(upet, A_u)
            return vpet.dot(A_u)

    class Objective(ROL.Objective):
        """Subclass of ROL.Objective to define value and gradient for problem"""

        def __init__(self, inner_product):
            ROL.Objective.__init__(self)
            self.inner_product = inner_product
            self.p_guess = None
            self.misfit = None
            self.p_exact_recv = _load_exact_shot()

        def value(self, x, tol):
            """Compute the functional"""

            if COMM_WORLD.rank == 0:
                print(
                    f"START OF ITER, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo",
                    flush=True,
                )

            J_total = np.zeros((1))
            for sn in range(model["acquisition"]["num_sources"]):
                if spyro.io.is_owner(comm, sn):

                    self.p_guess, p_guess_recv = spyro.solvers.Leapfrog(
                        model,
                        mesh,
                        comm,
                        vp_guess,
                        sources,
                        receivers,
                        source_num=sn,
                        output=False,
                    )
                    self.misfit = spyro.utils.evaluate_misfit(
                        model, comm, p_guess_recv, self.p_exact_recv
                    )
                    J_total[0] += spyro.utils.compute_functional(model, comm, self.misfit)

            # reduce over ALL cores
            J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)

            #gc.collect()

            if COMM_WORLD.rank == 0:
                print(
                    f"END OF FORWARD, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo",
                    flush=True,
                )

            return J_total[0]

        def gradient(self, g, x, tol):
            """Compute the gradient of the functional"""

            if COMM_WORLD.rank == 0:
                print(
                    f"START OF GRAD, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mb",
                    flush=True,
                )

            #gc.collect()

            dJ_local = Function(V, name="total_gradient")
            for sn in range(model["acquisition"]["num_sources"]):
                if spyro.io.is_owner(comm, sn):
                    dJ = spyro.solvers.Leapfrog_adjoint(
                        model,
                        mesh,
                        comm,
                        vp_guess,
                        self.p_guess,
                        self.misfit,
                    )

            # sum over all ensemble members
            comm.allreduce(dJ, dJ_local)

            # mask the water layer
            dJ_local.dat.data[water] = 0.0

            if comm.ensemble_comm.rank == 0:
                print('writing: '+str(comm.comm.rank)+" on ensemble "+str(comm.ensemble_comm.rank),flush=True)
                grad_file.write(dJ_local)

            g.scale(0)
            g.vec += dJ_local

            if COMM_WORLD.rank == 0:
                print(
                    f"END OF GRADIENT, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo",
                    flush=True,
                )

        def update(self, x, flag, iteration):
            """Update the control"""
            vp_guess.assign(Function(V, x.vec, name="velocity"))
            if iteration >= 0:
                chk.store(vp_guess)
                if comm.ensemble_comm.rank == 0:
                    control_file.write(vp_guess)

    # "Line Search": {"Descent Method": {"Type": "Steepest Descent"}},
    paramsDict = {
        "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
        "Step": {
            "Type": "Augmented Lagrangian",
            "Augmented Lagrangian": {
                "Subproblem Step Type": "Line Search",
                "Subproblem Iteration Limit": 10.0,
            },
            "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
        },
        "Status Test": {
            "Gradient Tolerance": 1e-16,
            "Iteration Limit": 50,
            "Step Tolerance": 1.0e-16,
        },
    }

    params = ROL.ParameterList(paramsDict, "Parameters")

    inner_product = L2Inner()

    obj = Objective(inner_product)

    xlo = Function(V)
    xlo.interpolate(Constant(1.0))
    x_lo = FeVector(xlo.vector(), inner_product)

    xup = Function(V)
    xup.interpolate(Constant(6.0))
    x_up = FeVector(xup.vector(), inner_product)

    bnd = ROL.Bounds(x_lo, x_up, 1.0)

    u = Function(V, name="velocity").assign(vp_guess)
    opt = FeVector(u.vector(), inner_product)

    algo = ROL.Algorithm("Line Search", params)

    algo.run(opt, obj, bnd)

    vp_guess = Function(V, opt.vec)

    if comm.ensemble_comm.rank == 0:
        File("overthrust3d_res5.pvd", comm=comm.comm).write(vp_guess)
                                                                           
