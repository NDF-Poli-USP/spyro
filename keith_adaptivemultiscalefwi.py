
from firedrake import *

import SeismicMesh
import meshio

import numpy as np
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
import spyro
from mpi4py import MPI

from pympler import muppy, summary, tracker
import psutil
import sys
import os
import gc



outdir = "testing_fwi/"

# START DEFINING OPTIONS
model = {}
# Specify Finite Element related options
model["opts"] = {
    "method": "KMV",  # CG, DG, KMV
    "variant": None,
    "type": "SIP",  # for DG only - SIP, NIP and IIP
    "element": "tria",  # tria or tetra
    "degree": 3,  # spatial polynomial order
    "quadrature": "KMV",  # # GLL, GL, Equi, KMV
    "dimension": 2,  # spatial dimension
    "beta": 0.0,  # for Newmark time integration only
    "gamma": 0.5,  # for Newmark time integration only
}
# Define the mesh geometry and filenames of the velocity models
model["mesh"] = {
    "Lz": 4.0,  # depth in km - always positive
    "Lx": 18.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/mm_init.msh",
    "initmodel": "velocity_models/mm_init.hdf5",
    "truemodel": "velocity_models/mm_exact.hdf5",
}
# Use a Perfectly Matched Layer to damp reflected waves.
# Note here, it's built to be 0.5 km thick on three sides of the domain
model["PML"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "exponent": 1,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.5,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}
# Simulate with 40 shots equi-spaced near the top of the domain
# and record the solution at 301 receivers.
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 40,
    "source_pos": spyro.create_receiver_transect((-0.15, 0.1), (-0.15, 16.9), 40),
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 301,
    "receiver_locations": spyro.create_receiver_transect(
        (-0.15, 0.1), (-0.15, 16.9), 301
    ),
}
# Perform each shot simulation for 3.0 seconds and save all
# timesteps for the gradient calculation.
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 3.00,  # Final time for event
    "dt": 0.0005,  # timestep size
    "nspool": 200,  # how frequently to output solution to pvds
    "fspool": 50,  # how frequently to save solution to RAM
}
# Use one core per shot.
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

model["inversion"] = {"freq_bands": [3.0, 5.0, 8.0]}

def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def remesh(fname, freq, mesh_iter, comm):
    """for now some hardcoded options"""
    if comm.ensemble_comm.rank == 0:
        bbox = (-4000.0, 0.0, -500.0, 17500.0)

        wl = 7

        # Desired minimum mesh size in domain
        hmin = 1500.0 / (wl * freq)

        rectangle = SeismicMesh.Rectangle(bbox)

        # Construct mesh sizing object from velocity model
        ef = SeismicMesh.get_sizing_function_from_segy(
            fname, bbox, hmin=hmin, wl=wl, freq=freq, dt=0.001, comm=comm.comm
        )

        SeismicMesh.write_velocity_model(
            fname,
            ofname="velocity_models/mm_GUESS" + str(mesh_iter),
            comm=comm.comm,
        )

        points, cells = SeismicMesh.generate_mesh(
            domain=rectangle, edge_length=ef, comm=comm.comm
        )

        if comm.comm.rank == 0:
            meshio.write_points_cells(
                "meshes/mm_GUESS" + str(mesh_iter) + ".msh",
                points / 1000,
                [("triangle", cells)],
                file_format="gmsh22",
                binary=False,
            )
            # for visualization
            meshio.write_points_cells(
                "meshes/mm_GUESS" + str(mesh_iter) + ".vtk",
                points / 1000.0,
                [("triangle", cells)],
            )


comm = spyro.utils.mpi_init(model)

mesh_iter = 0

for index, freq_band in enumerate(model["inversion"]["freq_bands"]):
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print(
            "INFO: Executing inversion for low-passed cut off of "
            + str(freq_band)
            + " Hz...",
            flush=True,
        )

    # do the mesh adaptation here based on the new guess mesh
    if mesh_iter > 0:

        segy_fname = "velocity_models/mm_GUESS" + str(mesh_iter) + ".segy"

        # interpolate vp_exact to a structured grid and write to a segy file for later meshing with SeismicMesh
        xi, yi, vp_i = spyro.utils.write_function_to_grid(
            vp_guess, V, grid_spacing=10.0 / 1000.0
        )

        # write a new file to be used in the re-meshing
        if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
            print("creating new velocity model...", flush=True)
            spyro.io.create_segy(vp_i, segy_fname)

        # call SeismicMesh in serial to build a new mesh of the domain based on new_segy
        print("About to re-mesh the velocity model...", flush=True)
        remesh(segy_fname, freq_band, mesh_iter, comm)

        # point to latest mesh file
        model["mesh"]["meshfile"] = "meshes/mm_GUESS" + str(mesh_iter) + ".msh"

        # point to the latest guess velocity model
        model["mesh"]["initmodel"] = (
            "velocity_models/mm_GUESS" + str(mesh_iter) + ".hdf5"
        )
        comm.ensemble_comm.barrier()

    # Given the new mesh, we need to reinitialize some things
    mesh, V = spyro.io.read_mesh(model, comm)

    vp_guess = spyro.io.interpolate(model, mesh, V, guess=True)

    sources = spyro.Sources(model, mesh, V, comm).create()

    receivers = spyro.Receivers(model, mesh, V, comm).create()

    water = np.where(vp_guess.dat.data[:] < 1.51)

    qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)

    if comm.ensemble_comm.rank == 0:
        control_file = File(
            outdir + "control" + str(freq_band) + "Hz+.pvd", comm=comm.comm
        )
        grad_file = File(outdir + "grad" + str(freq_band) + "Hz.pvd", comm=comm.comm)

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

    if COMM_WORLD.rank == 0:
        print(
            f"START OF FWI, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo"
        )

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print(
            "INFO: Executing inversion for low-passed cut off of "
            + str(freq_band)
            + " Hz...",
            flush=True,
        )

    def _load_exact_shot(freq_band):
        for sn in range(model["acquisition"]["num_sources"]):
            if spyro.io.is_owner(comm, sn):
                shot = spyro.io.load_shots(
                    "shots/mm_exact_" + str(10.0) + "_Hz_source_" + str(sn) + ".dat"
                )
                # low-pass filter the shot record for the current frequency band.
                return spyro.utils.butter_lowpass_filter(
                    shot, freq_band, 1.0 / model["timeaxis"]["dt"]
                )

    if comm.ensemble_comm.rank == 0:
        control_file = File(
            outdir + "control" + str(freq_band) + "Hz+.pvd", comm=comm.comm
        )
        grad_file = File(outdir + "grad" + str(freq_band) + "Hz.pvd", comm=comm.comm)

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
            self.misfit = 0.0
            self.p_exact_recv = _load_exact_shot(freq_band)

        def value(self, x, tol):
            """Compute the functional"""

            gc.collect()

            if COMM_WORLD.rank == 0:
                print(
                    f"START OF ITER, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo"
                )
                # biggest_vars = muppy.sort(muppy.get_objects())[-5:]
                # sum1 = summary.summarize(biggest_vars)
                # summary.print_(sum1)
                # objgraph.show_backrefs(biggest_vars, filename='backref_start1.png')

                # self.memory_tracker = tracker.SummaryTracker()

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
                        lp_freq_index=index,
                    )
                    self.misfit = spyro.utils.evaluate_misfit(
                        model, comm, p_guess_recv, self.p_exact_recv
                    )
                    J_total[0] += spyro.utils.compute_functional(
                        model, comm, self.misfit
                    )

            # reduce over all cores
            J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
            J_total[0] /= comm.ensemble_comm.size

            gc.collect()

            if COMM_WORLD.rank == 0:
                print(
                    f"END OF FORWARD, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo"
                )
                biggest_vars = muppy.sort(muppy.get_objects())[-600:]
                sum1 = summary.summarize(biggest_vars)
                summary.print_(sum1)
                #objgraph.show_backrefs(biggest_vars[:20], filename="backref_start2.png")
                #objgraph.show_refs(biggest_vars[:20], filename="forwardref_start2.png")

            return J_total[0]

        def gradient(self, g, x, tol):
            """Compute the gradient of the functional"""

            gc.collect()

            if COMM_WORLD.rank == 0:
                print(
                    f"START OF GRAD, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mb"
                )
            #    biggest_vars = muppy.sort(muppy.get_objects())[-20:]
            #    sum1 = summary.summarize(biggest_vars)
            #    summary.print_(sum1)
            #    objgraph.show_backrefs(biggest_vars, filename='backref_end1.png')

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
                        source_num=sn,
                    )
                    dJ_local.dat.data[:] += dJ.dat.data[:]

            # sum over all ensemble members
            dJ_local.dat.data[:] = comm.ensemble_comm.allreduce(
                dJ_local.dat.data[:], op=MPI.SUM
            )
            # mask the water layer
            dJ_local.dat.data[water] = 0.0
            if comm.ensemble_comm.rank == 0:
                grad_file.write(dJ_local)
            g.scale(0)
            g.vec += dJ_local
            # switch order of misfit calculation to switch this
            g.vec *= -1

            gc.collect()

            if COMM_WORLD.rank == 0:
                print(
                    f"END OF GRADIENT, rank {comm.comm.rank}, memory usage = {get_memory_usage():.3f} Mo"
                )
                biggest_vars = muppy.sort(muppy.get_objects())[-600:]
                sum1 = summary.summarize(biggest_vars)
                summary.print_(sum1)
                #objgraph.show_backrefs(biggest_vars[:20], filename="backref_end2.png")
                #objgraph.show_refs(biggest_vars[:20], filename="forwardref_end2.png")

            # if COMM_WORLD.rank == 0:
            #    self.memory_tracker.print_diff()

        def update(self, x, flag, iteration):
            """Update the control"""
            vp_guess.assign(Function(V, x.vec, name="velocity"))
            if iteration >= 0:
                if comm.ensemble_comm.rank == 0:
                    control_file.write(vp_guess)

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

    u = Function(V, name="velocity").assign(vp_guess)
    opt = FeVector(u.vector(), inner_product)

    #xlo = Function(V)
    #xlo.interpolate(Constant(1.0))
    #x_lo = FeVector(xlo.vector(), inner_product)

    #xup = Function(V)
    #xup.interpolate(Constant(5.0))
    #x_up = FeVector(xup.vector(), inner_product)

    #bnd = ROL.Bounds(x_lo, x_up, 1.0)

    algo = ROL.Algorithm("Line Search", params)

    algo.run(opt, obj) #, bnd)

    if comm.ensemble_comm.rank == 0:
        File("res" + str(freq_band) + ".pvd", comm=comm.comm).write(obj.vp_guess)

    # interpolate vp_exact to a structured grid and write to a segy file for later meshing with SeismicMesh
    _, _, vp_i = spyro.utils.write_function_to_grid(obj.vp_guess, V, grid_spacing=10.0 / 1000.0)
    # write a new file to be used in the re-meshing
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print("writing the final velocity model...", flush=True)
        spyro.io.create_segy(vp_i, "velocity_models/mm_FINAL_"+str(freq_band))

    # important: update the control for the next frequency band to start!
    vp_guess = Function(V, opt.vec)                                                        