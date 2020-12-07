from __future__ import print_function

import firedrake as fd
import numpy as np
from firedrake import Constant, div, dx, grad, inner

from .. import io, utils
from ..domains import quadrature
from ..sources import MMS_time, timedependentSource
from . import helpers


def SSPRK3(model, mesh, comm, c, excitations, receivers, source_num=0):
    """Acoustic wave equation solved using pressure-velocity formulation
    and Strong Stability Preserving Ruge-Kutta 3.

    Parameters
    ----------
    model: Python `dictionary`
        Contains model options and parameters
    mesh: Firedrake.mesh object
        The 2D/3D triangular mesh
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
    excitations: A list Firedrake.Functions
        Each function contains an interpolated space function
        emulated a Dirac delta at the location of source `source_num`
    receivers: A :class:`Spyro.Receivers` object.
        Contains the receiver locations and sparse interpolation methods.
    source_num: `int`, optional
        The source number you wish to simulate

    Returns
    -------
    usol: list of Firedrake.Functions
        The full field solution at `fspool` timesteps
    usol_recv: array-like
        The solution interpolated to the receivers at all timesteps

    """

    freq = model["acquisition"]["frequency"]
    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dimension = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    delay = model["acquisition"]["delay"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    source_type = model["acquisition"]["source_type"]

    # Solver parameters
    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        variant = "KMV"
        raise ValueError("SSPRK not yet completely compatible with KMV")
    else:
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
        variant = "equispaced"

    if dimension == 2:
        z, x = fd.SpatialCoordinate(mesh)
    elif dimension == 3:
        z, x, y = fd.SpatialCoordinate(mesh)
    else:
        raise ValueError("Spatial dimension is correct")

    nt = int(tf / dt)  # number of timesteps
    dstep = int(delay / dt)  # number of timesteps with source

    # Element
    element = fd.FiniteElement(method, mesh.ufl_cell(), degree, variant=variant)

    # Determine which receivers are local to the subdomain
    is_local = helpers.receivers_local(mesh, dimension, receivers.receiver_locations)

    VecFS = fd.VectorFunctionSpace(mesh, element)
    ScaFS = fd.FunctionSpace(mesh, element)
    V = VecFS * ScaFS

    qr_x, qr_s, qr_k = quadrature.quadrature_rules(V)

    # Initial condition
    (q_vec, q) = fd.TestFunctions(V)
    initialU = fd.as_vector((0, 0))
    initialP = fd.Function(V.sub(1)).interpolate(0.0 * x * z)
    UP = fd.Function(V)
    u, p = UP.split()
    u.assign(initialU)
    p.interpolate(initialP)
    UP0 = fd.Function(V)
    u0, p0 = UP0.split()
    u0.assign(u)
    p0.assign(p)

    # Defining boundary conditions
    bcp = fd.DirichletBC(V.sub(1), 0.0, "on_boundary")

    dUP = fd.Function(V)
    du, dp = dUP.split()
    K1 = fd.Function(V)
    K2 = fd.Function(V)
    K3 = fd.Function(V)

    du_trial, dp_trial = fd.TrialFunctions(V)

    # create output files
    outfile = helpers.create_output_file("SSPRK3.pvd", comm, source_num)

    # Distribute shots in a circular way between processors
    if io.is_owner(comm, source_num):

        helpers.display(comm, source_num)

        # current time
        t = 0.0
        # Time-dependent source
        f = fd.Function(ScaFS)
        excitation = excitations[source_num]
        if source_type == "Ricker":
            ricker = Constant(0)
            ricker.assign(timedependentSource(model, t, freq))
            expr = excitation * ricker

            f.assign(expr)
        if source_type == "MMS":
            MMS = Constant(0)
            MMS.assign(MMS_time(t))
            expr = excitation * MMS

        f.assign(expr)
        # Setting up equations
        LHS = (1 / c ** 2) * (dp_trial) * q * dx(rule=qr_k) + inner(
            du_trial, q_vec
        ) * dx(rule=qr_k)

        RHS = inner(u, grad(q)) * dx + f * q * dx + p * div(q_vec) * dx

        # we must save the data like so
        usol = [
            fd.Function(V.sub(1), name="pressure") for t in range(nt) if t % fspool == 0
        ]
        usol_recv = []

        saveIT = 0
        prob = fd.LinearVariationalProblem(LHS, RHS, dUP, bcp)
        solv = fd.LinearVariationalSolver(prob, solver_parameters=params)

        # Evolution in time
        for IT in range(nt):
            # uptade time
            t = IT * float(dt)

            if source_type == "Ricker":
                if IT < dstep:
                    ricker.assign(timedependentSource(model, t, freq))
                    # And set the function to the excitation
                    # multiplied by the wavelet.
                    f.assign(expr)
                elif IT == dstep:
                    # source is dead
                    ricker.assign(0.0)
                    # And set the function to the excitation
                    # multiplied by the wavelet.
                    f.assign(expr)
            elif source_type == "MMS":
                MMS.assign(timedependentSource(model, t))
                # And set the function to the excitation
                # multiplied by the wavelet.
                f.assign(expr)
            else:
                raise ValueError("source not estabilished")

            # solv.solve() #Solve for du and dp
            solv.solve()  # Solve for du and dp
            K1.assign(dUP)
            k1U, k1P = K1.split()

            # Second step
            u.assign(u0 + dt * k1U)
            p.assign(p0 + dt * k1P)

            # solv.solve() #Solve for du and dp
            solv.solve()  # Solve for du and dp
            K2.assign(dUP)
            k2U, k2P = K2.split()

            # Third step
            u.assign(0.75 * u0 + 0.25 * (u + dt * k2U))
            p.assign(0.75 * p0 + 0.25 * (p + dt * k2P))

            # solve.solve() #Solve for du and dp
            solv.solve()  # Solve for du and dp
            K3.assign(dUP)
            k3U, k3P = K3.split()

            # Updating answer
            u.assign((1.0 / 3.0) * u0 + (2.0 / 3.0) * (u + dt * k3U))
            p.assign((1.0 / 3.0) * p0 + (2.0 / 3.0) * (p + dt * k3P))

            u0.assign(u)
            p0.assign(p)

            # Save to shot record every timestep
            usol_recv.append(
                receivers.interpolate(p.dat.data_ro_with_halos[:], is_local)
            )

            # Save to RAM
            if IT % fspool == 0:
                usol[saveIT].assign(p)
                saveIT += 1

            if IT % nspool == 0:
                outfile.write(p)
                helpers.display_progress(comm, t)

        usol_recv = helpers.fill(usol_recv, is_local, nt, receivers.num_receivers)
        usol_recv = utils.communicate(usol_recv, comm)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(
            "---------------------------------------------------------------",
            flush=True,
        )

    return usol, usol_recv
