
from __future__ import print_function

import firedrake as fire
import numpy as np
from firedrake import Constant, div, dx, grad, inner

from .. import io, utils
from ..domains import quadrature
from ..sources import MMS_time, timedependentSource
from . import helpers


def SSPRK(model, mesh, comm, c, excitations, receivers, source_num=0, output = True):
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
        z, x = fire.SpatialCoordinate(mesh)
    elif dimension == 3:
        z, x, y = fire.SpatialCoordinate(mesh)
    else:
        raise ValueError("Spatial dimension is correct")

    nt = int(tf / dt)  # number of timesteps
    dstep = int(delay / dt)  # number of timesteps with source

    # Element
    element = fire.FiniteElement(method, mesh.ufl_cell(), degree, variant=variant)

    # Determine which receivers are local to the subdomain
    is_local = helpers.receivers_local(mesh, dimension, receivers.receiver_locations)

    VecFS = fire.VectorFunctionSpace(mesh, element)
    ScaFS = fire.FunctionSpace(mesh, element)
    V = VecFS * ScaFS

    qr_x, qr_s, qr_k = quadrature.quadrature_rules(V)

    # Initial condition
    (q_vec, q) = fire.TestFunctions(V)
    UP = fire.Function(V)
    u, p = UP.split()
    UP0 = fire.Function(V)
    u0, p0 = UP0.split()

    # Defining boundary conditions
    bcp = fire.DirichletBC(V.sub(1), 0.0, "on_boundary")

    dUP = fire.Function(V)
    du, dp = dUP.split()
    K  = fire.Function(V)
    K1 = fire.Function(V)
    K2 = fire.Function(V)
    K3 = fire.Function(V)

    du_trial, dp_trial = fire.TrialFunctions(V)

    # create output files
    outfile = helpers.create_output_file("SSPRK3.pvd", comm, source_num)

    # Distribute shots in a circular way between processors
    if io.is_owner(comm, source_num):

        helpers.display(comm, source_num)

        # current time
        t = 0.0
        # Time-dependent source
        f = fire.Function(ScaFS)
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

        #RHS = inner(u, grad(q)) * dx + f * q * dx + p * div(q_vec) * dx
        RHS_1 = inner(u, grad(q)) * dx + p * div(q_vec) * dx
        RHS_2 = f * q * dx

        # we must save the data like so
        usol = [
            fire.Function(V.sub(1), name="pressure") for t in range(nt) if t % fspool == 0
        ]
        usol_recv = []

        saveIT = 0
        A  = fire.assemble(LHS, bcs = bcp)
        b1 = fire.assemble(RHS_1, bcs = bcp)
        b2 = fire.assemble(RHS_2, bcs = bcp)
        solv = fire.LinearSolver(A, solver_parameters=params)

        # Evolution in time
        for IT in range(nt):
            # uptade time
            t = IT * float(dt)

            if source_type == "Ricker":
                if IT < dstep:
                    ricker.assign(timedependentSource(model, t, freq))
                    f.assign(expr)
                    b2 = fire.assemble(RHS_2, bcs = bcp)
                elif IT == dstep:
                    print('Source ended.', flush = True)
                    ricker.assign(0.0)
                    f.assign(expr)
                    b2 = fire.assemble(RHS_2, bcs = bcp)
            else:
                raise ValueError("source not estabilished")

            b1 = fire.assemble(RHS_1, bcs = bcp)
            solv.solve(dUP, b1)  # Solve for du and dp
            K.assign(dUP)
            solv.solve(dUP, b2)
            K.assign(K+dUP)
            K1.assign(K)
            k1U, k1P = K1.split()

            # Second step
            u.assign(u0 + dt * k1U)
            p.assign(p0 + dt * k1P)

            # solv.solve() #Solve for du and dp
            b1 = fire.assemble(RHS_1, bcs = bcp)
            solv.solve(dUP, b1)  # Solve for du and dp
            K.assign(dUP)
            solv.solve(dUP, b2)
            K.assign(K+dUP)
            K2.assign(K)
            k2U, k2P = K2.split()

            # Third step
            u.assign(0.75 * u0 + 0.25 * (u + dt * k2U))
            p.assign(0.75 * p0 + 0.25 * (p + dt * k2P))

            # solve.solve() #Solve for du and dp
            b1 = fire.assemble(RHS_1, bcs = bcp)
            solv.solve(dUP, b1)  # Solve for du and dp
            K.assign(dUP)
            solv.solve(dUP, b2)
            K.assign(K+dUP)
            K3.assign(K)
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
                if output:
                    outfile.write(p, time=t, name="Pressure")
                helpers.display_progress(comm, t)

        usol_recv = helpers.fill(usol_recv, is_local, nt, receivers.num_receivers)
        usol_recv = utils.communicate(usol_recv, comm)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(
            "---------------------------------------------------------------",
            flush=True,
        )

    return usol, usol_recv