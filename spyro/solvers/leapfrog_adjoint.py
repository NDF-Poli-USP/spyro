from __future__ import print_function

import numpy as np
from firedrake import *
from firedrake.assemble import create_assembly_callable

from scipy.sparse import csc_matrix

from ..domains import quadrature, space
from ..pml import damping
from ..sources import delta_expr, delta_expr_3d
from . import helpers

set_log_level(ERROR)

__all__ = ["Leapfrog_adjoint"]


def Leapfrog_adjoint(model, mesh, comm, c, guess, residual):
    """Discrete adjoint for secord-order in time fully-explicit Leapfrog scheme
    with implementation of a Perfectly Matched Layer (PML) using
    CG FEM with or without higher order mass lumping (KMV type elements).

    Parameters
    ----------
    model: Python `dictionary`
        Contains model options and parameters
    mesh: Firedrake.mesh object
        The 2D/3D triangular mesh
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
       c: Firedrake.Function
        The velocity model interpolated onto the mesh nodes.
    guess: A list of Firedrake functions
        Contains the forward wavefield at a set of timesteps
    residual: array-like
        The difference between the observed and modeled data at
        the receivers

    Returns
    -------
    dJdc_local: A Firedrake.Function containing the gradient of
                the functional w.r.t. `c`

    """

    numrecs = model["acquisition"]["num_receivers"]
    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    PML = model["PML"]["status"]
    if PML:
        Lx = model["mesh"]["Lx"]
        Lz = model["mesh"]["Lz"]
        lx = model["PML"]["lx"]
        lz = model["PML"]["lz"]
        x1 = 0.0
        x2 = Lx
        a_pml = lx
        z1 = 0.0
        z2 = -Lz
        c_pml = lz
        if dim == 3:
            Ly = model["mesh"]["Ly"]
            ly = model["PML"]["ly"]
            y1 = 0.0
            y2 = Ly
            b_pml = ly

    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif method == "CG":
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    else:
        raise ValueError("method is not yet supported")

    element = space.FE_method(mesh, method, degree)

    V = FunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    # Prepare receiver forcing terms
    if dim == 2:
        z, x = SpatialCoordinate(mesh)
        receiver = Constant([0, 0])
        delta = Interpolator(delta_expr(receiver, z, x), V)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)
        receiver = Constant([0, 0, 0])
        delta = Interpolator(delta_expr_3d(receiver, z, x, y), V)

    receiver_locations = model["acquisition"]["receiver_locations"]

    nt = int(tf / dt)  # number of timesteps
    timeaxis = np.linspace(model["timeaxis"]["t0"], model["timeaxis"]["tf"], nt)

    if dim == 2:
        is_local = [mesh.locate_cell([z, x]) for z, x in receiver_locations]
    elif dim == 3:
        is_local = [mesh.locate_cell([z, x, y]) for z, x, y in receiver_locations]

    dJdC_local = Function(V)

    # receivers are forced through sparse matrix vec multiplication
    sparse_excitations = csc_matrix((len(dJdC_local.dat.data), numrecs))
    for r, x0 in enumerate(receiver_locations):
        receiver.assign(x0)
        exct = delta.interpolate().dat.data_ro.copy()
        row = exct.nonzero()[0]
        col = np.repeat(r, len(row))
        sparse_exct = csc_matrix(
            (exct[row], (row, col)), shape=sparse_excitations.shape
        )
        sparse_excitations += sparse_exct

    # if using the PML
    if PML:
        Z = VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
        if dim == 2:
            W = V * Z
            u, pp = TrialFunctions(W)
            v, qq = TestFunctions(W)

            u_np1, pp_np1 = Function(W).split()
            u_n, pp_n = Function(W).split()
            u_nm1, pp_nm1 = Function(W).split()

        elif dim == 3:
            W = V * V * Z
            u, psi, pp = TrialFunctions(W)
            v, phi, qq = TestFunctions(W)

            u_np1, psi_np1, pp_np1 = Function(W).split()
            u_n, psi_n, pp_n = Function(W).split()
            u_nm1, psi_nm1, pp_nm1 = Function(W).split()

        # in 2d
        if dim == 2:
            (sigma_x, sigma_z) = damping.functions(
                model, V, dim, x, x1, x2, a_pml, z, z1, z2, c_pml
            )
            (Gamma_1, Gamma_2) = damping.matrices_2D(sigma_z, sigma_x)
            pml1 = (
                (sigma_x + sigma_z)
                * ((u - u_nm1) / (2.0 * Constant(dt)))
                * v
                * dx(rule=qr_x)
            )
        # in 3d
        elif dim == 3:

            sigma_x, sigma_y, sigma_z = damping.functions(
                model,
                V,
                dim,
                x,
                x1,
                x2,
                a_pml,
                z,
                z1,
                z2,
                c_pml,
                y,
                y1,
                y2,
                b_pml,
            )
            Gamma_1, Gamma_2, Gamma_3 = damping.matrices_3D(sigma_x, sigma_y, sigma_z)

    # typical CG in N-d
    else:
        u = TrialFunction(V)
        v = TestFunction(V)

        u_nm1 = Function(V)
        u_n = Function(V)
        u_np1 = Function(V)

    outfile = helpers.create_output_file("Leapfrog_adjoint.pvd", comm, 0)

    t = 0.0

    # -------------------------------------------------------
    m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(rule=qr_x)
    a = c * c * dot(grad(u_n), grad(v)) * dx(rule=qr_x)  # explicit

    if model["PML"]["outer_bc"] == "non-reflective":
        nf = c * ((u_n - u_nm1) / dt) * v * ds(rule=qr_s)
    else:
        nf = 0

    FF = m1 + a + nf

    if PML:
        X = Function(W)
        B = Function(W)

        if dim == 2:
            pml1 = (sigma_x + sigma_z) * ((u - u_n) / dt) * v * dx(rule=qr_x)
            pml2 = sigma_x * sigma_z * u_n * v * dx(rule=qr_x)
            pml3 = c * c * inner(grad(v), dot(Gamma_2, pp_n)) * dx(rule=qr_x)

            FF += pml1 + pml2 + pml3
            # -------------------------------------------------------
            mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dx(rule=qr_x)
            mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(rule=qr_x)
            dd = inner(qq, grad(u_n)) * dx(rule=qr_x)

            FF += mm1 + mm2 + dd
        elif dim == 3:
            pml1 = (sigma_x + sigma_y + sigma_z) * ((u - u_n) / dt) * v * dx(rule=qr_x)
            uuu1 = (-v * psi_n) * dx(rule=qr_x)
            pml2 = (
                (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
                * u_n
                * v
                * dx(rule=qr_x)
            )
            dd1 = c * c * inner(grad(v), dot(Gamma_2, pp_n)) * dx(rule=qr_x)

            FF += pml1 + pml2 + dd1 + uuu1
            # -------------------------------------------------------
            mm1 = (dot((pp - pp_n), qq) / dt) * dx(rule=qr_x)
            mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(rule=qr_x)
            pml4 = inner(qq, grad(u_n)) * dx(rule=qr_x)

            FF += mm1 + mm2 + pml4
            # -------------------------------------------------------
            pml3 = (sigma_x * sigma_y * sigma_z) * phi * u_n * dx(rule=qr_x)
            mmm1 = (dot((psi - psi_n), phi) / dt) * dx(rule=qr_x)
            mmm2 = -c * c * inner(grad(phi), dot(Gamma_3, pp_n)) * dx(rule=qr_x)

            FF += mmm1 + mmm2 + pml3
    else:
        X = Function(V)
        B = Function(V)

    lhs_ = lhs(FF)
    rhs_ = rhs(FF)

    A = assemble(lhs_, mat_type="matfree")
    solver = LinearSolver(A, solver_parameters=params)

    # Define gradient problem
    g_u = TrialFunction(V)
    g_v = TestFunction(V)

    mgrad = g_u * g_v * dx(rule=qr_x)

    uuadj = Function(V)  # auxiliarly function for the gradient compt.
    uufor = Function(V)  # auxiliarly function for the gradient compt.

    ffG = 2.0 * c * Constant(dt) * dot(grad(uuadj), grad(uufor)) * g_v * dx(rule=qr_x)

    G = mgrad - ffG
    lhsG, rhsG = lhs(G), rhs(G)

    gradi = Function(V)
    grad_prob = LinearVariationalProblem(lhsG, rhsG, gradi)
    if method == "KMV":
        grad_solv = LinearVariationalSolver(
            grad_prob,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "jacobi",
                "mat_type": "matfree",
            },
        )
    elif method == "CG":
        grad_solv = LinearVariationalSolver(
            grad_prob,
            solver_parameters={
                "mat_type": "matfree",
            },
        )

    assembly_callable = create_assembly_callable(rhs_, tensor=B)

    rhs_forcing = Function(V)  # forcing term
    for IT in range(nt - 1, -1, -1):
        t = IT * float(dt)

        # Solver - main equation - (I)
        # B = assemble(rhs_, tensor=B)
        assembly_callable()

        f = _adjoint_update_rhs(rhs_forcing, sparse_excitations, residual, IT, is_local)
        # add forcing term to solve scalar pressure
        B.sub(0).dat.data[:] += f.dat.data[:]

        # AX=B --> solve for X = B/Aˆ-1
        solver.solve(X, B)
        if PML:
            if dim == 2:
                u_np1, pp_np1 = X.split()
            elif dim == 3:
                u_np1, psi_np1, pp_np1 = X.split()

                psi_nm1.assign(psi_n)
                psi_n.assign(psi_np1)

            pp_nm1.assign(pp_n)
            pp_n.assign(pp_np1)
        else:
            u_np1.assign(X)

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        # compute the gradient increment
        uuadj.assign(u_n)

        # only compute for snaps that were saved
        if IT % fspool == 0:
            gradi.assign = 0.0
            uufor.assign(guess.pop())

            grad_solv.solve()
            dJdC_local += gradi

        if IT % nspool == 0:
            # if output:
            #    outfile.write(u_n, time=t)
            helpers.display_progress(comm, t)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(
            "---------------------------------------------------------------",
            flush=True,
        )

    return dJdC_local


def _adjoint_update_rhs(rhs_forcing, excitations, residual, IT, is_local):
    """Builds assembled forcing function f for adjoint for a given time_step
    given a number of receivers
    """
    recs = [recv for recv in range(excitations.shape[1]) if is_local[recv]]
    rhs_forcing.dat.data[:] = excitations[:, recs].dot(residual[IT][recs])

    return rhs_forcing
