from __future__ import print_function

from firedrake import *
import SeismicMesh

import numpy as np
from mpi4py import MPI
from scipy.sparse import csc_matrix

from ..sources import delta_expr, delta_expr_3d
from ..domains import quadrature, space
from ..pml import damping

from . import helpers


set_log_level(ERROR)

__all__ = ["Leapfrog_adjoint_level_set"]


def Leapfrog_adjoint_level_set(
    model,
    mesh,
    comm,
    c,
    guess,
    guess_dt,
    residual,
    subdomains,
    source_num=0,
):

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

    dx10, dx11 = subdomains

    # c10 = 2.0  # km/s outside circle
    # c11 = 4.5  # km/s inside circle
    c11, c10 = c

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

    dummy = Function(V)

    # receivers are forced through sparse matrix vec multiplication
    sparse_excitations = csc_matrix((len(dummy.dat.data), numrecs))
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

    outfile = helpers.create_output_file(
        "Leapfrog_adjoint_level_set.pvd", comm, source_num
    )

    # a weighting function that produces large values near the boundary
    # to diminish the gradient calculation near the boundary of the domain
    m = V.ufl_domain()
    W = VectorFunctionSpace(m, V.ufl_element())
    coords = interpolate(m.coordinates, W)
    z, x = coords.dat.data[:, 0], coords.dat.data[:, 1]

    # a weighting function that produces large values near the boundary
    # to diminish the gradient calculation near the boundary of the domain
    disk0 = SeismicMesh.Disk([-0.5, 0.5], 0.25)
    pts = np.column_stack((z[:, None], x[:, None]))
    d = disk0.eval(pts)
    d[d < 0] = 0.0
    vals = 1 + 1000.0 * d
    wei = Function(V, vals)
    File("weighting_function.pvd").write(wei)

    alpha1, alpha2 = 0.01, 0.97

    # ----------------------------------------
    # Define theta which is our descent direction
    # ---------------------------------------
    VF = VectorFunctionSpace(mesh, "CG", 1)
    theta = TrialFunction(VF)
    csi = TestFunction(VF)

    t = 0.0

    # -------------------------------------------------------
    m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(rule=qr_x)
    a10 = c10 * c10 * dot(grad(u_n), grad(v)) * dx10
    a11 = c11 * c11 * dot(grad(u_n), grad(v)) * dx11
    a = a10 + a11

    if model["PML"]["outer_bc"] == "non-reflective":
        nf = c10 * ((u_n - u_nm1) / dt) * v * ds(rule=qr_s)
    else:
        nf = 0

    FF = m1 + a + nf

    if PML:
        X = Function(W)
        B = Function(W)

        if dim == 2:
            pml1 = (sigma_x + sigma_z) * ((u - u_n) / dt) * v * dx(rule=qr_x)
            pml2 = sigma_x * sigma_z * u_n * v * dx(rule=qr_x)
            pml3 = inner(grad(v), dot(Gamma_2, pp_n)) * dx(rule=qr_x)

            FF += pml1 + pml2 + pml3
            # -------------------------------------------------------
            mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dx(rule=qr_x)
            mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(rule=qr_x)
            dd = inner(qq, grad(u_n)) * dx(rule=qr_x)

            FF += mm1 + mm2 + dd
        elif dim == 3:
            pml1 = (sigma_x + sigma_y + sigma_z) * ((u - u_n) / dt) * v * dx(rule=qr_x)
            pml2 = (
                (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
                * u_n
                * v
                * dx(rule=qr_x)
            )
            dd1 = inner(grad(v), dot(Gamma_2, pp_n)) * dx(rule=qr_x)

            FF += pml1 + pml2 + dd1
            # -------------------------------------------------------
            mm1 = (dot((pp - pp_n), qq) / dt) * dx(rule=qr_x)
            mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(rule=qr_x)
            pml4 = inner(qq, grad(u_n)) * dx(rule=qr_x)

            FF += mm1 + mm2 + pml4
            # -------------------------------------------------------
            pml3 = (sigma_x * sigma_y * sigma_z) * phi * u_n * dx(rule=qr_x)
            mmm1 = (dot((psi - psi_n), phi) / dt) * dx(rule=qr_x)
            uuu1 = (-u_n * phi) * dx(rule=qr_x)

            FF += mm1 + uuu1 + pml3
    else:
        X = Function(V)
        B = Function(V)

    lhs_ = lhs(FF)
    rhs_ = rhs(FF)

    A = assemble(lhs_, mat_type="matfree")
    solver = LinearSolver(A, solver_parameters=params)

    # Define gradient problem
    # this solves for the shape gradient WITHOUT the PML
    mgrad = u * v * dx(rule=qr_x)

    # mass matrix (note u trial function is the gradient of functional)
    uuadj = Function(V)  # auxiliarly function for the gradient compt.
    uufor = Function(V)  # auxiliarly function for the gradient compt.

    uuadj_dt = Function(V)  # the time deriv. of the adjoint solution at timestep n
    uufor_dt = Function(V)  # the time deriv. of the forward solution at timestep n

    k0_fe0 = dot(uufor_dt, uuadj_dt) * v  # defer subdomain integration until later

    ffG_11 = (
        (dot(grad(uuadj), grad(uufor)) - 2 * grad(uufor)[0] * grad(u_n)[0])
        * v
        * dx(rule=qr_x)
    )

    ffG_12 = (
        ((-2 * grad(uufor)[0] * grad(uuadj)[1] - 2 * grad(uufor)[1] * grad(uuadj)[0]))
        * v
        * dx(rule=qr_x)
    )

    ffG_21 = ffG_12

    ffG_22 = (
        (dot(grad(uuadj), grad(uufor)) - 2 * grad(uufor)[1] * grad(uuadj)[1])
        * v
        * dx(rule=qr_x)
    )

    G_11 = mgrad - ffG_11
    G_12 = mgrad - ffG_12
    G_21 = mgrad - ffG_21
    G_22 = mgrad - ffG_22

    lhsG_11, rhsG_11 = lhs(G_11), rhs(G_11)
    lhsG_12, rhsG_12 = lhs(G_12), rhs(G_12)
    lhsG_21, rhsG_21 = lhs(G_21), rhs(G_21)
    lhsG_22, rhsG_22 = lhs(G_22), rhs(G_22)

    ## Note to self, tensor is symmetric no need to compute gradi_21
    gradi_11 = Function(V)
    gradi_12 = Function(V)
    gradi_21 = Function(V)
    gradi_22 = Function(V)

    grad_prob_11 = LinearVariationalProblem(lhsG_11, rhsG_11, gradi_11)
    grad_prob_12 = LinearVariationalProblem(lhsG_12, rhsG_12, gradi_12)
    grad_prob_21 = LinearVariationalProblem(lhsG_21, rhsG_21, gradi_21)
    grad_prob_22 = LinearVariationalProblem(lhsG_22, rhsG_22, gradi_22)

    grad_solv_11 = LinearVariationalSolver(grad_prob_11, solver_parameters=params)
    grad_solv_12 = LinearVariationalSolver(grad_prob_12, solver_parameters=params)
    grad_solv_21 = LinearVariationalSolver(grad_prob_21, solver_parameters=params)
    grad_solv_22 = LinearVariationalSolver(grad_prob_22, solver_parameters=params)

    # these arrays are used for summing in parallel
    sz = len(guess[0].dat.data[:])
    gradi_11_np = np.zeros((sz))
    gradi_12_np = np.zeros((sz))
    gradi_21_np = np.zeros((sz))
    gradi_22_np = np.zeros((sz))

    k0_fe0_np = np.zeros(sz)

    uuadj = Function(V)  # auxiliarly function for the gradient compt.
    uufor = Function(V)  # auxiliarly function for the gradient compt.

    rhs_forcing = Function(V)  # forcing term
    for IT in range(nt - 1, 0, -1):
        t = IT * float(dt)

        # Solver - main equation - (I)
        B = assemble(rhs_, tensor=B)
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

        # calculate the time derivative of the adjoint here
        # which is used inside the gradient calculation
        uuadj_dt.assign((u_n - u_nm1) / float(dt))

        # compute the gradient increment
        # only compute for snaps that were saved
        if IT % fspool == 0:

            # time derivative of the full field forward solution
            uufor_dt.assign(guess_dt.pop())

            # full field forward solution
            uufor.assign(guess.pop())

            # scalar product
            k0_fe0_np += assemble(k0_fe0 * dx(rule=qr_x)).dat.data[:]

            # we assigned uufor_dt, uufor_n, uuadj_n, uufor_dt
            grad_solv_11.solve()
            grad_solv_12.solve()
            grad_solv_21.solve()
            grad_solv_22.solve()

            # produce all the incremental gradients for
            # each component of the tensor.
            gradi_11_np += gradi_11.dat.data[:]
            gradi_12_np += gradi_12.dat.data[:]
            gradi_21_np += gradi_21.dat.data[:]
            gradi_22_np += gradi_22.dat.data[:]

        # write the adjoint to disk for checking
        if IT % nspool == 0:
            outfile.write(u_n, time=t)
            helpers.display_progress(comm, t)

    # produces gradi_11, gradi_12, gradi_21, gradi_22 summed over all timesteps

    # assign the summed in time to the induvidual gradient components
    gradi_11.dat.data[:] = gradi_11_np
    gradi_12.dat.data[:] = gradi_12_np
    gradi_21.dat.data[:] = gradi_21_np
    gradi_22.dat.data[:] = gradi_22_np

    # k0_fe0 summed over all timesteps
    k0_fe0 = assemble(k0_fe0 * dx(rule=qr_x))
    k0_fe0.dat.data[:] = k0_fe0_np
    # variational formulation for the descent direction
    # calculation
    # theta = descent direction vector-valued function
    # csi = test function
    # alpha1, alpha2 = scalar weights
    # wei = a Function that damps the solution near the boundary
    # very large values on the boundary
    # sigma_inside

    a = wei * alpha1 * inner(grad(theta), grad(csi)) * dx(
        rule=qr_x
    ) + alpha2 * wei * inner(theta, csi) * dx(rule=qr_x)

    # gradient problem for two subdomains
    rhs_grad = -1.0 * (
        (1 / c10 ** 2) * k0_fe0 * div(csi) * dx10
        + (1 / c11 ** 2) * k0_fe0 * div(csi) * dx11
    )

    rhs_grad += -1.0 * (
        (
            2.0 * gradi_22 * grad(csi)[1, 1]
            + gradi_12 * (grad(csi)[0, 1] + grad(csi)[1, 0])
            + 2.0 * gradi_11 * grad(csi)[0, 0]
        )
        * dx(rule=qr_x)
    )

    L = a + rhs_grad
    lterm, rterm = lhs(L), rhs(L)
    Lterm, Rterm = assemble(lterm), assemble(rterm)
    solver_csi = LinearSolver(Lterm, solver_parameters=params)
    descent = Function(VF)
    solver_csi.solve(descent, Rterm)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(
            "---------------------------------------------------------------",
            flush=True,
        )

    return -descent


def _adjoint_update_rhs(rhs_forcing, excitations, residual, IT, is_local):
    """Builds assembled forcing function f for adjoint for a given time_step
    given a number of receivers
    """
    recs = [recv for recv in range(excitations.shape[1]) if is_local[recv]]
    rhs_forcing.dat.data[:] = excitations[:, recs].dot(residual[IT][recs])

    return rhs_forcing
