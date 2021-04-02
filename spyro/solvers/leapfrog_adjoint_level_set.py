from __future__ import print_function

from firedrake import *
from firedrake.assemble import create_assembly_callable

import numpy as np
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
    weighting,
    residual,
    source_num=0,
    output=False,
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
            sigma_x, sigma_z = damping.functions(
                model, V, dim, x, x1, x2, a_pml, z, z1, z2, c_pml
            )
            Gamma_1, Gamma_2 = damping.matrices_2D(sigma_z, sigma_x)
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

    if output:
        outfile = helpers.create_output_file(
            "Leapfrog_adjoint_level_set.pvd", comm, source_num
        )

    alpha1, alpha2 = 0.01, 0.97
    # alpha1, alpha2 = 0.0001, 0.99

    # ----------------------------------------
    # Define theta which is our descent direction
    # ---------------------------------------
    VF = VectorFunctionSpace(mesh, model["opts"]["method"], model["opts"]["degree"])
    theta = TrialFunction(VF)
    csi = TestFunction(VF)

    # -------------------------------------------------------
    m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(rule=qr_x)
    a = c * c * dot(grad(u_n), grad(v)) * dx(rule=qr_x)

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
            mm1 = (dot((pp - pp_n), qq) / dt) * dx(rule=qr_x)
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
    if PML:
        uuadj, _ = Function(W).split()  # auxiliarly function for the gradient compt.
        uufor, _ = Function(W).split()  # auxiliarly function for the gradient compt.

        uuadj_dt, _ = Function(
            W
        ).split()  # the time deriv. of the adjoint solution at timestep n
        uufor_dt, _ = Function(
            W
        ).split()  # the time deriv. of the forward solution at timestep n

        gradi_11, _ = Function(W).split()
        gradi_12, _ = Function(W).split()
        gradi_22, _ = Function(W).split()

        k0_fe0, _ = Function(W).split()

        rhs_forcing, _ = Function(W).split()  # forcing term
    else:
        uuadj = Function(V)
        uufor = Function(V)

        uuadj_dt = Function(V)
        uufor_dt = Function(V)

        gradi_11 = Function(V)
        gradi_12 = Function(V)
        gradi_22 = Function(V)

        k0_fe0 = Function(V)

        rhs_forcing = Function(V)

    G_11 = (
        (dot(grad(uuadj), grad(uufor)) - 2 * grad(uufor)[0] * grad(uuadj)[0])
        * v
        * dx(rule=qr_x)
    )
    G_12 = (
        ((-1 * grad(uufor)[0] * grad(uuadj)[1] - 1 * grad(uufor)[1] * grad(uuadj)[0]))
        * v
        * dx(rule=qr_x)
    )
    G_22 = (
        (dot(grad(uuadj), grad(uufor)) - 2 * grad(uufor)[1] * grad(uuadj)[1])
        * v
        * dx(rule=qr_x)
    )

    ke_fe0_list = []
    gradi_11_list = []
    gradi_12_list = []
    gradi_22_list = []

    assembly_callable = create_assembly_callable(rhs_, tensor=B)
    calc_grad = False
    for IT in range(nt - 1, 0, -1):
        t = IT * float(dt)

        # Solver - main equation - (I)
        assembly_callable()
        f = _adjoint_update_rhs(rhs_forcing, sparse_excitations, residual, IT, is_local)
        f *= c * c
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
        # This is reversed in sign to be consistent with
        # Antoine's and Yuri's theory on the calculation
        # of the shape derivative
        uuadj_dt.assign((u_nm1 - u_n) / float(dt))

        # compute the gradient increment
        # only compute for snaps that were saved
        if IT % fspool == 0:

            # time derivative of the full field forward solution
            uufor_dt.assign(guess_dt.pop())

            # full field forward solution
            uufor.assign(guess.pop())

            # scalar product
            ke_fe0_list.append(
                assemble(dot(uufor_dt, uuadj_dt) * v * dx(rule=qr_x)).sub(0)
            )

            # produce all the incremental gradients for
            # each component of the tensor.
            gradi_11_list.append(assemble(G_11).sub(0))
            gradi_12_list.append(assemble(G_12).sub(0))
            gradi_22_list.append(assemble(G_22).sub(0))

            if calc_grad:
                k0_fe0 += 0.5 * (ke_fe0_list[0] + ke_fe0_list[1]) * float(fspool * dt)
                gradi_11 += (
                    0.5 * (gradi_11_list[0] + gradi_11_list[1]) * float(fspool * dt)
                )
                gradi_12 += (
                    0.5 * (gradi_12_list[0] + gradi_12_list[1]) * float(fspool * dt)
                )
                gradi_22 += (
                    0.5 * (gradi_22_list[0] + gradi_22_list[1]) * float(fspool * dt)
                )
                ke_fe0_list = []
                gradi_11_list = []
                gradi_12_list = []
                gradi_22_list = []

                calc_grad = False
            else:
                calc_grad = True

        # write the adjoint to disk for checking
        if IT % nspool == 0:
            if output:
                outfile.write(u_n, time=t)
            helpers.display_progress(comm, t)

    # produces gradi_11, gradi_12, gradi_22, k0_fe0 time integrated

    # variational formulation for the descent direction
    # calculation
    # theta = descent direction vector-valued function
    # csi = test function
    # alpha1, alpha2 = scalar weights
    # wei = a Function that damps the solution near the boundary
    # very large values on the boundary
    # sigma_inside

    a = weighting * alpha1 * inner(grad(theta), grad(csi)) * dx(
        rule=qr_x
    ) + alpha2 * weighting * inner(theta, csi) * dx(rule=qr_x)

    rhs_grad = +1.0 * k0_fe0 * div(csi) * dx(rule=qr_x)

    rhs_grad += (
        -(c ** 2)
        * (
            1.0 * gradi_22 * grad(csi)[1, 1]
            + gradi_12 * (grad(csi)[0, 1] + grad(csi)[1, 0])
            + 1.0 * gradi_11 * grad(csi)[0, 0]
        )
        * dx(rule=qr_x)
    )

    L = a - rhs_grad
    lterm, rterm = lhs(L), rhs(L)
    bcval = Constant((0.0, 0.0))
    bcs = DirichletBC(VF, bcval, "on_boundary")
    Lterm, Rterm = assemble(lterm, bcs=bcs), assemble(rterm)
    solver_csi = LinearSolver(Lterm)
    descent = Function(VF, name="grad")
    solver_csi.solve(descent, Rterm)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(
            "---------------------------------------------------------------",
            flush=True,
        )

    return descent


def _adjoint_update_rhs(rhs_forcing, excitations, residual, IT, is_local):
    """Builds assembled forcing function f for adjoint for a given time_step
    given a number of receivers
    """
    recs = [recv for recv in range(excitations.shape[1]) if is_local[recv]]
    rhs_forcing.dat.data[:] = excitations[:, recs].dot(residual[IT][recs])

    return rhs_forcing
