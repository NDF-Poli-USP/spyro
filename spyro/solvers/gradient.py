from firedrake import *
from firedrake.assemble import create_assembly_callable

from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_gradient
from . import helpers

# Note this turns off non-fatal warnings
set_log_level(ERROR)

__all__ = ["gradient"]


@ensemble_gradient
def gradient(
    model, mesh, comm, c, receivers, guess, residual, output=False, save_adjoint=False
):
    """Discrete adjoint with secord-order in time fully-explicit timestepping scheme
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
    receivers: A :class:`spyro.Receivers` object.
        Contains the receiver locations and sparse interpolation methods.
    guess: A list of Firedrake functions
        Contains the forward wavefield at a set of timesteps
    residual: array-like [timesteps][receivers]
        The difference between the observed and modeled data at
        the receivers
    output: boolean
        optional, write the adjoint to disk (only for debugging)
    save_adjoint: A list of Firedrake functions
        Contains the adjoint at all timesteps

    Returns
    -------
    dJdc_local: A Firedrake.Function containing the gradient of
                the functional w.r.t. `c`
    adjoint: Optional, a list of Firedrake functions containing the adjoint

    """

    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    PML = model["BCs"]["status"]
    if PML:
        Lx = model["mesh"]["Lx"]
        Lz = model["mesh"]["Lz"]
        lx = model["BCs"]["lx"]
        lz = model["BCs"]["lz"]
        x1 = 0.0
        x2 = Lx
        a_pml = lx
        z1 = 0.0
        z2 = -Lz
        c_pml = lz
        if dim == 3:
            Ly = model["mesh"]["Ly"]
            ly = model["BCs"]["ly"]
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

    nt = int(tf / dt)  # number of timesteps

    receiver_locations = model["acquisition"]["receiver_locations"]

    dJ = Function(V, name="gradient")

    if dim == 2:
        z, x = SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)

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
        outfile = helpers.create_output_file("adjoint.pvd", comm, 0)

    t = 0.0

    # -------------------------------------------------------
    m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(rule=qr_x)
    a = c * c * dot(grad(u_n), grad(v)) * dx(rule=qr_x)  # explicit

    nf = 0
    if model["BCs"]["outer_bc"] == "non-reflective":
        nf = c * ((u_n - u_nm1) / dt) * v * ds(rule=qr_s)

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
    m_u = TrialFunction(V)
    m_v = TestFunction(V)
    mgrad = m_u * m_v * dx(rule=qr_x)

    uuadj = Function(V)  # auxiliarly function for the gradient compt.
    uufor = Function(V)  # auxiliarly function for the gradient compt.

    ffG = 2.0 * c * 1.0 * dot(grad(uuadj), grad(uufor)) * m_v * dx(rule=qr_x)

    G = mgrad - ffG
    lhsG, rhsG = lhs(G), rhs(G)

    gradi = Function(V)
    grad_prob = LinearVariationalProblem(lhsG, rhsG, gradi)
    if method == "KMV":
        grad_solver = LinearVariationalSolver(
            grad_prob,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "jacobi",
                "mat_type": "matfree",
            },
        )
    elif method == "CG":
        grad_solver = LinearVariationalSolver(
            grad_prob,
            solver_parameters={
                "mat_type": "matfree",
            },
        )

    assembly_callable = create_assembly_callable(rhs_, tensor=B)

    rhs_forcing = Function(V)  # forcing term
    if save_adjoint:
        adjoint = [Function(V, name="adjoint_pressure") for t in range(nt)]
    for step in range(nt - 1, -1, -1):
        t = step * float(dt)
        rhs_forcing.assign(0.0)
        # Solver - main equation - (I)
        # B = assemble(rhs_, tensor=B)
        assembly_callable()

        f = receivers.apply_receivers_as_source(rhs_forcing, residual, step)
        # add forcing term to solve scalar pressure
        B0 = B.sub(0)
        B0 += f

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

        # only compute for snaps that were saved
        if step % fspool == 0:
            # compute the gradient increment
            uuadj.assign(u_np1)
            uufor.assign(guess.pop())

            grad_solver.solve()
            dJ += gradi

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        if step % nspool == 0:
            if output:
                outfile.write(u_n, time=t)
            if save_adjoint:
                adjoint.append(u_n)
            helpers.display_progress(comm, t)

    if save_adjoint:
        return dJ, adjoint
    else:
        return dJ
