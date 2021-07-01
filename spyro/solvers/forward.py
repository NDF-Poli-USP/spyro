from firedrake import *
from firedrake.assemble import create_assembly_callable

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_forward
from . import helpers

# Note this turns off non-fatal warnings
set_log_level(ERROR)


@ensemble_forward
def forward(
    model,
    mesh,
    comm,
    c,
    excitations,
    wavelet,
    receivers,
    source_num=0,
    output=False,
):
    """Secord-order in time fully-explicit scheme
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
        The velocity model interpolated onto the mesh.
    excitations: A list Firedrake.Functions
    wavelet: array-like
        Time series data that's injected at the source location.
    receivers: A :class:`spyro.Receivers` object.
        Contains the receiver locations and sparse interpolation methods.
    source_num: `int`, optional
        The source number you wish to simulate
    output: `boolean`, optional
        Whether or not to write results to pvd files.

    Returns
    -------
    usol: list of Firedrake.Functions
        The full field solution at `fspool` timesteps
    usol_recv: array-like
        The solution interpolated to the receivers at all timesteps

    """

    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    PML = model["BCs"]["status"]
    excitations.current_source = source_num
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

    nt = int(tf / dt)  # number of timesteps

    if method == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif (
        method == "CG"
        and mesh.ufl_cell() != quadrilateral
        and mesh.ufl_cell() != hexahedron
    ):
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    elif method == "CG" and (
        mesh.ufl_cell() == quadrilateral or mesh.ufl_cell() == hexahedron
    ):
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    else:
        raise ValueError("method is not yet supported")

    element = space.FE_method(mesh, method, degree)

    V = FunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

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
            sigma_x, sigma_z = damping.functions(
                model, V, dim, x, x1, x2, a_pml, z, z1, z2, c_pml
            )
            Gamma_1, Gamma_2 = damping.matrices_2D(sigma_z, sigma_x)
            pml1 = (
                (sigma_x + sigma_z)
                * ((u - u_nm1) / Constant(2.0 * dt))
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

    # typical CG FEM in 2d/3d
    else:
        u = TrialFunction(V)
        v = TestFunction(V)

        u_nm1 = Function(V)
        u_n = Function(V)
        u_np1 = Function(V)

    if output:
        outfile = helpers.create_output_file("forward.pvd", comm, source_num)

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
            pml2 = sigma_x * sigma_z * u_n * v * dx(rule=qr_x)
            pml3 = inner(pp_n, grad(v)) * dx(rule=qr_x)
            FF += pml1 + pml2 + pml3
            # -------------------------------------------------------
            mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dx(rule=qr_x)
            mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(rule=qr_x)
            dd = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dx(rule=qr_x)
            FF += mm1 + mm2 + dd
        elif dim == 3:
            pml1 = (
                (sigma_x + sigma_y + sigma_z)
                * ((u - u_n) / Constant(dt))
                * v
                * dx(rule=qr_x)
            )
            pml2 = (
                (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
                * u_n
                * v
                * dx(rule=qr_x)
            )
            pml3 = (sigma_x * sigma_y * sigma_z) * psi_n * v * dx(rule=qr_x)
            pml4 = inner(pp_n, grad(v)) * dx(rule=qr_x)

            FF += pml1 + pml2 + pml3 + pml4
            # -------------------------------------------------------
            mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dx(rule=qr_x)
            mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(rule=qr_x)
            dd1 = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dx(rule=qr_x)
            dd2 = -c * c * inner(grad(psi_n), dot(Gamma_3, qq)) * dx(rule=qr_x)

            FF += mm1 + mm2 + dd1 + dd2
            # -------------------------------------------------------
            mmm1 = (dot((psi - psi_n), phi) / Constant(dt)) * dx(rule=qr_x)
            uuu1 = (-u_n * phi) * dx(rule=qr_x)

            FF += mmm1 + uuu1
    else:
        X = Function(V)
        B = Function(V)

    lhs_ = lhs(FF)
    rhs_ = rhs(FF)

    A = assemble(lhs_, mat_type="matfree")
    solver = LinearSolver(A, solver_parameters=params)

    usol = [Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
    usol_recv = []
    save_step = 0

    assembly_callable = create_assembly_callable(rhs_, tensor=B)

    rhs_forcing = Function(V)

    for step in range(nt):
        rhs_forcing.assign(0.0)
        assembly_callable()
        f = excitations.apply_source(rhs_forcing, wavelet[step])
        B0 = B.sub(0)
        B0 += f
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

        usol_recv.append(receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

        if step % fspool == 0:
            usol[save_step].assign(u_np1)
            save_step += 1

        if step % nspool == 0:
            assert (
                norm(u_n) < 1
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if output:
                outfile.write(u_n, time=t, name="Pressure")
            if t > 0:
                helpers.display_progress(comm, t)

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

    usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
    usol_recv = utils.communicate(usol_recv, comm)

    return usol, usol_recv
