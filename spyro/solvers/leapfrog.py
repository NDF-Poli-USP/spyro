from firedrake import *
from firedrake.assemble import create_assembly_callable


from .. import io, utils
from ..domains import quadrature, space
from ..pml import damping
from ..sources import FullRickerWavelet
from . import helpers

set_log_level(ERROR)

def Leapfrog(
    model, mesh, comm, c, excitations, receivers, source_num=0, lp_freq_index=0
):
    """Secord order in time fully-explicit Leapfrog scheme
    with implementation of a Perfectly Matched Layer (PML) using
    CG FEM with optionally higher order mass lumping (KMV type elements).

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

    if "amplitude" in model["acquisition"]:
        amp = model["acquisition"]["amplitude"]
    else:
        amp = 1
    freq = model["acquisition"]["frequency"]
    if "inversion" in model:
        freq_bands = model["inversion"]["freq_bands"]
    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    delay = model["acquisition"]["delay"]
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

    nt = int(tf / dt)  # number of timesteps
    dstep = int(delay / dt)  # number of timesteps with source

    if method == "KMV" or method == "Lagrange":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    elif method == "CG":
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    else:
        print (f"method: {method}")
        raise ValueError("method is not yet supported")

    if isinstance(c, function.Function):
        # V = c.function_space()
        V = excitations[0].function_space()
        element = V.ufl_element()
    # elif isinstance(c, constant.Constant):
        # element = space.FE_method(mesh, method, degree)
        # V = FunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    if dim == 2:
        z, x = SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)

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
                * ((u - u_nm1) / Constant(2.0 * dt))
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

    is_local = helpers.receivers_local(mesh, dim, receivers.receiver_locations)

    outfile = helpers.create_output_file("Leapfrog.pvd", comm, source_num)

    if io.is_owner(comm, source_num):

        t = 0.0

        cutoff = freq_bands[lp_freq_index] if "inversion" in model else None
        RW = FullRickerWavelet(dt, tf, freq, amp=amp, cutoff=cutoff)

        excitation = excitations[source_num]
        ricker = Constant(0)
        f = excitation * ricker
        ricker.assign(RW[0])
        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(rule=qr_x)
        if model["opts"]["timestepping"] == "implicit":
            a = c * c * dot(grad(u), grad(v)) * dx(rule=qr_x)
        else:
            a = c * c * dot(grad(u_n), grad(v)) * dx(rule=qr_x)

        if model["PML"]["outer_bc"] == "non-reflective":
            if model["opts"]["timestepping"] == "implicit":
                nf = c * ((u - u_nm1) / dt) * v * ds(rule=qr_s)
            else:
                nf = c * ((u_n - u_nm1) / dt) * v * ds(rule=qr_s)
        else:
            nf = 0

        FF = m1 + a + nf - f * v * dx(rule=qr_x)

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
                dd = inner(grad(u_n), dot(Gamma_2, qq)) * dx(rule=qr_x)
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
                dd1 = inner(grad(u_n), dot(Gamma_2, qq)) * dx(rule=qr_x)
                dd2 = -inner(grad(psi_n), dot(Gamma_3, qq)) * dx(rule=qr_x)

                FF += mm1 + mm2 + dd1 + dd2
                # -------------------------------------------------------
                mmm1 = (dot((psi - psi_n), phi) / Constant(dt)) * dx(rule=qr_x)
                uuu1 = (-u_n * phi) * dx(rule=qr_x)

                FF += mm1 + uuu1
        else:
            X = Function(V)
            B = Function(V)

        lhs_ = lhs(FF)
        rhs_ = rhs(FF)

        A = assemble(lhs_, mat_type="matfree")
        solver = LinearSolver(A, solver_parameters=params)

        usol = [Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
        usol_recv = []
        saveIT = 0

        assembly_callable = create_assembly_callable(rhs_, tensor=B)

        for IT in range(nt):

            if IT < dstep:
                ricker.assign(RW[IT])
            elif IT == dstep:
                ricker.assign(0.0)

            # AX=B --> solve for X = B/Aˆ-1
            # B = assemble(rhs_, tensor=B)
            assembly_callable()

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

            usol_recv.append(
                receivers.interpolate(u_n.dat.data_ro_with_halos[:], is_local)
            )

            if IT % fspool == 0:
                usol[saveIT].assign(u_n)
                saveIT += 1

            if IT % nspool == 0:
                outfile.write(u_n, time=t)
                helpers.display_progress(comm, t)

            t = IT * float(dt)

        usol_recv = helpers.fill(usol_recv, is_local, nt, receivers.num_receivers)
        usol_recv = utils.communicate(usol_recv, comm)

    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print(
            "---------------------------------------------------------------",
            flush=True,
        )

    return usol, usol_recv
