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
def old_forward(
    wave_object,
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

    method = "KMV"
    degree = wave_object.degree
    dim = wave_object.dimension
    dt = wave_object.dt
    tf = wave_object.final_time
    nspool = wave_object.output_frequency
    fspool = wave_object.gradient_sampling_frequency

    excitations.current_source = 0

    Lx = wave_object.length_x
    Lz = wave_object.length_z
    lx = wa
    lz = model["BCs"]["lz"]
    x1 = 0.0
    x2 = Lx
    a_pml = lx
    z1 = 0.0
    z2 = -Lz
    c_pml = lz

    nt = int(tf / dt)  # number of timesteps

    params = {"ksp_type": "preonly", "pc_type": "jacobi"}

    element = space.FE_method(mesh, method, degree)

    V = FunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    z, x = SpatialCoordinate(mesh)

    Z = VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    W = V * Z
    u, pp = TrialFunctions(W)
    v, qq = TestFunctions(W)

    u_np1, pp_np1 = Function(W).split()
    u_n, pp_n = Function(W).split()
    u_nm1, pp_nm1 = Function(W).split()

    sigma_x, sigma_z = damping.functions(
        model, V, dim, x, x1, x2, a_pml, z, z1, z2, c_pml
    )
    Gamma_1, Gamma_2 = damping.matrices_2D(sigma_z, sigma_x)
    pml1 = (
        (sigma_x + sigma_z)
        * ((u - u_nm1) / Constant(2.0 * dt))
        * v
        * dx(scheme=qr_x)
    )

    # typical CG FEM in 2d/3d

    if output:
        outfile = helpers.create_output_file("forward.pvd", comm, source_num)

    t = 0.0

    # -------------------------------------------------------
    m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dx(scheme=qr_x)
    a = c * c * dot(grad(u_n), grad(v)) * dx(scheme=qr_x)  # explicit

    nf = c * ((u_n - u_nm1) / dt) * v * ds(scheme=qr_s)

    FF = m1 + a + nf

    X = Function(W)
    B = Function(W)

    pml2 = sigma_x * sigma_z * u_n * v * dx(scheme=qr_x)
    pml3 = inner(pp_n, grad(v)) * dx(scheme=qr_x)
    FF += pml1 + pml2 + pml3
    # -------------------------------------------------------
    mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dx(scheme=qr_x)
    mm2 = inner(dot(Gamma_1, pp_n), qq) * dx(scheme=qr_x)
    dd = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dx(scheme=qr_x)
    FF += mm1 + mm2 + dd

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

        u_np1, pp_np1 = X.split()

        pp_nm1.assign(pp_n)
        pp_n.assign(pp_np1)

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