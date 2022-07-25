import firedrake as fire
from firedrake.assemble import create_assembly_callable
from firedrake import Constant, dx, dot, inner, grad, ds
import FIAT
import finat

from .. import utils
from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_forward
from . import helpers


def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())


# 3D
def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result


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
    dt = model["timeaxis"]["dt"]
    final_time = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    fspool = model["timeaxis"]["fspool"]
    excitations.current_source = source_num

    nt = int(final_time / dt)  # number of timesteps


    element = fire.FiniteElement(method, mesh.ufl_cell(), degree=degree)

    V = fire.FunctionSpace(mesh, element)


    # typical CG FEM in 2d/3d
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    u_nm1 = fire.Function(V)
    u_n = fire.Function(V, name = "pressure")
    u_np1 = fire.Function(V)

    if output:
        outfile = helpers.create_output_file("forward.pvd", comm, source_num)

    t = 0.0

    # -------------------------------------------------------
    m1 = ((u ) / Constant(dt ** 2)) * v * dx
    a = c * c * dot(grad(u_n), grad(v)) * dx +((- 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx# explicit

    X = fire.Function(V)
    B = fire.Function(V)

    lhs = m1
    rhs = -a

    A = fire.assemble(lhs)
    solver = fire.LinearSolver(A)

    usol = [fire.Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
    usol_recv = []
    save_step = 0

    assembly_callable = create_assembly_callable(rhs, tensor=B)

    rhs_forcing = fire.Function(V)

    for step in range(nt):
        rhs_forcing.assign(0.0)
        assembly_callable()
        f = excitations.apply_source(rhs_forcing, wavelet[step])
        B0 = B.sub(0)
        B0 += f
        solver.solve(X, B)

        u_np1.assign(X)

        usol_recv.append(receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

        if step % fspool == 0:
            usol[save_step].assign(u_np1)
            save_step += 1

        if step % nspool == 0:
            assert (
                fire.norm(u_n) < 1
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
