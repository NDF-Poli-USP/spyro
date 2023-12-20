from firedrake import *
from ..domains import quadrature, space
from . import helpers
from ..sources.Sources import Sources

# Note this turns off non-fatal warnings
set_log_level(ERROR)


# @ensemble_forward
def forward(model, mesh, comm, c, wavelet, source_num=0, fwi=False, **kwargs):
    """Secord-order in time fully-explicit scheme.

    Parameters
    ----------
    model: dict
        Contains model options and parameters.
    mesh: firedrake.mesh
        The 2D/3D triangular mesh
    comm: firedrake.ensemble_communicator
        The MPI communicator for parallelism
    c: firedrake.Function
        The velocity model interpolated onto the mesh.
    wavelet: array-like
        Time series data that's injected at the source location.
    source_num: `int`, optional
        The source number you wish to simulate
    fwi: `bool`, optional
        Whether this forward simulation is for FWI or not.

    Returns
    -------
    usol_recv: list
        The receiver data.
    J: float
        The functional for FWI. Only returned if `fwi=True`.
    """
    method = model["opts"]["method"]
    degree = model["opts"]["degree"]
    dim = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nspool = model["timeaxis"]["nspool"]
    receiver_points = model["acquisition"]["receiver_locations"]
    nt = int(tf / dt)  # number of timesteps
    params = set_params(method, mesh)
    element = space.FE_method(mesh, method, degree)

    V = FunctionSpace(mesh, element)

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)

    if dim == 2:
        z, x = SpatialCoordinate(mesh)
    elif dim == 3:
        z, x, y = SpatialCoordinate(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = Function(V)
    u_n = Function(V)
    u_np1 = Function(V)

    m = 1 / (c * c)
    m1 = m * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dx(scheme=qr_x)
    a = dot(grad(u_n), grad(v)) * dx(scheme=qr_x)  # explicit
    f = Function(V)
    nf = 0

    if model["BCs"]["outer_bc"] == "non-reflective":
        nf = c * ((u_n - u_nm1) / dt) * v * ds(scheme=qr_s)

    FF = m1 + a + nf - f * v * dx(scheme=qr_x)
    X = Function(V)

    lhs_ = lhs(FF)
    rhs_ = rhs(FF)

    problem = LinearVariationalProblem(lhs_, rhs_, X)
    solver = LinearVariationalSolver(problem, solver_parameters=params)

    # This part of the code is the base for receivers and sources.
    # definition
    usol_recv = []
    # Source object.
    source = Sources(model, mesh, V, comm)
    # Receiver mesh.
    vom = VertexOnlyMesh(mesh, receiver_points)
    # P0DG is the only function space you can make on a vertex-only mesh.
    P0DG = FunctionSpace(vom, "DG", 0)
    interpolator = Interpolator(u_np1, P0DG)
    if fwi:
        # Get the true receiver data.
        # In FWI, we need to calculate the objective function,
        # which requires the true receiver data.
        true_receivers = kwargs.get("true_receiver")
        # cost function
        J = 0.0
    for step in range(nt):
        f.assign(source.apply_source_based_in_vom(wavelet[step], source_num))
        solver.solve()
        u_np1.assign(X)
        # receiver function
        receivers = Function(P0DG)
        interpolator.interpolate(output=receivers)
        usol_recv.append(receivers)
        if fwi:
            J += compute_functional(receivers, true_receivers[step])
        if step % nspool == 0:
            assert (
                norm(u_n) < 1
            ), "Numerical instability. Try reducing dt or building the mesh differently"
            if float(step*dt) > 0:
                helpers.display_progress(comm, float(step*dt))
        u_nm1.assign(u_n)
        u_n.assign(u_np1)
    debug = kwargs.get("debug")
    if debug:
        # Save the solution for debugging.
        outfile = File("output.pvd")
        outfile.write(u_n)

    if fwi:
        return usol_recv, J
    else:
        return usol_recv


def compute_functional(guess_receivers, true_receivers):
    """Compute the functional for FWI.

    Parameters
    ----------
    guess_receivers : firedrake.Function
        The receivers from the forward simulation.
    true_receivers : firedrake.Function
        Supposed to be the receivers data from the true model.

    Returns
    -------
    J : float
        The functional.
    """
    misfit = guess_receivers - true_receivers
    J = 0.5 * assemble(inner(misfit, misfit) * dx)
    return J


def set_params(method, mesh):
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

    return params
