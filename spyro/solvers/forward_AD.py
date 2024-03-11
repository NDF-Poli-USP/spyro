from firedrake import *
from firedrake.adjoint import get_working_tape
from ..domains import quadrature, space
from . import helpers
from ..sources.Sources import Sources
from firedrake.__future__ import Interpolator, interpolate

# Note this turns off non-fatal warnings
set_log_level(ERROR)


def forward(
            model, mesh, comm, c, wavelet, receiver_mesh, source_number=0,
            fwi=False, **kwargs
    ):
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
    source_number: `int`, optional
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
    source_function = Function(V)
    nf = 0

    if model["BCs"]["outer_bc"] == "non-reflective":
        nf = c * ((u_n - u_nm1) / dt) * v * ds(scheme=qr_s)

    FF = m1 + a + nf - source_function * v * dx(scheme=qr_x)
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
    # P0DG is the only function space you can make on a vertex-only mesh.
    P0DG = FunctionSpace(receiver_mesh, "DG", 0)
    interpolator_receivers = Interpolator(u_np1, P0DG)
    interpolator_sources, forcing_point = source.apply_source_based_in_vom(source_number, V)

    def only_forward():
        for step in range(nt):
            forcing_point.dat.data[:] = wavelet[step]
            source_function.assign(1000*assemble(interpolator_sources.interpolate(forcing_point, transpose=True)
                                            ).riesz_representation(riesz_map="l2"))
            solver.solve()
            u_np1.assign(X)
            receivers = assemble(interpolator_receivers.interpolate())
            usol_recv.append(receivers)
            if step % nspool == 0:
                assert (
                    norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"
                if float(step*dt) > 0:
                    helpers.display_progress(comm, float(step*dt))
            u_nm1.assign(u_n)
            u_n.assign(u_np1)

    def for_fwi():
        true_receiver_data = kwargs.get("true_receiver_data")
        J = 0.0
        for step in get_working_tape().timestepper(iter(range(nt))):
            forcing_point.dat.data[:] = wavelet[step]
            source_function.assign(1000*assemble(interpolator_sources.interpolate(forcing_point, transpose=True)
                                            ).riesz_representation(riesz_map="l2"))
            solver.solve()
            u_np1.assign(X)
            receivers = assemble(interpolator_receivers.interpolate())
            usol_recv.append(receivers)
            if fwi:
                J += compute_functional(receivers, true_receiver_data[step], P0DG)
            if step % nspool == 0:
                assert (
                    norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"
                if float(step*dt) > 0:
                    helpers.display_progress(comm, float(step*dt))
            u_nm1.assign(u_n)
            u_n.assign(u_np1)
        return J
    debug = kwargs.get("debug")
    if debug:
        # Save the solution for debugging.
        outfile = File("output.pvd")
        outfile.write(u_n)
    if fwi:
        J = for_fwi()
        return usol_recv, J
    else:
        only_forward()
        return usol_recv


def compute_functional(guess_receivers, true_receiver_data, P0DG):
    """Compute the functional for FWI.

    Parameters
    ----------
    guess_receivers : firedrake.Function
        The receivers from the forward simulation.
    true_receiver_data : firedrake.Function
        Supposed to be the receivers data from the true model.

    Returns
    -------
    J : float
        The functional.
    """
    misfit = Function(P0DG)
    misfit.assign(guess_receivers - true_receiver_data)
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
