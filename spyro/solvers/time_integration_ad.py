import firedrake as fire
from ..domains import quadrature


def central_difference_acoustic(forwardsolver, c, source_function):
    """
    Perform central difference time integration for wave propagation.

    Parameters:
    -----------
    forwardsolver: Spyro object
        The Wave object containing the necessary data and parameters.

    Returns:
    --------
        (receiver_data : list, J_val : float)
            Receiver data and functional value.
    """
    # Acoustic linear variational solver.
    V = forwardsolver.function_space
    dt = forwardsolver.model["timeaxis"]["dt"]
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)
    u_np1 = fire.Function(V)  # timestep n+1
    u_n = fire.Function(V)  # timestep n
    u_nm1 = fire.Function(V)  # timestep n-1

    qr_x, qr_s, _ = quadrature.quadrature_rules(V)
    time_term = (u - 2.0 * u_n + u_nm1) / \
        fire.Constant(dt**2) * v * fire.dx(scheme=qr_x)

    nf = 0
    if forwardsolver.model["absorving_boundary_conditions"]["status"] is True:
        nf = (1/c) * ((u_n - u_nm1) / dt) * v * fire.ds(scheme=qr_s)

    a = c * c * fire.dot(fire.grad(u_n), fire.grad(v)) * fire.dx(scheme=qr_x)
    F = time_term + a + nf
    lin_var = fire.LinearVariationalProblem(
        fire.lhs(F), fire.rhs(F) + source_function, u_np1)
    solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=forwardsolver._solver_parameters())
    return solver, u_np1, u_n, u_nm1
