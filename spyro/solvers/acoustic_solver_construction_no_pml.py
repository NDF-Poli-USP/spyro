import firedrake as fire
from firedrake import dx, Constant, dot, grad


def construct_solver_or_matrix_no_pml(Wave_object):
    """Builds solver operators for wave object without a PML. Doesn't create mass matrices if
    matrix_free option is on, which it is by default.

    Parameters
    ----------
    Wave_object: :class: 'Wave' object
        Waveform object that contains all simulation parameters
    """
    V = Wave_object.function_space
    quad_rule = Wave_object.quadrature_rule

    # typical CG FEM in 2d/3d
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    u_nm1 = fire.Function(V, name="pressure t-dt")
    u_n = fire.Function(V, name="pressure")
    u_np1 = fire.Function(V, name="pressure t+dt")
    Wave_object.u_nm1 = u_nm1
    Wave_object.u_n = u_n
    Wave_object.u_np1 = u_np1

    Wave_object.current_time = 0.0
    dt = Wave_object.dt

    # -------------------------------------------------------
    m1 = (
        (1 / (Wave_object.c * Wave_object.c))
        * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
        * v
        * dx(scheme=quad_rule)
    )
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)  # explicit

    B = fire.Cofunction(V.dual())

    form = m1 + a
    lhs = fire.lhs(form)
    rhs = fire.rhs(form)
    Wave_object.lhs = lhs

    A = fire.assemble(lhs, mat_type="matfree")
    Wave_object.solver = fire.LinearSolver(
        A, solver_parameters=Wave_object.solver_parameters
    )
    # lin_var = fire.LinearVariationalProblem(lhs, rhs + B, u_np1)
    # solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    # Wave_object.solver = fire.LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)

    Wave_object.rhs = rhs
    Wave_object.B = B
