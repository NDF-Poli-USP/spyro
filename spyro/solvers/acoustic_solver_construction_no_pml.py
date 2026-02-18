import firedrake as fire
from firedrake import ds, dx, Constant, dot, grad


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
        * dx(**quad_rule)
    )
    a = dot(grad(u_n), grad(v)) * dx(**quad_rule)  # explicit

    le = 0.0
    q = Wave_object.source_expression
    if q is not None:
        le += - q * v * dx(**quad_rule)

    if Wave_object.abc_active:
        weak_expr_abc = dot((u_n - u_nm1) / Constant(dt), v)

        f_abc = (1 / Wave_object.c) * weak_expr_abc
        qr_s = Wave_object.surface_quadrature_rule

        if Wave_object.abc_boundary_layer_type == "hybrid":

            # NRBC
            le += Wave_object.cosHig * f_abc * ds(**qr_s)

            # Damping
            le += Wave_object.eta_mask * weak_expr_abc * \
                (1 / (Wave_object.c * Wave_object.c)) * \
                Wave_object.eta_habc * dx(**quad_rule)

        else:
            if Wave_object.absorb_top:
                le += f_abc*ds(1, **qr_s)
            if Wave_object.absorb_bottom:
                le += f_abc*ds(2, **qr_s)
            if Wave_object.absorb_right:
                le += f_abc*ds(3, **qr_s)
            if Wave_object.absorb_left:
                le += f_abc*ds(4, **qr_s)
            if Wave_object.dimension == 3:
                if Wave_object.absorb_front:
                    le += f_abc*ds(5, **qr_s)
                if Wave_object.absorb_back:
                    le += f_abc*ds(6, **qr_s)

    # form = m1 + a - le
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    form = m1 + a + le
    Wave_object.rhs = fire.rhs(form)
    Wave_object.lhs = fire.lhs(form)
    Wave_object.B = fire.Cofunction(V.dual())

    Wave_object.source_function = fire.Cofunction(V.dual())

    lin_var = fire.LinearVariationalProblem(
        Wave_object.lhs,
        Wave_object.rhs + Wave_object.source_function,
        u_np1, constant_jacobian=True)
    solver_parameters = dict(Wave_object.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    Wave_object.solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters,
    )
