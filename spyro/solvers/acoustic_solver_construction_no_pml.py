"""Constructs Firedrake solver for the acosutic wave with typical BCs, NRBCs or HABCs."""

import firedrake as fire
from firedrake import ds, dx, dot, grad
from ..utils.typing import AbsorbingBCsType


def acoustic_form_no_pml(wave, u, u_t, u_tt, v):
    """Return the weak form of the acoustic wave equation without a PML.

    The form is written for the expressions standing for the pressure and
    its first and second time derivatives, so that every time integrator
    substitutes its own discretization of the time derivatives: the
    central-difference scheme passes finite differences of the stored time
    levels, Irksome passes symbolic time derivatives of the unknown.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    u : ufl.core.expr.Expr
        Expression standing for the pressure in the stiffness term.
    u_t : ufl.core.expr.Expr
        Expression standing for the first time derivative of the pressure,
        which the absorbing boundary conditions act on.
    u_tt : ufl.core.expr.Expr
        Expression standing for the second time derivative of the pressure.
    v : firedrake.TestFunction
        Test function of the wave function space.

    Returns
    -------
    ufl.Form
        The weak form ``F`` such that ``F == 0`` is the equation, with the
        UFL source expression of the solver already included.
    """
    quad_rule = wave.quadrature_rule

    m1 = (1 / (wave.c * wave.c)) * u_tt * v * dx(**quad_rule)
    a = dot(grad(u), grad(v)) * dx(**quad_rule)

    le = 0.0
    q = wave.source_expression
    if q is not None:
        le += - q * v * dx(**quad_rule)

    if wave.abc_active and not wave.abc_get_ref_model:
        weak_expr_abc = dot(u_t, v)

        f_abc = (1 / wave.c) * weak_expr_abc
        qr_s = wave.surface_quadrature_rule

        if wave.abc_type == AbsorbingBCsType.HYBRID:

            # NRBC
            le += wave.cosHig * f_abc * ds(**qr_s)

            # Damping
            le += wave.eta_mask * weak_expr_abc * \
                (1 / (wave.c * wave.c)) * \
                wave.eta_habc * dx(**quad_rule)

        else:
            if wave.absorb_top:
                le += f_abc*ds(1, **qr_s)
            if wave.absorb_bottom:
                le += f_abc*ds(2, **qr_s)
            if wave.absorb_right:
                le += f_abc*ds(3, **qr_s)
            if wave.absorb_left:
                le += f_abc*ds(4, **qr_s)
            if wave.dimension == 3:
                if wave.absorb_front:
                    le += f_abc*ds(5, **qr_s)
                if wave.absorb_back:
                    le += f_abc*ds(6, **qr_s)

    # form = m1 + a - le
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    # TODO: Add citation
    return m1 + a + le


def construct_solver_or_matrix_no_pml(wave):
    """Builds solver operators for wave propagator with typical BCs, NRBCs or HABCs.

    Doesn't create mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """
    V = wave.function_space

    # typical CG FEM in 2d/3d
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    u_nm1 = fire.Function(V, name="pressure t-dt")
    u_n = fire.Function(V, name="pressure")
    u_np1 = fire.Function(V, name="pressure t+dt")
    wave.u_nm1 = u_nm1
    wave.u_n = u_n
    wave.u_np1 = u_np1

    wave.current_time = 0.0
    dt = wave.dt

    # -------------------------------------------------------
    # Central differences: the stiffness term is explicit, evaluated at the
    # current time level.
    form = acoustic_form_no_pml(
        wave,
        u=u_n,
        u_t=(u_n - u_nm1) / dt,
        u_tt=(u - 2.0 * u_n + u_nm1) / dt**2,
        v=v,
    )
    wave.rhs = fire.rhs(form)
    wave.lhs = fire.lhs(form)
    wave.source_function = fire.Cofunction(V.dual())

    lin_var = fire.LinearVariationalProblem(
        wave.lhs,
        wave.rhs + wave.source_function,
        u_np1, bcs=wave.bcs, constant_jacobian=True)
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    wave.solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters,
    )
