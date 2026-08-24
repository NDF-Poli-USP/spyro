"""Constructs Firedrake solver for the acosutic wave with typical BCs, NRBCs or HABCs."""

import firedrake as fire
from firedrake import ds, dx, dot, grad
from ..utils.typing import AbsorbingBCsType


def build_acoustic_form(wave, u_trial, v_test, u_n,
                        u_nm1, quad_rule, c=None, implicit=False):

    if c is None:
        c = wave.c
    dt = wave.dt
    
    m1 = (
        (1 / (c * c))
        * ((u_trial - 2.0 * u_n + u_nm1) / dt**2)
        * v_test
        * dx(**quad_rule)
    )
    stiffness_field = u_trial if implicit else u_n
    a = dot(grad(stiffness_field), grad(v_test)) * dx(**quad_rule)

    le = 0.0
    q = wave.source_expression
    if q is not None:
        le += - q * v_test * dx(**quad_rule)

    if wave.abc_active and not wave.abc_get_ref_model:
        weak_expr_abc = dot((u_n - u_nm1) / dt, v_test)

        f_abc = (1 / c) * weak_expr_abc
        qr_s = wave.surface_quadrature_rule

        if wave.abc_type == AbsorbingBCsType.HYBRID:

            # NRBC
            le += wave.cosHig * f_abc * ds(**qr_s)

            # Damping
            le += wave.eta_mask * weak_expr_abc * \
                (1 / (c * c)) * \
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
    quad_rule = wave.quadrature_rule

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

    form = build_acoustic_form(wave, u, v, u_n, u_nm1, quad_rule)

    wave.rhs = fire.rhs(form)
    wave.lhs = fire.lhs(form)
    wave.source_function = fire.Cofunction(V.dual())

    lin_var = fire.LinearVariationalProblem(
        wave.lhs,
        wave.rhs + wave.source_function,
        u_np1, constant_jacobian=True)
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    wave.solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters,
    )