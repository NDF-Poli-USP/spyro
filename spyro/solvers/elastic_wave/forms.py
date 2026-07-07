from firedrake import (Cofunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver, div,
                       dot, dx, grad, inner, lhs, replace, rhs, TestFunction,
                       TrialFunction)

from .local_abc import local_abc_form


def _set_forward_residual_form(wave, residual_form, trial_state):
    """Expose the elastic forward residual for UFL adjoint differentiation.

    The residual is written with formal state Functions that are independent of
    the live time-stepping registers.  During the adjoint replay these formal
    states are assigned from the stored forward solution before differentiating

        R(u^{n+1}, u^n, u^{n-1}; m)

    with respect to the state and material controls.
    """
    V = wave.function_space
    residual_u_np1 = Function(V, name="residual displacement t+dt")
    residual_u_n = Function(V, name="residual displacement")
    residual_u_nm1 = Function(V, name="residual displacement t-dt")
    wave.forward_residual_states = (
        residual_u_np1, residual_u_n, residual_u_nm1,
    )
    wave.forward_residual_form = replace(
        residual_form,
        {
            trial_state: residual_u_np1,
            wave.u_n: residual_u_n,
            wave.u_nm1: residual_u_nm1,
        },
    )


def isotropic_elastic_without_pml(wave):
    V = wave.function_space
    quad_rule = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    F_m = (rho/(dt**2))*dot(u - 2*u_n + u_nm1, v)*dx(**quad_rule)

    eps = lambda v: 0.5*(grad(v) + grad(v).T)
    F_k = lmbda*div(u_n)*div(v)*dx(**quad_rule) \
        + 2*mu*inner(eps(u_n), eps(v))*dx(**quad_rule)

    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v)*dx(**quad_rule)

    F_t = local_abc_form(wave)

    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    wave.rhs = rhs(F)
    _set_forward_residual_form(wave, F, u)
    wave.B = Cofunction(V.dual())
    wave.source_function = Cofunction(V.dual())

    lin_var = LinearVariationalProblem(
        wave.lhs,
        wave.rhs + wave.source_function,
        wave.u_np1,
        bcs=wave.bcs,
        constant_jacobian=True,
    )
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    wave.solver = LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters
    )


def isotropic_elastic_with_pml():
    raise NotImplementedError
