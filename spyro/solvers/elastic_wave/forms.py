from firedrake import (Cofunction, Constant, LinearVariationalProblem,
                       LinearVariationalSolver, div, dot, dx, grad, inner,
                       lhs, rhs, TestFunction, TrialFunction)

from .local_abc import local_abc_form


def build_elastic_form(wave, u_trial, v_test, u_n, u_nm1, quad_rule, implicit=False):
    dt = Constant(wave.dt)
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    F_m = (rho/(dt**2))*dot(u_trial - 2*u_n + u_nm1, v_test)*dx(**quad_rule)

    eps = lambda v: 0.5*(grad(v) + grad(v).T)
    stiffness_field = u_trial if implicit else u_n
    F_k = lmbda*div(stiffness_field)*div(v_test)*dx(**quad_rule) \
        + 2*mu*inner(eps(stiffness_field), eps(v_test))*dx(**quad_rule)

    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v_test)*dx(**quad_rule)

    F_t = local_abc_form(wave)

    return F_m + F_k - F_s - F_t


def isotropic_elastic_without_pml(wave):
    V = wave.function_space
    quad_rule = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    F = build_elastic_form(wave, u, v, u_n, u_nm1, quad_rule)

    wave.lhs = lhs(F)
    wave.rhs = rhs(F)
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
