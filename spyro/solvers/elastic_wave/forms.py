from firedrake import (Cofunction, Constant, LinearVariationalProblem,
                       LinearVariationalSolver, div, dot, dx, grad, inner,
                       lhs, rhs, TestFunction, TrialFunction)

from .local_abc import local_abc_form, local_abc_velocity


def isotropic_elastic_form(wave, u, u_t, u_tt, v):
    """Return the weak form of the isotropic elastic wave equation without a PML.

    The form is written for the expressions standing for the displacement
    and its first and second time derivatives, so that every time
    integrator substitutes its own discretization of the time derivatives:
    the central-difference scheme passes finite differences of the stored
    time levels, Irksome passes symbolic time derivatives of the unknown.

    Parameters
    ----------
    wave : `isotropic_wave.IsotropicWave`
        Elastic wave solver holding the material parameters, the body forces
        and the absorbing boundary condition settings.
    u : ufl.core.expr.Expr
        Expression standing for the displacement in the stiffness and local
        absorbing boundary terms.
    u_t : ufl.core.expr.Expr
        Expression standing for the velocity, which the local absorbing
        boundary conditions act on.
    u_tt : ufl.core.expr.Expr
        Expression standing for the acceleration.
    v : firedrake.TestFunction
        Test function of the wave function space.

    Returns
    -------
    ufl.Form
        The weak form ``F`` such that ``F == 0`` is the equation, body forces
        included.
    """
    quad_rule = wave.quadrature_rule
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    F_m = rho*dot(u_tt, v)*dx(**quad_rule)

    eps = lambda v: 0.5*(grad(v) + grad(v).T)
    F_k = lmbda*div(u)*div(v)*dx(**quad_rule) \
        + 2*mu*inner(eps(u), eps(v))*dx(**quad_rule)

    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v)*dx(**quad_rule)

    F_t = local_abc_form(wave, u, u_t)

    return F_m + F_k - F_s - F_t


def isotropic_elastic_without_pml(wave):
    V = wave.function_space

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)

    # Central differences: the stiffness term is explicit, evaluated at the
    # current time level.
    F = isotropic_elastic_form(
        wave,
        u=u_n,
        u_t=local_abc_velocity(wave),
        u_tt=(u - 2*u_n + u_nm1)/(dt**2),
        v=v,
    )

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
