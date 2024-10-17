from firedrake import (assemble, Cofunction, Constant, div, dot, dx, grad,
                       inner, lhs, LinearSolver, rhs, TestFunction, TrialFunction)

from .local_abc import clayton_engquist_A1


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

    F_m = (rho/(dt**2))*dot(u - 2*u_n + u_nm1, v)*dx(scheme=quad_rule)

    eps = lambda v: 0.5*(grad(v) + grad(v).T)
    F_k = lmbda*div(u_n)*div(v)*dx(scheme=quad_rule) \
        + 2*mu*inner(eps(u_n), eps(v))*dx(scheme=quad_rule)

    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v)*dx(scheme=quad_rule)

    abc_dict = wave.input_dictionary.get("absorving_boundary_conditions", None)
    if abc_dict is None:
        F_t = 0
    else:
        abc_active = abc_dict.get("status", False)
        if abc_active:
            F_t = clayton_engquist_A1(wave)
        else:
            F_t = 0

    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())


def isotropic_elastic_with_pml():
    raise NotImplementedError
