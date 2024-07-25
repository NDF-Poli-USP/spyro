import numpy as np

from firedrake import (assemble, Constant, div, dot, dx, Function, grad,
                       inner, lhs, LinearSolver, rhs, TestFunction, TrialFunction)

def isotropic_elastic_without_pml(wave):
    V = wave.function_space
    quad_rule = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    def constant_wrapper(value):
        if np.isscalar(value):
            return Constant(value)
        else:
            return value
    
    dt = Constant(wave.dt)
    rho = constant_wrapper(wave.rho)
    lmbda = constant_wrapper(wave.lmbda)
    mu = constant_wrapper(wave.mu)

    F_m = (rho/(dt**2))*dot(u - 2*u_n + u_nm1, v)*dx(scheme=quad_rule)

    eps = lambda v: 0.5*(grad(v) + grad(v).T)
    F_k = lmbda*div(u_n)*div(v)*dx(scheme=quad_rule) \
          + 2*mu*inner(eps(u_n), eps(v))*dx(scheme=quad_rule)

    F = F_m + F_k

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Function(V)

def isotropic_elastic_with_pml():
    raise NotImplementedError