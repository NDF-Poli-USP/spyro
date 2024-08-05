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

    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v)*dx(scheme=quad_rule)
    
    '''
    from firedrake import FacetNormal, perp, outer, ds
    n = FacetNormal(wave.mesh)
    t = perp(n)
    c_p = ((lmbda + 2*mu)/rho)**0.5
    c_s = (mu/rho)**0.5
    C = c_p * outer(n, n) + c_s * outer(t, t)
    qr_s = wave.surface_quadrature_rule
    F_t = rho * dot( C * ((u_n - u_nm1) / dt), v ) * ds(scheme=qr_s) # backward-difference scheme
    '''

    F_t = 0

    from firedrake import ds
    c_p = ((lmbda + 2*mu)/rho)**0.5
    c_s = (mu/rho)**0.5
    qr_s = wave.surface_quadrature_rule

    iz = 0
    ix = 1
    uz_dt = (u_n[iz] - u_nm1[iz])/dt
    ux_dt = (u_n[ix] - u_nm1[ix])/dt
    uz_dx = u_n[iz].dx(ix)
    ux_dx = u_n[ix].dx(ix)
    sig_zz = rho*c_p*(uz_dt - (c_p - 2*c_s*c_s/c_p)*ux_dx)
    sig_xz = rho*c_s*(ux_dt - c_s*uz_dx)
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*ds(2, scheme=qr_s)
    sig_zz = rho*c_p*(uz_dt + (c_p - 2*c_s*c_s/c_p)*ux_dx)
    sig_xz = rho*c_s*(ux_dt + c_s*uz_dx)
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*ds(1, scheme=qr_s)

    uz_dz = u_n[iz].dx(iz)
    ux_dz = u_n[ix].dx(iz)
    sig_zx = rho*c_s*(uz_dt - c_s*ux_dz)
    sig_xx = rho*c_p*(ux_dt - (c_p - 2*c_s*c_s/c_p)*uz_dz)
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*ds(4, scheme=qr_s)
    sig_zx = rho*c_s*(uz_dt + c_s*ux_dz)
    sig_xx = rho*c_p*(ux_dt + (c_p - 2*c_s*c_s/c_p)*uz_dz)
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*ds(3, scheme=qr_s)

    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Function(V)

def isotropic_elastic_with_pml():
    raise NotImplementedError