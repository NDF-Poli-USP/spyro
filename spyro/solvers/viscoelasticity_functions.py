from firedrake import *

def epsilon(u):
    return sym(grad(u))

def sigma_visco_kelvin(u, eps_old, dt, lmbda, mu, rho, eta):
    eps = epsilon(u)
    deps_dt = (eps - eps_old) / dt
    elastic = lmbda * tr(eps) * Identity(2) + 2 * mu * eps
    viscous = 2 * eta * deps_dt
    return elastic + viscous
    
def sigma_visco_zener(eps, eps_old, sigma_old):
    dte = tau_epsilon / dt
    dts = tau_sigma / dt
    elastic_term = C(eps) + dte * (eps - eps_old)
    memory_term = dts * sigma_old
    return (elastic_term + memory_term) / (1.0 + dts)
