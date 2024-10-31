from firedrake import (Constant, div, dx, grad, inner)


def mechanical_energy_form(wave):
    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    # Kinetic energy
    v = (u_n - u_nm1)/dt
    K = (rho/2)*inner(v, v)*dx

    # Strain energy
    eps = lambda v: 0.5*(grad(v) + grad(v).T)
    U = (lmbda*div(u_n)*div(u_n) + 2*mu*inner(eps(u_n), eps(u_n)))*dx

    return K + U