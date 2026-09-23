from firedrake import (Constant, div, dx, grad, inner)


def mechanical_energy_form(wave, velocity=None):
    """Return the mechanical energy of the current displacement level.

    Parameters
    ----------
    wave : `isotropic_wave.IsotropicWave`
        Elastic wave solver.
    velocity : ufl.core.expr.Expr, optional
        Expression of the velocity entering the kinetic energy. ``None``
        uses the backward difference of the two stored displacement levels,
        which is what the central-difference integrator keeps; integrators
        carrying the velocity as a field pass it instead.

    Returns
    -------
    ufl.Form
        Kinetic plus strain energy.
    """
    u_n = wave.u_n

    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    # Kinetic energy
    if velocity is None:
        velocity = (u_n - wave.u_nm1)/Constant(wave.dt)
    v = velocity
    K = (rho/2)*inner(v, v)*dx

    # Strain energy
    eps = lambda v: 0.5*(grad(v) + grad(v).T)
    U = (lmbda*div(u_n)*div(u_n) + 2*mu*inner(eps(u_n), eps(u_n)))*dx

    return K + U
