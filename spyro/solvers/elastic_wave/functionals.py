from firedrake import (Constant, div, dx, grad, inner)
from .forms import strain_tensor


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


def mechanical_energy_form_elastic(wave):
    """Mechanical energy functional for elastic wave equation.

    Parameters
    ----------
    C_tensor : `ufl.tensors.ListTensor`
        Elastic tensor
    u_ant1 : `firedrake.Function`
        Displacement field at previous timestep
    u : `firedrake.Function`
        Displacement field at current timestep

    Returns
    -------
    energy : `firedrake.Form`
        Mechanical energy functional (K + U)
    """

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.PropISO.rho
    C_tensor = wave.C_tensor

    # Kinetic energy
    v = (u_n - u_nm1) / dt
    K = Constant(1 / 2) * rho * inner(v, v) * dx

    # Strain energy
    strain = strain_tensor(u_n)
    sigma = C_tensor * strain
    U = Constant(1 / 2) * inner(sigma, strain) * dx

    return K + U
