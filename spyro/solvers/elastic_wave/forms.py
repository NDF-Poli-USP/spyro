from firedrake import (Cofunction, Constant, LinearVariationalProblem,
                       LinearVariationalSolver, div, dot, dx, grad, inner,
                       lhs, rhs, TestFunction, TrialFunction, as_vector)

from .local_abc import local_abc_form


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


def strain_tensor(u):
    """Compute the strain tensor in Voight notation.

    Parameters
    ----------
    u : `firedrake.Function`
        Displacement field

    Returns
    -------
    eps_tensor : `ufl.tensors.ListTensor`
        Strain tensor in Voight notation
    """

    # Components
    eps_x = u[0].dx(0)
    eps_y = u[1].dx(1)
    eps_z = u[2].dx(2)
    gamma_xy = u[0].dx(1) + u[1].dx(0)
    gamma_yz = u[1].dx(2) + u[2].dx(1)
    gamma_xz = u[0].dx(2) + u[2].dx(0)

    # Assembling the strain
    eps_tensor = as_vector((eps_x, eps_y, eps_z, gamma_yz, gamma_xz, gamma_xy))

    return eps_tensor


def anisotropic_elastic_without_pml(wave):
    V = wave.function_space
    quad_rule = wave.quadrature_rule

    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # State variables
    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    # Time step
    dt = Constant(wave.dt)

    # Mass density
    rho = wave.PropISO.rho

    # Strain tensor
    eps_tensor = strain_tensor(u)

    # Virtual strain tensor
    eps_tensor_v = strain_tensor(v)

    # Cauchy's stress tensor
    sigma = wave.C_tensor * eps_tensor

    # Variational problem
    F_k = inner(sigma, eps_tensor_v) * dx(**quad_rule)
    F_m = (1 / dt ** 2) * rho * inner(u - 2 * u_n + u_nm1, v) * dx(**quad_rule)
    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v)*dx(**quad_rule)
    F_t = local_abc_form(wave)
    F = F_m + F_k - F_s - F_t

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
