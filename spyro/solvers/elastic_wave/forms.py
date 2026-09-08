from firedrake import *

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

def elastic_without_pml(wave):

    V = wave.function_space
    quad_rule = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
        
    dim = V.mesh().topological_dimension()

    Elastic_C = wave.Elastic_C

    # -------------------------------------------------
    # Conversion to Voigt notation
    # -------------------------------------------------
    def strain_vector_from_displacement(w):
        g = grad(w)
        if dim == 2:
            return as_vector([g[0, 0], g[1, 1], g[0, 1] + g[1, 0]])
        else:
            return as_vector([g[0, 0], g[1, 1], g[2, 2],
                              g[1, 2] + g[2, 1],
                              g[0, 2] + g[2, 0],
                              g[0, 1] + g[1, 0]])

    # -------------------------------------------------
    # Inertial term
    # -------------------------------------------------
    F_m = (rho / (dt**2)) * dot(u - 2*u_n + u_nm1, v) * dx(**quad_rule)

    # -------------------------------------------------
    # Displacement strain
    # -------------------------------------------------
    e_n = strain_vector_from_displacement(u_n)
    e_v = strain_vector_from_displacement(v)

    # -------------------------------------------------
    # Total elastic strain
    # -------------------------------------------------
    sigma_vec = dot(Elastic_C, e_n)

    # -------------------------------------------------
    # Variational form
    # -------------------------------------------------
    F_k = dot(e_v, sigma_vec) * dx(**quad_rule)
    # -------------------------------------------------
    # Sources and boundary
    # -------------------------------------------------
    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(**quad_rule)

    F_t = local_abc_form(wave)

    # -------------------------------------------------
    # Total form
    # -------------------------------------------------
    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    wave.rhs = rhs(F)
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

def viscoelastic_without_pml(wave):
    print("Viscoelastic Maxwell GSLS")

    V = wave.function_space
    quad_rule = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    
    zeta_list = wave.zeta_list
    y_list = wave.y_list
    
    dim = V.mesh().topological_dimension()
    voigt_size = 3 if dim == 2 else 6

    Elastic_C = wave.Elastic_C
    Gamma = wave.Gamma

    def strain_vector_from_displacement(w):
        g = grad(w)
        if dim == 2:
            return as_vector([g[0, 0], g[1, 1], g[0, 1] + g[1, 0]])
        else:
            return as_vector([g[0, 0], g[1, 1], g[2, 2],
                              g[1, 2] + g[2, 1],
                              g[0, 2] + g[2, 0],
                              g[0, 1] + g[1, 0]])

    def tensor_to_voigt(T):
        if dim == 2:
            return as_vector([T[0, 0], T[1, 1], T[0, 1] + T[1, 0]])
        else:
            return as_vector([T[0, 0], T[1, 1], T[2, 2],
                              T[1, 2] + T[2, 1],
                              T[0, 2] + T[2, 0],
                              T[0, 1] + T[1, 0]])

    F_m = (rho / (dt**2)) * dot(u - 2*u_n + u_nm1, v) * dx(**quad_rule)

    M = as_matrix([[Elastic_C[i, j] * Gamma[i, j] for j in range(voigt_size)] for i in range(voigt_size)])

    e_n = strain_vector_from_displacement(u_n)
    e_v = strain_vector_from_displacement(v)

    e_mem_components = [0.0] * voigt_size
        
    if len(zeta_list) > 0:
        for i in range(len(zeta_list)):
            zeta_voigt = tensor_to_voigt(zeta_list[i])
            for j in range(voigt_size):
                e_mem_components[j] += y_list[i] * zeta_voigt[j]
    e_mem = as_vector(e_mem_components)

    sigma_visco_vec = dot(Elastic_C, e_n) - dot(M, e_mem)

    F_k = dot(e_v, sigma_visco_vec) * dx(**quad_rule)

    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(**quad_rule)

    F_t = local_abc_form(wave)

    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    wave.rhs = rhs(F)
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
