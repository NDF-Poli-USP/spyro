from firedrake import *
from .local_abc import local_abc_form

def isotropic_elastic_without_pml(wave):
    print("Elastic wave propagation")
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

    F_t = local_abc_form(wave)

    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())
    
##################################################################################################
def viscoelastic_maxwell_gsls_without_pml_Q(wave):
    """
    Versão generalizada da formulação viscoelástica GSLS (Maxwell) com Q,
    usando a operação C::Gamma como produto Hadamard (elemento a elemento)
    na notação de Voigt.

    Parâmetros adicionais:
        Gamma : matriz 6x6 (opcional) representando Q^{-1}_{IJ}.
                Se None, constrói uma matriz diagonal a partir de Qp_inv e Qs_inv.
                Se fornecida, pode ser cheia (anisotrópica) e substitui a construção interna.
    """
    print("Viscoelastic Maxwell/GSLS (Q-based) - Voigt notation (generalized C::Gamma)")

    V = wave.function_space
    quad = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    if wave.viscoelastic == True:
        xi_list = wave.xi_list
        y_list = wave.y_list
    else:
        xi_list = []
        y_list = []
        
    dim = V.mesh().topological_dimension()
    voigt_size = 3 if dim == 2 else 6

    C_elas = wave.C_elas
    Gamma = wave.Gamma

    # -------------------------------------------------
    # 1) Funções de conversão para Voigt
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

    def tensor_to_voigt(T):
        if dim == 2:
            return as_vector([T[0, 0], T[1, 1], T[0, 1] + T[1, 0]])
        else:
            return as_vector([T[0, 0], T[1, 1], T[2, 2],
                              T[1, 2] + T[2, 1],
                              T[0, 2] + T[2, 0],
                              T[0, 1] + T[1, 0]])

    # -------------------------------------------------
    # 2) Termo inercial
    # -------------------------------------------------
    F_m = (rho / (dt**2)) * dot(u - 2*u_n + u_nm1, v) * dx(scheme=quad)

    # Produto Hadamard (elemento a elemento) de C_elas e Gamma
    M = as_matrix([[C_elas[i, j] * Gamma[i, j] for j in range(voigt_size)] for i in range(voigt_size)])

    # -------------------------------------------------
    # 5) Deformações do deslocamento
    # -------------------------------------------------
    e_n = strain_vector_from_displacement(u_n)
    e_v = strain_vector_from_displacement(v)

    # -------------------------------------------------
    # 6) Deformação de memória acumulada
    # -------------------------------------------------
    e_mem_components = [0.0] * voigt_size
    if len(xi_list) > 0:
        for i in range(len(xi_list)):
            xi_voigt = tensor_to_voigt(xi_list[i])
            for j in range(voigt_size):
                e_mem_components[j] += y_list[i] * xi_voigt[j]
    e_mem = as_vector(e_mem_components)

    # -------------------------------------------------
    # 7) Tensão viscoelástica total em Voigt
    # -------------------------------------------------
    sigma_visco_vec = dot(C_elas, e_n) - dot(M, e_mem)

    # -------------------------------------------------
    # 8) Forma variacional
    # -------------------------------------------------
    F_k = dot(e_v, sigma_visco_vec) * dx(scheme=quad)

    # -------------------------------------------------
    # 9) Fontes e contorno
    # -------------------------------------------------
    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(scheme=quad)

    # Certifique-se de que local_abc_form está definida/importada
    F_t = local_abc_form(wave)

    # -------------------------------------------------
    # 10) Forma total e montagem
    # -------------------------------------------------
    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())

#######################################################################################    

def viscoelastic_maxwell_gsls_without_pml_Q_voigt(wave):
    print("Viscoelastic Maxwell/GSLS (Q-based) - Voigt notation")

    V = wave.function_space
    quad = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    xi_list = wave.xi_list
    y_list = wave.y_list

    Qp_inv = wave.Qp_inv
    Qs_inv = wave.Qs_inv

    dim = V.mesh().topological_dimension()
    voigt_size = 3 if dim == 2 else 6

    # -------------------------------------------------
    # 1) Funções de conversão para Voigt
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

    def tensor_to_voigt(T):
        if dim == 2:
            return as_vector([T[0, 0], T[1, 1], T[0, 1] + T[1, 0]])
        else:
            return as_vector([T[0, 0], T[1, 1], T[2, 2],
                              T[1, 2] + T[2, 1],
                              T[0, 2] + T[2, 0],
                              T[0, 1] + T[1, 0]])

    # -------------------------------------------------
    # 2) Termo inercial
    # -------------------------------------------------
    F_m = (rho / (dt**2)) * dot(u - 2*u_n + u_nm1, v) * dx(scheme=quad)

    # -------------------------------------------------
    # 3) Matriz elástica C_elas
    # -------------------------------------------------
    if dim == 2:
        C_elas = as_matrix([
            [lmbda + 2*mu, lmbda,       0],
            [lmbda,       lmbda + 2*mu, 0],
            [0,           0,            mu]
        ])
    else:
        C_elas = as_matrix([
            [lmbda + 2*mu, lmbda,       lmbda,       0,    0,    0],
            [lmbda,       lmbda + 2*mu, lmbda,       0,    0,    0],
            [lmbda,       lmbda,       lmbda + 2*mu, 0,    0,    0],
            [0,           0,           0,           mu,   0,    0],
            [0,           0,           0,           0,    mu,   0],
            [0,           0,           0,           0,    0,    mu]
        ])

    # -------------------------------------------------
    # 4) Matriz viscoelástica de memória C_Q
    # -------------------------------------------------
    kappa = lmbda + (2/3) * mu
    alpha_sq = (lmbda + 2*mu) / rho
    beta_sq = mu / rho
    denom = alpha_sq - (4/3) * beta_sq

    from ufl import conditional, lt
    Qkappa_inv = conditional(
        lt(abs(denom), 1e-12),
        Qp_inv,
        (alpha_sq * Qp_inv - (4/3) * beta_sq * Qs_inv) / denom
    )

    lmbda_Q = kappa * Qkappa_inv - (2/3) * mu * Qs_inv
    mu_Q = mu * Qs_inv

    if dim == 2:
        C_Q = as_matrix([
            [lmbda_Q + 2*mu_Q, lmbda_Q,       0],
            [lmbda_Q,          lmbda_Q + 2*mu_Q, 0],
            [0,                0,               mu_Q]
        ])
    else:
        C_Q = as_matrix([
            [lmbda_Q + 2*mu_Q, lmbda_Q,       lmbda_Q,       0,    0,    0],
            [lmbda_Q,          lmbda_Q + 2*mu_Q, lmbda_Q,       0,    0,    0],
            [lmbda_Q,          lmbda_Q,       lmbda_Q + 2*mu_Q, 0,    0,    0],
            [0,                0,             0,              mu_Q, 0,    0],
            [0,                0,             0,              0,    mu_Q, 0],
            [0,                0,             0,              0,    0,    mu_Q]
        ])

    # -------------------------------------------------
    # 5) Deformações do deslocamento
    # -------------------------------------------------
    e_n = strain_vector_from_displacement(u_n)
    e_v = strain_vector_from_displacement(v)

    # -------------------------------------------------
    # 6) Deformação de memória acumulada (CONSTRUÇÃO EXPLÍCITA)
    # -------------------------------------------------
    # Inicializa uma lista de expressões para cada componente
    e_mem_components = [0.0] * voigt_size

    if len(xi_list) > 0:
        for i in range(len(xi_list)):
            xi_voigt = tensor_to_voigt(xi_list[i])
            for j in range(voigt_size):
                e_mem_components[j] += y_list[i] * xi_voigt[j]

    # Cria o vetor UFL a partir da lista de componentes
    e_mem = as_vector(e_mem_components)

    # -------------------------------------------------
    # 7) Tensão viscoelástica total em Voigt
    # -------------------------------------------------
    sigma_visco_vec = dot(C_elas, e_n) - dot(C_Q, e_mem)

    # -------------------------------------------------
    # 8) Forma variacional
    # -------------------------------------------------
    F_k = dot(e_v, sigma_visco_vec) * dx(scheme=quad)

    # -------------------------------------------------
    # 9) Fontes e contorno
    # -------------------------------------------------
    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(scheme=quad)

    F_t = local_abc_form(wave)

    # -------------------------------------------------
    # 10) Forma total e montagem
    # -------------------------------------------------
    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())

###################################################################################################

def viscoelastic_kelvin_voigt_without_pml(wave):
    print("Viscoelastic Kelvin–Voigt")

    V = wave.function_space
    W = wave.strain_space  # TensorFunctionSpace to store deformations
    quad_rule = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)
    u_nm1 = wave.u_nm1 
    u_n = wave.u_n   

    dt = Constant(wave.dt)
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu
    eta = Constant(wave.eta) 

    eps_old = wave.eps_old 

    F_m = (rho/(dt**2))*dot(u - 2*u_n + u_nm1, v)*dx(scheme=quad_rule)
    
    epsilon = lambda v: 0.5*(grad(v) + grad(v).T)
    
    eps = epsilon(u)
    
    deps_dt = (eps - eps_old) / dt
    
    elastic = lmbda*div(u_n)*div(v)*dx(scheme=quad_rule) \
        + 2*mu*inner(epsilon(u_n), epsilon(v))*dx(scheme=quad_rule)
    
    viscous = 2 * eta * deps_dt
    
    F_k = elastic + inner(viscous, epsilon(v))*dx(scheme=quad_rule)
    
    F = F_m + F_k
    
    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v) * dx(scheme=quad_rule)

    F_t = local_abc_form(wave)

    F = F_m + F_k - F_s - F_t
    
    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())
        
##################################################################################################
    
def viscoelastic_zener_without_pml(wave):
    print("Viscoelastic Zener")

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
    tau_epsilon = wave.tau_epsilon  # Relaxation time for strain
    tau_sigma = wave.tau_sigma      # Relaxation time for stress

    eps_old = wave.eps_old          # ε^{n}, TensorFunction
    sigma_old = wave.sigma_old      # σ^{n}, TensorFunction
    
    epsilon = lambda v: 0.5*(grad(v) + grad(v).T)
    
    # Inertial term

    F_m = (rho/(dt**2))*dot(u - 2*u_n + u_nm1, v)*dx(scheme=quad_rule)
    
    def sigma_visco_zener(u, eps_old, sigma_old, dt, lmbda, mu, tau_epsilon, tau_sigma):
        dte = tau_epsilon / dt
        dts = tau_sigma / dt

        # Symmetric strain tensor
        eps = epsilon(u)

        elastic_term = lmbda * div(u) * div(v) + 2 * mu * inner(eps, epsilon(v))
        viscous_term = dte * inner(eps - eps_old, epsilon(v))
        memory_term = dts * inner(sigma_old, epsilon(v))

        return (elastic_term + viscous_term + memory_term) / (1.0 + dts)
    
    # Stiffness term
    F_k = sigma_visco_zener(u_n, eps_old, sigma_old, dt, lmbda, mu, tau_epsilon, tau_sigma)*dx(scheme=quad_rule)

    # Body force term
    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v) * dx(scheme=quad_rule)

    # Absorbing boundary condition (ABC)
    F_t = local_abc_form(wave)

    # Full weak form
    F = F_m + F_k- F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())
        
##################################################################################################

def viscoelastic_gsls_without_pml(wave):
    print("Viscoelastic GSLS")

    V = wave.function_space
    W = wave.strain_space
    quad_rule = wave.quadrature_rule

    u = TrialFunction(V)
    v = TestFunction(V)
    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    lmbda = wave.lmbda
    mu = wave.mu

    # Relaxation parameters for multiple viscoelastic branches
    tau_epsilons = wave.tau_epsilons
    tau_sigmas = wave.tau_sigmas

    eps_old_list = wave.eps_old_list
    sigma_old_list = wave.sigma_old_list

    epsilon = lambda v: 0.5 * (grad(v) + grad(v).T)
    dim = V.mesh().topological_dimension()
    I = Identity(dim)

    # Inertial term
    F_m = (rho / dt**2) * dot(u - 2*u_n + u_nm1, v) * dx(scheme=quad_rule)

    # Strain at current step
    eps_n = epsilon(u_n)

    n_branches = len(tau_epsilons)
    lmbda_share = lmbda / n_branches
    mu_share = mu / n_branches

    sigma_total = 0

    # Sum contributions of all branches to total stress
    for i, (tau_epsilon, tau_sigma) in enumerate(zip(tau_epsilons, tau_sigmas)):
        eps_old = eps_old_list[i]
        sigma_old = sigma_old_list[i]

        dte = tau_epsilon / dt
        dts = tau_sigma / dt

        elastic_term = lmbda_share * div(u_n) * I + mu_share * (grad(u_n) + grad(u_n).T)
        viscous_term = dte * (eps_n - eps_old)
        memory_term = dts * sigma_old

        sigma_branch = (elastic_term + viscous_term + memory_term) / (1.0 + dts)

        sigma_total += sigma_branch

    # Weak form of internal term
    F_k = inner(sigma_total, epsilon(v)) * dx(scheme=quad_rule)

    # External body force term
    F_s = 0
    b = wave.body_forces
    if b is not None:
        F_s += dot(b, v) * dx(scheme=quad_rule)

    # Absorbing boundary condition (ABC)
    F_t = local_abc_form(wave)

    # Full weak form
    F = F_m + F_k - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())
    
##################################################################################################

def viscoelastic_maxwell_without_pml(wave):
    print("Viscoelastic Maxwell")

    V = wave.function_space
    W = wave.strain_space
    quad = wave.quadrature_rule

    # Displacement trial/test
    u = TrialFunction(V)
    v = TestFunction(V)

    # Time states
    u_nm1 = wave.u_nm1
    u_n   = wave.u_n

    # Global parameters
    dt   = Constant(wave.dt)
    rho  = wave.rho
    lam  = wave.lmbda_s[0]
    mu   = wave.mu_s[0]

    # Branch parameters
    tau_eps = wave.tau_epsilon   # [τ_ε,m]
    tau_sig = wave.tau_sigma     # [τ_σ,m]

    eps_old   = wave.eps_old
    sigma_old = wave.sigma_old 

    dim = V.mesh().topological_dimension()
    I = Identity(dim)

    eps = lambda w: 0.5*(grad(w) + grad(w).T)  # linear strain

    # Inertial term
    F_m = (rho/dt**2) * dot(u - 2*u_n + u_nm1, v) * dx(scheme=quad)

    eps_n = eps(u_n)
    sigma_el_n = lam*tr(eps_n)*I + 2.0*mu*eps_n
    F_k_el = inner(sigma_el_n, eps(v)) * dx(scheme=quad)

    F_k_mem = 0
    F_k_mem += inner(sigma_old, eps(v)) * dx(scheme=quad)

    # External body forces
    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(scheme=quad)
        
    # Absorbing boundary condition
    F_t = local_abc_form(wave)

    # Full weak form
    F = F_m + F_k_el + F_k_mem - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())

##################################################################################################
    
def viscoelastic_maxwell_gsls_without_pml(wave):
    
    print("Viscoelastic Maxwell/GSLS")

    V = wave.function_space
    W = wave.strain_space
    quad = wave.quadrature_rule

    # Displacement trial/test
    u = TrialFunction(V)
    v = TestFunction(V)

    # Time states
    u_nm1 = wave.u_nm1
    u_n   = wave.u_n

    # Global parameters
    dt   = Constant(wave.dt)
    rho  = wave.rho
    lam  = wave.lmbda
    mu   = wave.mu
    tau = Constant(wave.taus[0])
    
    # Branch parameters
    sigma_old_list = wave.sigma_old_list  

    dim = V.mesh().topological_dimension()
    I = Identity(dim)

    eps = lambda w: 0.5*(grad(w) + grad(w).T)  # linear strain

    # Inertial term
    F_m = (rho/dt**2) * dot(u - 2*u_n + u_nm1, v) * dx(scheme=quad)

    eps_n = eps(u_n)
    sigma_el_n = lam*(1 + tau)*tr(eps_n)*I + 2.0*mu*(1 + tau)*eps_n
    F_k_el = inner(sigma_el_n, eps(v)) * dx(scheme=quad)

    F_k_mem = 0
    for sigma_old in sigma_old_list:
        F_k_mem += inner(sigma_old, eps(v)) * dx(scheme=quad)

    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(scheme=quad)

    F_t = local_abc_form(wave) 

    # Full weak form and solver
    F = F_m + F_k_el + F_k_mem - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())
#####################################################################################

def viscoelastic_maxwell_gsls_without_pml_Q_original(wave):

    print("Viscoelastic Maxwell/GSLS (Q-based)")

    V = wave.function_space
    W = wave.strain_space
    quad = wave.quadrature_rule

    # -------------------------------------------------
    # Trial / Test
    # -------------------------------------------------

    u = TrialFunction(V)
    v = TestFunction(V)

    # -------------------------------------------------
    # Time states
    # -------------------------------------------------

    u_nm1 = wave.u_nm1
    u_n   = wave.u_n

    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------

    dt   = Constant(wave.dt, domain=V.mesh())

    rho  = wave.rho
    lam  = wave.lmbda
    mu   = wave.mu

    xi_list     = wave.xi_list
    y_list      = wave.y_list
    omega_list  = wave.omega_list
    
    if wave.Q_type in ['Constant', 'constant', 'const']:
        Qp_inv = wave.Qp_inv
        Qs_inv = wave.Qs_inv
    elif wave.Q_type in ['cond', 'Conditional', 'conditional']:
        Qp_inv = wave.Qp_inv(wave.mesh)
        Qs_inv = wave.Qs_inv(wave.mesh)

    dim = V.mesh().topological_dimension()
    I = Identity(dim)

    # -------------------------------------------------
    # Operators
    # -------------------------------------------------

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    def dev(X):
        return X - (1.0/dim) * tr(X) * I

    def C_action(X):
        return lam * tr(X) * I + 2.0 * mu * X

    def C_colon_Qinv_action(X):

        trX  = tr(X)
        devX = dev(X)

        kappa = lam + 2.0 * mu / 3.0

        alpha_sq = (lam + 2.0 * mu) / rho
        beta_sq  = mu / rho

        denom = alpha_sq - (4.0/3.0) * beta_sq

        from ufl import conditional, lt

        Qkappa_inv = conditional(
            lt(abs(denom), 1e-12),
            Qp_inv,
            (
                alpha_sq * Qp_inv
                - (4.0/3.0) * beta_sq * Qs_inv
            ) / denom
        )

        return (
            kappa * Qkappa_inv * trX * I
            + 2.0 * mu * Qs_inv * devX
        )

    # -------------------------------------------------
    # Inertial term
    # -------------------------------------------------

    F_m = (
        rho / dt**2
    ) * dot(
        u - 2*u_n + u_nm1,
        v
    ) * dx(scheme=quad)

    # -------------------------------------------------
    # Elastic stress
    # -------------------------------------------------
    sigma_elastic = C_action(eps(u_n))

    # -------------------------------------------------
    # Memory contribution
    # -------------------------------------------------

    mem_term = 0.0

    for i in range(len(xi_list)):
        mem_term += y_list[i] * xi_list[i]

    sigma_memory = C_colon_Qinv_action(mem_term)

    # -------------------------------------------------
    # Total viscoelastic stress
    # -------------------------------------------------

    sigma_visco = sigma_elastic - sigma_memory

    # -------------------------------------------------
    # Variational elastic/viscoelastic term
    # -------------------------------------------------

    F_k = inner(sigma_visco, eps(v)) * dx(scheme=quad)

    # -------------------------------------------------
    # Sources
    # -------------------------------------------------

    F_s = 0

    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(scheme=quad)

    # -------------------------------------------------
    # Absorbing BC
    # -------------------------------------------------

    F_t = local_abc_form(wave)

    # -------------------------------------------------
    # Full form
    # -------------------------------------------------

    F = F_m + F_k - F_s - F_t

    # -------------------------------------------------
    # Solver
    # -------------------------------------------------

    wave.lhs = lhs(F)

    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")

    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())

#####################################################################################

def viscoelastic_maxwell(wave):
    print("Viscoelastic Maxwell (2D/3D)")

    V = wave.function_space
    W = wave.strain_space
    quad = wave.quadrature_rule

    # Trial/test
    u = TrialFunction(V)
    v = TestFunction(V)

    u_nm1 = wave.u_nm1
    u_n   = wave.u_n
    
    dt   = Constant(wave.dt)
    rho  = wave.rho
    lam  = wave.lmbda
    mu   = wave.mu

    # Parâmetros por ramo
    tau_eps = wave.tau_epsilon
    tau_sig = wave.tau_sigma

    # Estados por ramo
    eps_old   = wave.eps_old
    sigma_old = wave.sigma_old

    # Detecta dimensão automaticamente
    dim = V.mesh().topological_dimension()   # 2 ou 3
    I = Identity(dim)

    # Strain linear
    eps = lambda w: 0.5*(grad(w) + grad(w).T)

    # -----------------------------------------------
    # 1) TERMO INERCIAL
    # -----------------------------------------------
    F_m = (rho/dt**2) * dot(u - 2*u_n + u_nm1, v) * dx(scheme=quad)

    # -----------------------------------------------
    # 2) RIGIDEZ ELÁSTICA INSTANTÂNEA
    # -----------------------------------------------
    eps_n = eps(u_n)
    sigma_el_n = lam*tr(eps_n)*I + 2.0*mu*eps_n
    F_k_el = inner(sigma_el_n, eps(v)) * dx(scheme=quad)

    # -----------------------------------------------
    # 3) CONTRIBUIÇÃO DAS MEMÓRIAS
    # -----------------------------------------------
    F_k_mem = 0
    for sigma_m in sigma_old:           # loop compatível 2D/3D
        F_k_mem += inner(sigma_m, eps(v)) * dx(scheme=quad)

    # -----------------------------------------------
    # 4) FORÇAS DE CORPO E ABSORÇÃO
    # -----------------------------------------------
    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v) * dx(scheme=quad)

    F_t = local_abc_form(wave)  # se já implementado

    # -----------------------------------------------
    # 5) FORMA TOTAL E SOLVER
    # -----------------------------------------------
    F = F_m + F_k_el + F_k_mem - F_s - F_t

    wave.lhs = lhs(F)
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())

    # -----------------------------------------------
    # 6) Atualização das memórias (Backward-Euler)
    # -----------------------------------------------
    for m in range(len(sigma_old)):
        # Atualiza ε_m explicitamente se necessário
        eps_m = eps(u) if len(eps_old) == 1 else eps_old[m]  # pode ser único ou por ramo
        sigma_new = Function(W)
        # Fórmula completa Maxwell: σ_m^{n+1} = (τ_σ σ_m^n + dt * 2 μ_m ε_m^{n+1}) / (τ_σ + dt)
        sigma_new.assign((tau_sig[m]*sigma_old[m] + dt*2*mu*eps_m) / (tau_sig[m] + dt))
        sigma_old[m].assign(sigma_new)

def isotropic_elastic_with_pml():
    raise NotImplementedError
