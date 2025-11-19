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
    A = assemble(wave.lhs, bcs=wave.bcs, mat_type="matfree")
    wave.solver = LinearSolver(A, solver_parameters=wave.solver_parameters)

    wave.rhs = rhs(F)
    wave.B = Cofunction(V.dual())
    
##################################################################################################

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

    # Branch parameters
    lambda_ms    = wave.lmbda_s      # [λ_m]
    mu_ms        = wave.mu_s         # [μ_m]
    tau_eps_list = wave.tau_epsilons # [τ_ε,m]
    tau_sig_list = wave.tau_sigmas   # [τ_σ,m]

    eps_old_list   = wave.eps_old_list    
    sigma_old_list = wave.sigma_old_list  

    dim = V.mesh().topological_dimension()
    I = Identity(dim)

    eps = lambda w: 0.5*(grad(w) + grad(w).T)  # linear strain

    # Inertial term
    F_m = (rho/dt**2) * dot(u - 2*u_n + u_nm1, v) * dx(scheme=quad)

    eps_n = eps(u_n)
    sigma_el_n = lam*tr(eps_n)*I + 2.0*mu*eps_n
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

def isotropic_elastic_with_pml():
    raise NotImplementedError
