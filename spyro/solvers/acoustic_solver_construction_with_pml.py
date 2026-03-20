import firedrake as fire

# Work from Keith Roberts, Eduardo Moscatelli,
# Ruben Andres Salas and Alexandre Olender
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)


def forms_pml(Wave_obj, W):
    '''
    Build the variational form for the wave equation with a PML.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    W : Firedrake 'MixedFunctionSpace'
        Mixed function space for the wave equation with PML

    Returns
    -------
    FF : `firedrake form`
        Variational form for the wave equation with PML
    fix_bnd : `firedrake DirichletBC'
        Dirichlet boundary conditions applied to the PML boundaries
    '''

    dt = Wave_obj.dt
    c = Wave_obj.c
    c_sqr_inv = 1. / (Wave_obj.c * Wave_obj.c)
    q_rule = Wave_obj.quadrature_rule
    dx = fire.dx(**q_rule) if q_rule else fire.dx

    # Trial and test functions, and state variables
    if Wave_obj.dimension == 2:
        u, pp = fire.TrialFunctions(W)
        v, qq = fire.TestFunctions(W)
        u_n, pp_n = Wave_obj.X_n.subfunctions
        u_nm1, _ = Wave_obj.X_nm1.subfunctions

    elif Wave_obj.dimension == 3:
        u, psi, pp = fire.TrialFunctions(W)
        v, phi, qq = fire.TestFunctions(W)
        u_n, psi_n, pp_n = Wave_obj.X_n.subfunctions
        u_nm1, psi_nm1, _ = Wave_obj.X_nm1.subfunctions

    # Acoustic form
    m1 = (c_sqr_inv * (u - 2. * u_n + u_nm1) / fire.Constant(dt**2)) * v
    a = fire.dot(fire.grad(u_n), fire.grad(v))  # explicit
    FF = (m1 + a) * dx

    if not Wave_obj.abc_get_ref_model:

        # Damping profiles and matrices
        sigma_x, sigma_z = Wave_obj.sigma_x, Wave_obj.sigma_z
        if Wave_obj.dimension == 2:
            Gamma_1, Gamma_2 = Wave_obj.damping_pml_2d()

        elif Wave_obj.dimension == 3:
            sigma_y = Wave_obj.sigma_y
            Gamma_1, Gamma_2, Gamma_3 = Wave_obj.damping_pml_3d()

        # PML forms
        if Wave_obj.dimension == 2:
            pml1 = (sigma_z + sigma_x) * \
                fire.dot((u_n - u_nm1) / fire.Constant(dt), v)
            pml2 = sigma_z * sigma_x * fire.dot(u, v)
            pml3 = -fire.dot(fire.div(pp_n), v)
            FF += c_sqr_inv * (pml1 + pml2 + pml3) * dx
            # -------------------------------------------------------
            mm1 = fire.inner((pp - pp_n) / fire.Constant(dt), qq)
            mm2 = fire.inner(fire.dot(Gamma_1, pp_n), qq)
            dd = fire.inner(Gamma_2 * fire.grad(u_n), qq)
            FF += (c_sqr_inv * (mm1 + mm2) + dd) * dx

        elif Wave_obj.dimension == 3:
            pml1 = (sigma_z + sigma_x + sigma_y) * \
                fire.dot((u_n - u_nm1) / fire.Constant(dt), v)
            pml2 = (sigma_z * sigma_x + sigma_x * sigma_y
                    + sigma_z * sigma_y) * fire.dot(u, v)
            pml3 = -fire.dot(fire.div(pp_n), v)
            pml4 = (sigma_z * sigma_x * sigma_y) * fire.dot(psi_n, v)
            FF += c_sqr_inv * (pml1 + pml2 + pml3 + pml4) * dx
            # -------------------------------------------------------
            mm1 = fire.dot((pp - pp_n) / fire.Constant(dt), qq)
            mm2 = fire.inner(fire.dot(Gamma_1, pp_n), qq)
            dd1 = fire.inner(fire.dot(Gamma_2, fire.grad(u_n)), qq)
            dd2 = -fire.inner(fire.dot(Gamma_3, fire.grad(psi_n)), qq)
            FF += (c_sqr_inv * (mm1 + mm2) + dd1 + dd2) * dx
            # -------------------------------------------------------
            mmm1 = fire.dot((psi - psi_n) / fire.Constant(dt), phi)
            uuu1 = -fire.dot(u_n, phi)
            FF += (mmm1 + uuu1) * dx

        # Apply NRBCs (Higdon or Sommerfeld) at PML boundaries
        abc_surf = Wave_obj.where_to_absorb
        if not Wave_obj.bc_boundary_pml == "Dirichlet":
            ds = fire.ds(abc_surf, **q_rule) if q_rule else fire.ds(abc_surf)
            f_abc = (fire.Constant(1.) / c) * fire.dot((
                u_n - u_nm1) / fire.Constant(dt), v)
            le = Wave_obj.cosHig * f_abc * ds
            FF += le

    # Dirichlet BCs for PML model or Neumann BCs for the reference model
    fix_bnd = fire.DirichletBC(W.sub(0), fire.Constant(0.), abc_surf) \
        if Wave_obj.bc_boundary_pml == "Dirichlet" and \
        not Wave_obj.abc_get_ref_model else None

    return FF, fix_bnd


def construct_solver_or_matrix_with_pml(Wave_obj):
    '''
    Build solver operators for wave propagator with a PML. Doesn't create
    mass matrices if matrix_free option is on, which it is by default.


    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class

    Returns
    -------
    None
    '''

    # Build mixed function space
    V = Wave_obj.function_space
    Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    Wave_obj.vector_function_space = Z

    if Wave_obj.dimension == 2:
        W = V * Z
    elif Wave_obj.dimension == 3:
        W = V * V * Z

    Wave_obj.mixed_function_space = W

    # State variables
    X_np1 = fire.Function(W)
    X_n = fire.Function(W)
    X_nm1 = fire.Function(W)

    if Wave_obj.dimension == 2:
        u_n, _ = X_n.subfunctions

    elif Wave_obj.dimension == 3:
        u_n, _, _ = X_n.subfunctions

    Wave_obj.u_n = u_n
    Wave_obj.X_np1 = X_np1
    Wave_obj.X_n = X_n
    Wave_obj.X_nm1 = X_nm1

    # Build variational forms
    FF, fix_bnd = forms_pml(Wave_obj, W)
    Wave_obj.lhs = fire.lhs(FF)
    Wave_obj.rhs = fire.rhs(FF)
    Wave_obj.source_function = fire.Cofunction(W.dual())

    # Build solver
    lin_var = fire.LinearVariationalProblem(
        Wave_obj.lhs, Wave_obj.rhs + Wave_obj.source_function,
        X_np1, bcs=fix_bnd, constant_jacobian=True)
    solver_parameters = dict(Wave_obj.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    Wave_obj.solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters)
