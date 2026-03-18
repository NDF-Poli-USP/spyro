import firedrake as fire
from numpy import where, log

# Work from Keith Roberts, Eduardo Moscatelli,
# Ruben Andres Salas and Alexandre Olender
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)


def forms_pml(Wave_obj, W, bc_type="nrbc"):
    '''
    Build the variational form for the wave equation with a PML.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    W : Firedrake 'MixedFunctionSpace'
        Mixed function space for the wave equation with PML
    bc_type : `str`, optional
        Type of boundary condition to apply on the PML boundaries. Options are
        "nrbc" for non-reflecting boundary conditions (Higdon or Sommerfeld)
        and "dirichlet" for Dirichlet boundary conditions. Default is "nrbc".

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
    dx = fire.dx(**Wave_obj.quadrature_rule)

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

        # Tuple of boundary ids for NRBC
        bnds = [Wave_obj.absorb_top, Wave_obj.absorb_bottom,
                Wave_obj.absorb_right, Wave_obj.absorb_left]

        # PML forms
        if Wave_obj.dimension == 2:
            pml1 = (sigma_z + sigma_x) * \
                fire.dot((u_n - u_nm1) / fire.Constant(dt), v)
            pml2 = sigma_z * sigma_x * fire.dot(u, v)
            # fire.dot(u_n, v) * dx
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
            uuu1 = -fire.dot(u_n * phi)
            FF += (mmm1 + uuu1) * dx
            # -------------------------------------------------------
            # Tuple of boundary ids for NRBC
            bnds.extend([Wave_obj.absorb_front, Wave_obj.absorb_back])

        # Apply boundary conditions to the PML boundaries.
        where_to_absorb = tuple(where(bnds)[0] + 1)  # ds starts at 1
        if bc_type == "nrbc":  # NRBC: Higdon or Sommerfeld
            qr_s = Wave_obj.surface_quadrature_rule
            f_abc = (1. / c) * fire.dot((u_n - u_nm1) / fire.Constant(dt), v)
            le = Wave_obj.cosHig * f_abc * fire.ds(where_to_absorb, **qr_s)
            FF += le
            fix_bnd = None
        elif bc_type == "dirichlet":
            fix_bnd = fire.DirichletBC(
                W.sub(0), fire.Constant(0.), where_to_absorb)

    else:
        fix_bnd = None

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

    FF, fix_bnd = forms_pml(Wave_obj, W, bc_type="dirichlet")

    Wave_obj.lhs = fire.lhs(FF)
    Wave_obj.rhs = fire.rhs(FF)
    Wave_obj.source_function = fire.Cofunction(W.dual())

    lin_var = fire.LinearVariationalProblem(
        Wave_obj.lhs, Wave_obj.rhs + Wave_obj.source_function,
        X_np1, bcs=fix_bnd, constant_jacobian=True)
    solver_parameters = dict(Wave_obj.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    Wave_obj.solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters)
