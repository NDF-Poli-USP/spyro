import firedrake as fire
from numpy import where

# Work from Keith Roberts, Eduardo Moscatelli,
# Ruben Andres Salas and Alexandre Olender
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)


def construct_solver_or_matrix_with_pml(Wave_object):
    '''
    Build solver operators for wave propagator with a PML. Doesn't create
    mass matrices if matrix_free option is on, which it is by default.


    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class

    Returns
    -------
    solver: Firedrake 'LinearSolver'
        Linear solver for the wave equation with PML
    '''
    V = Wave_object.function_space
    Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    Wave_object.vector_function_space = Z
    if Wave_object.dimension == 2:
        return construct_solver_or_matrix_with_pml_2d(Wave_object)
    elif Wave_object.dimension == 3:
        return construct_solver_or_matrix_with_pml_3d(Wave_object)


def damping_pml_2d(sigma_z, sigma_x):
    '''
    Build damping matrices for a two-dimensional problem using PML.

    Parameters
    ----------
    sigma_z: Firedrake 'Function'
        Damping profile in the z direction
    sigma_x: Firedrake 'Function'
        Damping profile in the x direction

    Returns
    -------
    Gamma_1: Firedrake 'TensorFunction'
        First damping matrix
    Gamma_2: Firedrake 'TensorFunction'
        Second damping matrix
    '''
    Gamma_1 = fire.as_tensor([[sigma_z, 0.], [0., sigma_x]])
    Gamma_2 = fire.as_tensor([[sigma_z - sigma_x, 0.],
                              [0., sigma_x - sigma_z]])

    return Gamma_1, Gamma_2


def damping_pml_3d(sigma_z, sigma_x, sigma_y):
    '''
    Build  Damping matrices for a three-dimensional problem using PML.

    Parameters
    ----------
    sigma_z: Firedrake 'Function'
        Damping profile in the z direction
    sigma_x: Firedrake 'Function'
        Damping profile in the x direction
    sigma_y: Firedrake 'Function'
        Damping profile in the y direction

    Returns
    -------
    Gamma_1: Firedrake 'TensorFunction'
        First damping matrix
    Gamma_2: Firedrake 'TensorFunction'
        Second damping matrix
    Gamma_3: Firedrake 'TensorFunction'
        Third damping matrix
    '''
    Gamma_1 = fire.as_tensor([
        [sigma_z, 0., 0.], [0., sigma_x, 0.], [0., 0., sigma_y]])
    Gamma_2 = fire.as_tensor([[sigma_z - sigma_x - sigma_y, 0., 0.],
                              [0., sigma_x - sigma_z - sigma_y, 0.],
                              [0., 0., sigma_y - sigma_x - sigma_z]])
    Gamma_3 = fire.as_tensor([[sigma_x * sigma_y, 0., 0.],
                              [0., sigma_z * sigma_y, 0.],
                              [0., 0., sigma_z * sigma_x]])

    return Gamma_1, Gamma_2, Gamma_3


def Dirichlet_bc_pml(Wave_object, W):
    '''
    Apply Dirichlet boundary conditions to the PML boundaries.

    Parameters
    ----------
    Wave_object : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    W : Firedrake 'MixedFunctionSpace'
        Mixed function space for the wave equation with PML

    Returns
    -------
    fix_bnd : Firedrake 'DirichletBC'
        Dirichlet boundary conditions applied to the PML boundaries
    '''

    bnds = [Wave_object.absorb_top, Wave_object.absorb_bottom,
            Wave_object.absorb_right, Wave_object.absorb_left]

    if Wave_object.dimension == 3:
        bnds.extend([Wave_object.absorb_front,
                     Wave_object.absorb_back])

    # Tuple of boundary ids for Dirichlet BC
    where_to_absorb = tuple(where(bnds)[0] + 1)  # ds starts at 1

    # Boundary nodes indices
    fix_bnd = fire.DirichletBC(W.sub(0), fire.Constant(0.), where_to_absorb)

    return fix_bnd


def construct_solver_or_matrix_with_pml_2d(Wave_object):
    '''
    Builds solver operators for 2D wave propagator with a PML. Doesn't create
    mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class

    Returns
    -------
    None
    '''
    dt = Wave_object.dt
    c_sqr_inv = 1. / (Wave_object.c * Wave_object.c)

    V = Wave_object.function_space
    Z = Wave_object.vector_function_space
    W = V * Z
    Wave_object.mixed_function_space = W
    dx = fire.dx(scheme=Wave_object.quadrature_rule)
    # ds = fire.ds(scheme=Wave_object.surface_quadrature_rule)

    u, pp = fire.TrialFunctions(W)
    v, qq = fire.TestFunctions(W)

    X = fire.Function(W)
    X_n = fire.Function(W)
    X_nm1 = fire.Function(W)

    u_n, pp_n = X_n.subfunctions
    u_nm1, _ = X_nm1.subfunctions

    Wave_object.u_n = u_n
    Wave_object.X = X
    Wave_object.X_n = X_n
    Wave_object.X_nm1 = X_nm1

    sigma_x, sigma_z = Wave_object.sigma_x, Wave_object.sigma_z
    Gamma_1, Gamma_2 = damping_pml_2d(sigma_z, sigma_x)

    B = fire.Cofunction(W.dual())
    # -------------------------------------------------------
    m1 = (c_sqr_inv * (u - 2. * u_n + u_nm1) / fire.Constant(dt**2)) * v * dx
    a = fire.dot(fire.grad(u_n), fire.grad(v)) * dx  # explicit
    FF = m1 + a
    # -------------------------------------------------------
    pml1 = c_sqr_inv * (sigma_z + sigma_x) * \
        fire.dot((u_n - u_nm1) / fire.Constant(dt), v) * dx
    pml2 = c_sqr_inv * sigma_z * sigma_x * fire.dot(u, v) * dx
    # fire.dot(u_n, v) * dx
    pml3 = -c_sqr_inv * fire.dot(fire.div(pp_n), v) * dx
    FF += pml1 + pml2 + pml3
    # -------------------------------------------------------
    mm1 = c_sqr_inv * fire.inner((pp - pp_n) / fire.Constant(dt), qq) * dx
    mm2 = c_sqr_inv * fire.inner(fire.dot(Gamma_1, pp_n), qq) * dx
    dd = fire.inner(Gamma_2 * fire.grad(u_n), qq) * dx
    FF += mm1 + mm2 + dd
    # -------------------------------------------------------

    lhs_ = fire.lhs(FF)
    rhs_ = fire.rhs(FF)

    # Apply Dirichlet BCs to the PML boundaries
    fix_bnd = Dirichlet_bc_pml(Wave_object, W)

    A = fire.assemble(lhs_, mat_type="matfree", bcs=fix_bnd)
    solver = fire.LinearSolver(
        A, solver_parameters=Wave_object.solver_parameters)
    Wave_object.solver = solver
    Wave_object.rhs = rhs_
    Wave_object.B = B


def construct_solver_or_matrix_with_pml_3d(Wave_object):
    """
    Builds solver operators for 3D wave propagator with a PML. Doesn't create
    mass matrices if matrix_free option is on, which it is by default.
    """
    dt = Wave_object.dt
    c_sqr_inv = 1. / (Wave_object.c * Wave_object.c)

    V = Wave_object.function_space
    Z = Wave_object.vector_function_space
    W = V * V * Z
    Wave_object.mixed_function_space = W
    dx = fire.dx(scheme=Wave_object.quadrature_rule)
    # ds = fire.ds(scheme=Wave_object.surface_quadrature_rule)

    u, psi, pp = fire.TrialFunctions(W)
    v, phi, qq = fire.TestFunctions(W)

    X = fire.Function(W)
    X_n = fire.Function(W)
    X_nm1 = fire.Function(W)

    u_n, psi_n, pp_n = X_n.subfunctions
    u_nm1, psi_nm1, _ = X_nm1.subfunctions

    Wave_object.u_n = u_n
    Wave_object.X = X
    Wave_object.X_n = X_n
    Wave_object.X_nm1 = X_nm1

    sigma_x, sigma_z = Wave_object.sigma_x, Wave_object.sigma_z
    sigma_y = Wave_object.sigma_y
    Gamma_1, Gamma_2, Gamma_3 = damping_pml_3d(sigma_z, sigma_x, sigma_y)

    B = fire.Cofunction(W.dual())
    # -------------------------------------------------------
    m1 = (c_sqr_inv * (u - 2. * u_n + u_nm1) / fire.Constant(dt**2)) * v * dx
    a = fire.dot(fire.grad(u_n), fire.grad(v)) * dx  # explicit
    FF = m1 + a
    # -------------------------------------------------------
    pml1 = c_sqr_inv * (sigma_z + sigma_x + sigma_y) * \
        fire.dot((u_n - u_nm1) / fire.Constant(dt), v) * dx
    pml2 = c_sqr_inv * (sigma_z * sigma_x + sigma_x * sigma_y
                        + sigma_z * sigma_y) * fire.dot(u, v) * dx
    pml3 = -c_sqr_inv * fire.dot(fire.div(pp_n), v) * dx
    pml4 = c_sqr_inv * (sigma_z * sigma_x * sigma_y) * dot(psi_n, v) * dx
    FF += pml1 + pml2 + pml3 + pml4
    # -------------------------------------------------------
    mm1 = c_sqr_inv * fire.dot((pp - pp_n) / fire.Constant(dt), qq) * dx
    mm2 = c_sqr_inv * fire.inner(fire.dot(Gamma_1, pp_n), qq) * dx
    dd1 = fire.inner(dot(Gamma_2, fire.grad(u_n)), qq) * dx
    dd2 = -fire.inner(dot(Gamma_3, fire.grad(psi_n)), qq) * dx
    FF += mm1 + mm2 + dd1 + dd2
    # -------------------------------------------------------
    mmm1 = fire.dot((psi - psi_n) / fire.Constant(dt), phi) * dx
    uuu1 = -fire.dot(u_n * phi) * dx
    FF += mmm1 + uuu1
    # -------------------------------------------------------

    lhs_ = fire.lhs(FF)
    rhs_ = fire.rhs(FF)

    # Apply Dirichlet BCs to the PML boundaries
    fix_bnd = Dirichlet_bc_pml(Wave_object, W)

    A = fire.assemble(lhs_, mat_type="matfree", bcs=fix_bnd)
    solver = fire.LinearSolver(
        A, solver_parameters=Wave_object.solver_parameters)
    Wave_object.solver = solver
    Wave_object.rhs = rhs_
    Wave_object.B = B

    return
