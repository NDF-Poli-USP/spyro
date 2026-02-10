import firedrake as fire
from numpy import where, log

# Work from Keith Roberts, Eduardo Moscatelli,
# Ruben Andres Salas and Alexandre Olender
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)


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
    solver: Firedrake 'LinearSolver'
        Linear solver for the wave equation with PML
    '''
    V = Wave_obj.function_space
    Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    Wave_obj.vector_function_space = Z
    if Wave_obj.dimension == 2:
        return construct_solver_or_matrix_with_pml_2d(Wave_obj)
    elif Wave_obj.dimension == 3:
        return construct_solver_or_matrix_with_pml_3d(Wave_obj)


def calc_pml_damping(pad_len, dgr_prof=2, CR=0.001):
    '''
    Calculate the maximum damping coefficient for the PML layer.

    Parameters
    ----------
    pad_len : `float`
        Length of the PML layer
    dgr_prof : `int`, optional
        Degree of the damping profile within the PML layer.
    CR : `float`, optional
        Desired reflection coefficient at outer boundary of PML layer.
        Default is 0.001

    Returns
    -------
    sigma_max : `float`
        Maximum damping coefficient within the PML layer
    '''

    sigma_max = (dgr_prof + 1.) / (2. * pad_len) * log(1 / CR)

    return sigma_max


def pml_sigma_field(Wave_obj):
    '''
    Generate a damping profile for the PML.

    Parameters
    ----------
    coords : 'ufl.geometry.SpatialCoordinate'
        Domain Coordinates including the absorbing layer
    V : `firedrake function space`
        Function space for the mask field
    pad_len : `float`
        Length of the PML layer
    sigma_max : `float`
        Maximum damping coefficient within the PML layer

    Returns
    -------
    sigma_x : obj
        Firedrake function with the damping function in the x direction
    sigma_z : obj
        Firedrake function with the damping function in the z direction
    sigma_y : obj
        Firedrake function with the damping function in the y direction
    '''

    ps = Wave_obj.abc_exponent
    cmax = Wave_obj.abc_cmax  # maximum acoustic wave velocity
    R = Wave_obj.abc_R  # theoretical reclection coefficient
    pad_length = Wave_obj.abc_pad_length  # length of the padding
    V = Wave_obj.function_space
    dimension = Wave_obj.dimension
    z = Wave_obj.mesh_z
    x = Wave_obj.mesh_x
    x1 = 0.0
    x2 = Wave_obj.mesh_parameters.length_x
    z2 = -Wave_obj.mesh_parameters.length_z

    # Compute the maximum damping coefficient
    bar_sigma = cmax * calc_pml_damping(pad_length, CR=R)

    aux1 = fire.Function(V)
    aux2 = fire.Function(V)

    # Sigma X
    sigma_max_x = bar_sigma  # Max damping
    aux1.interpolate(
        fire.conditional(
            fire.And((x >= x1 - pad_length), x < x1),
            ((abs(x - x1) ** (ps)) / (pad_length ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    aux2.interpolate(
        fire.conditional(
            fire.And(x > x2, (x <= x2 + pad_length)),
            ((abs(x - x2) ** (ps)) / (pad_length ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    sigma_x = fire.Function(V, name="sigma_x").interpolate(aux1 + aux2)

    # Sigma Z
    tol_z = 1.000001
    sigma_max_z = bar_sigma  # Max damping
    aux1.interpolate(
        fire.conditional(
            fire.And(z < z2, (z >= z2 - tol_z * pad_length)),
            ((abs(z - z2) ** (ps)) / (pad_length ** (ps))) * sigma_max_z,
            0.0,
        )
    )

    sigma_z = fire.Function(V, name="sigma_z").interpolate(aux1)

    if dimension == 2:
        return (sigma_x, sigma_z)

    elif dimension == 3:
        # Sigma Y
        sigma_max_y = bar_sigma  # Max damping
        y = Wave_obj.mesh_y
        y1 = 0.0
        y2 = Wave_obj.length_y
        aux1.interpolate(
            fire.conditional(
                fire.And((y >= y1 - pad_length), y < y1),
                ((abs(y - y1) ** (ps)) / (pad_length ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        aux2.interpolate(
            fire.conditional(
                fire.And(y > y2, (y <= y2 + pad_length)),
                ((abs(y - y2) ** (ps)) / (pad_length ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        sigma_y = fire.Function(V, name="sigma_y").interpolate(aux1 + aux2)
        # sgm_y = VTKFile("pmlField/sigma_y.pvd")
        # sgm_y.write(sigma_y)

        return (sigma_x, sigma_y, sigma_z)


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


def Dirichlet_bc_pml(Wave_obj, W):
    '''
    Apply Dirichlet boundary conditions to the PML boundaries.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    W : Firedrake 'MixedFunctionSpace'
        Mixed function space for the wave equation with PML

    Returns
    -------
    fix_bnd : Firedrake 'DirichletBC'
        Dirichlet boundary conditions applied to the PML boundaries
    '''

    bnds = [Wave_obj.absorb_top, Wave_obj.absorb_bottom,
            Wave_obj.absorb_right, Wave_obj.absorb_left]

    if Wave_obj.dimension == 3:
        bnds.extend([Wave_obj.absorb_front,
                     Wave_obj.absorb_back])

    # Tuple of boundary ids for Dirichlet BC
    where_to_absorb = tuple(where(bnds)[0] + 1)  # ds starts at 1

    # Boundary nodes indices
    fix_bnd = fire.DirichletBC(W.sub(0), fire.Constant(0.), where_to_absorb)

    return fix_bnd


def construct_solver_or_matrix_with_pml_2d(Wave_obj):
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
    dt = Wave_obj.dt
    c_sqr_inv = 1. / (Wave_obj.c * Wave_obj.c)

    V = Wave_obj.function_space
    Z = Wave_obj.vector_function_space
    W = V * Z
    Wave_obj.mixed_function_space = W
    dx = fire.dx(**Wave_obj.quadrature_rule)
    # ds = fire.ds(**Wave_obj.surface_quadrature_rule)

    u, pp = fire.TrialFunctions(W)
    v, qq = fire.TestFunctions(W)

    X = fire.Function(W)
    X_n = fire.Function(W)
    X_nm1 = fire.Function(W)
    X_np1 = fire.Function(W)  # ToDo: Not used?

    u_n, pp_n = X_n.subfunctions
    u_nm1, _ = X_nm1.subfunctions

    Wave_obj.u_n = u_n
    Wave_obj.X = X
    Wave_obj.X_n = X_n
    Wave_obj.X_nm1 = X_nm1
    Wave_obj.X_np1 = X_np1  # ToDo: Not used?

    # sigma_x, sigma_z = Wave_obj.sigma_x, Wave_obj.sigma_z
    sigma_x, sigma_z = pml_sigma_field(Wave_obj)
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
    f_abc = (1. / Wave_obj.c) * fire.dot((u_n - u_nm1) / fire.Constant(dt), v)
    qr_s = Wave_obj.surface_quadrature_rule

    # Tuple of boundary ids for NRBC
    bnds = [Wave_obj.absorb_top, Wave_obj.absorb_bottom,
            Wave_obj.absorb_right, Wave_obj.absorb_left]
    where_to_absorb = tuple(where(bnds)[0] + 1)  # ds starts at 1
    le = f_abc * ds(where_to_absorb, **qr_s)  # NRBC
    FF += le
    # -------------------------------------------------------

    lhs_ = fire.lhs(FF)
    rhs_ = fire.rhs(FF)

    # Apply Dirichlet BCs to the PML boundaries
    fix_bnd = Dirichlet_bc_pml(Wave_obj, W)

    A = fire.assemble(lhs_, mat_type="matfree", bcs=fix_bnd)
    solver = fire.LinearSolver(
        A, solver_parameters=Wave_obj.solver_parameters)
    Wave_obj.solver = solver
    Wave_obj.rhs = rhs_
    Wave_obj.B = B


def construct_solver_or_matrix_with_pml_3d(Wave_obj):
    """
    Builds solver operators for 3D wave propagator with a PML. Doesn't create
    mass matrices if matrix_free option is on, which it is by default.
    """
    dt = Wave_obj.dt
    c_sqr_inv = 1. / (Wave_obj.c * Wave_obj.c)

    V = Wave_obj.function_space
    Z = Wave_obj.vector_function_space
    W = V * V * Z
    Wave_obj.mixed_function_space = W
    dx = fire.dx(**Wave_obj.quadrature_rule)
    # ds = fire.ds(**Wave_obj.surface_quadrature_rule)

    u, psi, pp = fire.TrialFunctions(W)
    v, phi, qq = fire.TestFunctions(W)

    X = fire.Function(W)
    X_n = fire.Function(W)
    X_nm1 = fire.Function(W)
    X_np1 = fire.Function(W)  # ToDo: Not used?

    u_n, psi_n, pp_n = X_n.subfunctions
    u_nm1, psi_nm1, _ = X_nm1.subfunctions

    Wave_obj.u_n = u_n
    Wave_obj.X = X
    Wave_obj.X_n = X_n
    Wave_obj.X_nm1 = X_nm1
    Wave_obj.X_np1 = X_np1  # ToDo: Not used?

    # sigma_x, sigma_z = Wave_obj.sigma_x, Wave_obj.sigma_z
    # sigma_y = Wave_obj.sigma_y
    sigma_x, sigma_y, sigma_z = pml_sigma_field(Wave_obj)
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
    pml4 = c_sqr_inv * (sigma_z * sigma_x * sigma_y) * fire.dot(psi_n, v) * dx
    FF += pml1 + pml2 + pml3 + pml4
    # -------------------------------------------------------
    mm1 = c_sqr_inv * fire.dot((pp - pp_n) / fire.Constant(dt), qq) * dx
    mm2 = c_sqr_inv * fire.inner(fire.dot(Gamma_1, pp_n), qq) * dx
    dd1 = fire.inner(fire.dot(Gamma_2, fire.grad(u_n)), qq) * dx
    dd2 = -fire.inner(fire.dot(Gamma_3, fire.grad(psi_n)), qq) * dx
    FF += mm1 + mm2 + dd1 + dd2
    # -------------------------------------------------------
    mmm1 = fire.dot((psi - psi_n) / fire.Constant(dt), phi) * dx
    uuu1 = -fire.dot(u_n * phi) * dx
    FF += mmm1 + uuu1
    # -------------------------------------------------------
    f_abc = (1. / Wave_obj.c) * fire.dot((u_n - u_nm1) / fire.Constant(dt), v)
    qr_s = Wave_obj.surface_quadrature_rule

    # Tuple of boundary ids for NRBC
    bnds = [Wave_obj.absorb_top, Wave_obj.absorb_bottom,
            Wave_obj.absorb_right, Wave_obj.absorb_left,
            Wave_obj.absorb_front, Wave_obj.absorb_back]
    where_to_absorb = tuple(where(bnds)[0] + 1)  # ds starts at 1
    le = f_abc * ds(where_to_absorb, **qr_s)  # NRBC
    FF += le
    # -------------------------------------------------------

    lhs_ = fire.lhs(FF)
    rhs_ = fire.rhs(FF)

    # Apply Dirichlet BCs to the PML boundaries
    fix_bnd = Dirichlet_bc_pml(Wave_obj, W)

    A = fire.assemble(lhs_, mat_type="matfree", bcs=fix_bnd)
    solver = fire.LinearSolver(
        A, solver_parameters=Wave_obj.solver_parameters)
    Wave_obj.solver = solver
    Wave_obj.rhs = rhs_
    Wave_obj.B = B

    return
