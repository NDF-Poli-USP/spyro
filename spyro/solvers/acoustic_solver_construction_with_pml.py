import firedrake as fire
from firedrake import dx, ds, Constant, dot, grad, inner

from ..pml import damping


def construct_solver_or_matrix_with_pml(Wave_object):
    '''
    Builds solver operators for wave propagator with a PML. Doesn't create
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
    V = self.function_space
    Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    Wave_object.vector_function_space = Z
    if Wave_object.dimension == 2:
        return construct_solver_or_matrix_with_pml_2d(Wave_object)
    elif Wave_object.dimension == 3:
        return construct_solver_or_matrix_with_pml_3d(Wave_object)


def matrices_2D(sigma_z, sigma_x):
    '''
    Damping matrices for a two-dimensional problem.

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
    Gamma_1 = as_tensor([[sigma_z, 0.], [0., sigma_x]])
    Gamma_2 = as_tensor([[sigma_z - sigma_x, 0.], [0., sigma_x - sigma_z]])

    return Gamma_1, Gamma_2


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
    c = Wave_object.c
    c_sqr_inv = 1. / (c * c)

    V = Wave_object.function_space
    Z = Wave_object.vector_function_space
    W = V * Z
    Wave_object.mixed_function_space = W
    dx = dx(scheme=Wave_object.quadrature_rule)
    ds = ds(scheme=Wave_object.surface_quadrature_rule)

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
    Gamma_1, Gamma_2 = matrices_2D(sigma_z, sigma_x)

    B = fire.Cofunction(W.dual())
    # -------------------------------------------------------
    m1 = (c_sqr_inv * (u - 2. * u_n + u_nm1) / Constant(dt**2)) * v * dx
    a = dot(grad(u_n), grad(v)) * dx  # explicit
    FF = m1 + a
    # -------------------------------------------------------
    pml1 = c_sqr_inv * (sigma_z + sigma_z) * \
        dot((u_n - u_nm1) / Constant(dt), v) * dx
    pml2 = c_sqr_inv * sigma_z * sigma_x * dot(u_n, v) * dx
    pml3 = -c_sqr_inv * inner(pp_n, grad(v)) * dx
    FF += pml1 + pml2 + pml3
    # -------------------------------------------------------
    mm1 = c_sqr_inv * dot((pp - pp_n) / Constant(dt), qq) * dx
    mm2 = c_sqr_inv * inner(dot(Gamma_1, pp_n), qq) * dx
    dd = inner(grad(u_n), dot(Gamma_2, qq)) * dx
    FF += mm1 + mm2 + dd

    lhs_ = fire.lhs(FF)
    rhs_ = fire.rhs(FF)

    A = fire.assemble(lhs_, mat_type="matfree")
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
    c = Wave_object.c

    V = Wave_object.function_space
    Z = Wave_object.vector_function_space
    W = V * V * Z
    Wave_object.mixed_function_space = W
    dxlump = dx(scheme=Wave_object.quadrature_rule)
    dslump = ds(scheme=Wave_object.surface_quadrature_rule)

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

    sigma_x, sigma_y, sigma_z = damping.functions(Wave_object)
    Gamma_1, Gamma_2, Gamma_3 = damping.matrices_3D(sigma_x, sigma_y, sigma_z)

    pml1 = (
        (sigma_x + sigma_y + sigma_z)
        * ((u - u_nm1) / Constant(2.0 * dt))
        * v
        * dxlump
    )

    pml2 = (
        (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
        * u_n
        * v
        * dxlump
    )

    pml3 = (sigma_x * sigma_y * sigma_z) * psi_n * v * dxlump
    pml4 = inner(pp_n, grad(v)) * dxlump

    # typical CG FEM in 2d/3d

    # -------------------------------------------------------
    m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dxlump
    a = c * c * dot(grad(u_n), grad(v)) * dxlump  # explicit

    nf = c * ((u_n - u_nm1) / dt) * v * dslump

    FF = m1 + a + nf

    B = fire.Cofunction(W.dual())

    FF += pml1 + pml2 + pml3 + pml4
    # -------------------------------------------------------
    mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dxlump
    mm2 = inner(dot(Gamma_1, pp_n), qq) * dxlump
    dd1 = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dxlump
    dd2 = -c * c * inner(grad(psi_n), dot(Gamma_3, qq)) * dxlump
    FF += mm1 + mm2 + dd1 + dd2

    mmm1 = (dot((psi - psi_n), phi) / Constant(dt)) * dxlump
    uuu1 = (-u_n * phi) * dxlump
    FF += mmm1 + uuu1

    lhs_ = fire.lhs(FF)
    rhs_ = fire.rhs(FF)

    A = fire.assemble(lhs_, mat_type="matfree")
    solver = fire.LinearSolver(
        A, solver_parameters=Wave_object.solver_parameters
    )
    Wave_object.solver = solver
    Wave_object.rhs = rhs_
    Wave_object.B = B

    return
