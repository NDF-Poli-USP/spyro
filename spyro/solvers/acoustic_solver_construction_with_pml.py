import firedrake as fire
from numpy import where, log

# Work from Keith Roberts, Eduardo Moscatelli,
# Ruben Andres Salas and Alexandre Olender
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)


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
    cmax = Wave_obj.c_max  # maximum acoustic wave velocity
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
    bar_sigma = cmax * (0. if Wave_obj.abc_get_ref_model else
                        calc_pml_damping(pad_length, CR=R))

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

        if Wave_obj.dimension == 2:
            sigma_x, sigma_z = pml_sigma_field(Wave_obj)
            Gamma_1, Gamma_2 = damping_pml_2d(sigma_z, sigma_x)

        elif Wave_obj.dimension == 3:
            sigma_x, sigma_y, sigma_z = pml_sigma_field(Wave_obj)
            Gamma_1, Gamma_2, Gamma_3 = damping_pml_3d(sigma_z, sigma_x, sigma_y)

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

    if not Wave_obj.abc_get_ref_model:
        Wave_obj.cosHig = fire.Constant(1.)

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

    FF, fix_bnd = forms_pml(Wave_obj, W, bc_type="nrbc")

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
