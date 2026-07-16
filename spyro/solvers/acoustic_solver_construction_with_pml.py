"""Constructs Firedrake solver for the acosutic wave with a PML."""

from firedrake import (Cofunction, DirichletBC, div, dot, ds as fire_ds, dx as fire_dx,
                       Function, grad, inner, lhs, LinearVariationalProblem,
                       LinearVariationalSolver, rhs, split, TestFunctions, TrialFunctions)
from ..domains.space import create_function_space
from ..utils.typing import BoundaryConditionsType


# Work from Keith Roberts, Eduardo Moscatelli,
# Ruben Andres Salas and Alexandre Olender
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)
# TODO: Add citations

def forms_pml(Wave_object, W, X_n, X_nm1):
    """
    Build the variational form for the wave equation with a PML.

    Parameters
    ----------
    Wave_object : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    W : `Firedrake.MixedFunctionSpace`
        Mixed function space for the wave equation with PML
    X_n : `Firedrake.Function`
        State variable at time step n
    X_nm1 : `Firedrake.Function`
        State variable at time step n - 1

    Returns
    -------
    FF : `Firedrake.Form`
        Variational form for the wave equation with PML
    fix_bnd : `Firedrake.DirichletBC`
        Dirichlet boundary conditions applied to the PML boundaries

    Notes
    -----
    In the UFL forms, however, reference ``X_n``/``X_nm1`` directly via ``fire.split``.
    The pyadjoint tape tracks ``X_n.assign(X_np1)`` between time steps; the subfunction
    Functions are separately-tracked tape variables that are never explicitly written to
    in the tape, so on replay their tape values stay at the initial (zero) state, making
    ``J`` constant w.r.t. the control. Using ``fire.split(X_n)`` keeps the form dependent
    on ``X_n`` itself, which is updated correctly on replay.
    This is what makes the PML Taylor test converge.
    """

    # Simulation parameters for PML formulation
    dt = Wave_object.dt
    c = Wave_object.c
    c_sqr_inv = 1. / (c * c)
    q_rule = Wave_object.quadrature_rule
    dx = fire_dx(**q_rule) if q_rule else fire_dx
    Wave_object.layer_ops.pml_layer(Wave_object)

    # Trial and test functions, and state variables
    if Wave_object.dimension == 2:
        u, pp = TrialFunctions(W)
        v, qq = TestFunctions(W)
        u_n, pp_n = split(X_n)
        u_nm1, _ = split(X_nm1)

    elif Wave_object.dimension == 3:
        u, psi, pp = TrialFunctions(W)
        v, phi, qq = TestFunctions(W)
        u_n, psi_n, pp_n = split(X_n)
        u_nm1, psi_nm1, _ = split(X_nm1)

    # Acoustic form
    m1 = c_sqr_inv * ((u - 2. * u_n + u_nm1) / dt**2) * v
    a = dot(grad(u_n), grad(v))
    FF = (m1 + a) * dx

    # Common PML forms (Centered difference for first time derivative)
    pml3 = -div(pp_n) * v
    # -------------------------------------------------------
    mm1 = dot((pp - pp_n) / dt, qq)

    # Surfaces to apply boundary conditions (NRBCs or Traditional BCs)
    bc_surf = tuple([non_free_surf for non_free_surf, status in
                     Wave_object.mesh_parameters.boundary_ids_map.items() if status])
    abc_type = Wave_object.layer_ops.bc_boundary_pml

    if not Wave_object.abc_get_ref_model:

        # Damping profiles and matrices
        sigma_x, sigma_z = Wave_object.layer_ops.sigma_x, Wave_object.layer_ops.sigma_z
        if Wave_object.dimension == 2:
            Gamma_1, Gamma_2 = Wave_object.layer_ops.damping_pml_2d()

        elif Wave_object.dimension == 3:
            sigma_y = Wave_object.layer_ops.sigma_y
            Gamma_1, Gamma_2, Gamma_3 = Wave_object.layer_ops.damping_pml_3d()

        # PML forms (Centered difference for first time derivative)
        mm2 = inner(dot(Gamma_1, pp_n), qq)
        dd1 = inner(dot(Gamma_2, grad(u_n)), qq)
        # -------------------------------------------------------
        if Wave_object.dimension == 2:
            pml1 = (sigma_z + sigma_x) * ((u_n - u_nm1) / dt) * v
            pml2 = sigma_z * sigma_x * u_n * v

        elif Wave_object.dimension == 3:
            pml1 = (sigma_z + sigma_x + sigma_y) * ((u_n - u_nm1) / dt) * v
            pml2 = (sigma_z * sigma_x + sigma_x * sigma_y + sigma_z * sigma_y) * u_n * v
            pml4 = (sigma_z * sigma_x * sigma_y) * psi_n * v
            FF += c_sqr_inv * pml4 * dx
            # -------------------------------------------------------
            dd2 = -inner(dot(Gamma_3, grad(psi_n)), qq)
            FF += c * c * dd2 * dx
            # -------------------------------------------------------
            mmm1 = ((psi - psi_n) / dt) * phi
            uuu1 = -u_n * phi
            FF += (mmm1 + uuu1) * dx

        # Adding common PML forms to the variational form
        FF += c_sqr_inv * (pml1 + pml2 + pml3) * dx
        # -------------------------------------------------------
        FF += (mm1 + mm2 + c * c * dd1) * dx

        # exterior_markers = set(Wave_object.mesh.exterior_facets.unique_markers)
        # print("Available boundary markers:", exterior_markers)

        # Apply NRBCs (Higdon or Sommerfeld) at PML boundaries
        if abc_type in [BoundaryConditionsType.HIGDON, BoundaryConditionsType.SOMMERFELD]:
            ds = fire_ds(bc_surf, **q_rule) if q_rule else fire_ds(bc_surf)
            f_abc = c * ((u_n - u_nm1) / dt) * v
            le = Wave_object.layer_ops.cosHig * f_abc * ds
            FF += le

    # Dirichlet BCs for PML model or Neumann BCs for the reference model
    get_ref_model_negat = not Wave_object.abc_get_ref_model
    fix_bnd = DirichletBC(W.sub(0), 0., bc_surf) \
        if abc_type == BoundaryConditionsType.DIRICHLET and get_ref_model_negat else None

    return FF, fix_bnd


def construct_solver_or_matrix_with_pml(Wave_object):
    """Build solver operators for wave propagator with a PML.

    Doesn't create mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    Wave_object : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.

    Returns
    -------
    None

    Notes
    -----
    Keep Function-typed subfunction views for non-form usage (in-place ``.assign``,
    ``.dat`` access for receiver output, etc.). They share storage with ``X_n``/``X_nm1``
    so writes to the mixed Functions are visible through these views.
    """

    # Build mixed function space
    V = Wave_object.function_space
    Z = create_function_space(Wave_object.mesh, V.ufl_element(),
                              dim=Wave_object.dimension)
    Wave_object.vector_function_space = Z
    if Wave_object.dimension == 2:
        W = V * Z
    elif Wave_object.dimension == 3:
        W = V * V * Z
    Wave_object.mixed_function_space = W

    # State variables
    X_np1 = Function(W)
    X_n = Function(W)
    X_nm1 = Function(W)

    if Wave_object.dimension == 2:
        u_n_func, pp_n_func = X_n.subfunctions
        u_nm1_func, _ = X_nm1.subfunctions

    elif Wave_object.dimension == 3:
        u_n_func, psi_n_func, pp_n_func = X_n.subfunctions
        u_nm1_func, psi_nm1_func, _ = X_nm1.subfunctions

    Wave_object.u_n = u_n_func
    Wave_object.X_np1 = X_np1
    Wave_object.X_n = X_n
    Wave_object.X_nm1 = X_nm1

    # Build variational forms
    FF, fix_bnd = forms_pml(Wave_object, W, X_n, X_nm1)
    Wave_object.lhs = lhs(FF)
    Wave_object.rhs = rhs(FF)
    Wave_object.source_function = Cofunction(W.dual())
    Wave_object.B = Cofunction(W.dual())

    # Build solver
    lin_var = LinearVariationalProblem(
        Wave_object.lhs, Wave_object.rhs + Wave_object.source_function,
        X_np1, bcs=fix_bnd, constant_jacobian=True)
    solver_parameters = dict(Wave_object.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    Wave_object.solver = \
        LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
