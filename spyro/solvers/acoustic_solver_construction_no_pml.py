import firedrake as fire
from firedrake import ds, dx, Constant, dot, grad
from numpy import where

# Modifications by Ruben Andres Salas
# see Salas et al (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# "Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations"


def construct_solver_or_matrix_no_pml(Wave_object):
    '''
    Builds solver operators for wave object without a PML. Doesn't create
    mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    Wave_object: :class: 'Wave' object
        Waveform object that contains all simulation parameters

    Returns
    -------
    None
    '''
    V = Wave_object.function_space
    quad_rule = Wave_object.quadrature_rule

    # Test and trial functions
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    # State variables
    u_nm1 = fire.Function(V, name="pressure t-dt")
    u_n = fire.Function(V, name="pressure")
    u_np1 = fire.Function(V, name="pressure t+dt")
    Wave_object.u_nm1 = u_nm1
    Wave_object.u_n = u_n
    Wave_object.u_np1 = u_np1

    Wave_object.current_time = 0.0
    dt = Wave_object.dt

    # -------------------------------------------------------
    m1 = ((1 / (Wave_object.c * Wave_object.c))
          * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
          * v * dx(scheme=quad_rule))
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)  # explicit

    le = 0.0
    q = Wave_object.source_expression
    if q is not None:
        le += - q * v * dx(scheme=quad_rule)

    B = fire.Cofunction(V.dual())

    if Wave_object.abc_active:
        weak_expr_abc = dot((u_n - u_nm1) / Constant(dt), v)
        f_abc = (1 / Wave_object.c) * weak_expr_abc
        qr_s = Wave_object.surface_quadrature_rule

        if Wave_object.abc_boundary_layer_type == "hybrid":

            # NRBC
            le += Wave_object.cosHig * f_abc * ds(scheme=qr_s)

            # Damping
            le += Wave_object.eta_mask * weak_expr_abc * \
                (1 / (Wave_object.c * Wave_object.c)) * \
                Wave_object.eta_habc * dx(scheme=quad_rule)

        else:
            # Only NRBC
            bnds = [Wave_object.absorb_top, Wave_object.absorb_bottom,
                    Wave_object.absorb_right, Wave_object.absorb_left]

            if Wave_object.dimension == 3:
                bnds.extend([Wave_object.absorb_front,
                             Wave_object.absorb_back])

            # Tuple of boundary ids for NRBC
            where_to_absorb = tuple(where(bnds)[0] + 1)  # ds starts at 1
            le += f_abc * ds(where_to_absorb, scheme=qr_s)

    # form = m1 + a - le
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    form = m1 + a + le
    lhs = fire.lhs(form)
    rhs = fire.rhs(form)
    Wave_object.lhs = lhs

    A = fire.assemble(lhs, mat_type="matfree")
    Wave_object.solver = fire.LinearSolver(
        A, solver_parameters=Wave_object.solver_parameters)
    Wave_object.rhs = rhs
    Wave_object.B = B
