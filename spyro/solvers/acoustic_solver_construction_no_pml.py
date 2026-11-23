"""Constructs Firedrake solver for the acoustic wave.

Handles typical BCs, NRBCs or HABCs.
"""

from firedrake import (
    Cofunction,
    dot,
    dx as fire_dx,
    Function,
    grad,
    lhs,
    LinearVariationalProblem,
    LinearVariationalSolver,
    rhs,
    TestFunction,
    TrialFunction,
)
from ..utils.typing import AbsorbingBCsType, BoundaryConditionsType


def build_acoustic_form(wave, u_trial, v_test, u_n, u_nm1, quad_rule):
    """Build the weak form of the acoustic wave equation for one time step.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    u_trial : `firedrake.TrialFunction`
        Trial function for the pressure field at the next time step.
    v_test : `firedrake.TestFunction`
        Test function for the pressure field.
    u_n : `firedrake.Function`
        Pressure field at the current time step.
    u_nm1 : `firedrake.Function`
        Pressure field at the previous time step.
    quad_rule : dict
        Quadrature rule for volume integration.

    Returns
    -------
    form : UFL form
        The combined weak form (mass + stiffness + source + absorbing terms).

    """
    c = wave.c
    dt = wave.dt

    m1 = (
        (1 / (c * c))
        * ((u_trial - 2.0 * u_n + u_nm1) / dt**2)
        * v_test
        * dx(**quad_rule)
    )
    a = dot(grad(u_n), grad(v_test)) * dx(**quad_rule)

    le = 0.0
    q = wave.source_expression
    if q is not None:
        le += -q * v_test * dx(**quad_rule)
    if wave.point_source_in_form:
        # See Sources.source_term. The time function is zero until a forward
        # solve sets it, and the implemented adjoint switches it off again.
        le += -wave.sources.source_term(v_test, dx(**quad_rule))

    if wave.abc_active and not wave.abc_get_ref_model:
        weak_expr_abc = dot((u_n - u_nm1) / dt, v_test)

        f_abc = (1 / c) * weak_expr_abc
        qr_s = wave.surface_quadrature_rule

        if wave.abc_type == AbsorbingBCsType.HYBRID:
            # NRBC
            le += wave.cosHig * f_abc * ds(**qr_s)

            # Damping
            le += (
                wave.eta_mask
                * weak_expr_abc
                * (1 / (c * c))
                * wave.eta_habc
                * dx(**quad_rule)
            )

        else:
            if wave.absorb_top:
                le += f_abc * ds(1, **qr_s)
            if wave.absorb_bottom:
                le += f_abc * ds(2, **qr_s)
            if wave.absorb_right:
                le += f_abc * ds(3, **qr_s)
            if wave.absorb_left:
                le += f_abc * ds(4, **qr_s)
            if wave.dimension == 3:
                if wave.absorb_front:
                    le += f_abc * ds(5, **qr_s)
                if wave.absorb_back:
                    le += f_abc * ds(6, **qr_s)

    # form = m1 + a - le
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    # TODO: Add citation
    return m1 + a + le


def construct_solver_or_matrix_no_pml(wave):
    """Build the Firedrake solver for the acoustic wave, without PML.

    Handles typical BCs, NRBCs or HABCs.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """

    # Simulation parameters for acoustic propagation
    V = wave.function_space
    dt = wave.dt
    c = wave.c
    c_sqr_inv = 1.0 / (c * c)
    quad_rule = wave.quadrature_rule
    dx = fire_dx(**quad_rule) if quad_rule else fire_dx

    # Trial and test functions, and state variables
    u = TrialFunction(V)
    v = TestFunction(V)
    u_nm1 = Function(V, name="pressure t-dt")
    u_n = Function(V, name="pressure")
    u_np1 = Function(V, name="pressure t+dt")
    wave.u_nm1 = u_nm1
    wave.u_n = u_n
    wave.u_np1 = u_np1

    # Acoustic form
    m1 = c_sqr_inv * ((u - 2.0 * u_n + u_nm1) / dt**2) * v * dx
    a = dot(grad(u_n), grad(v)) * dx

    # Load term for sources and ABCs
    le = 0.0

    # Load term for sources
    q = wave.source_expression
    if q is not None:
        le += -q * v * dx

    # Surfaces to apply boundary conditions (NRBCs or Traditional BCs)
    bc_surf = tuple(
        [
            non_free_surf
            for non_free_surf, status in wave.mesh_parameters.boundary_ids_map.items()
            if status
        ]
    )
    bc_bndr = BoundaryConditionsType.NEUMANN  # TODO: include Dirichlet BCs
    # bc_bndr = wave.layer_ops.bc_boundary_habc  # TODO: Include this in the wave object

    # Load term for ABCs
    include_bcs = not wave.abc_get_ref_model
    if wave.abc_active and include_bcs:
        # General weak form for ABCs
        weak_expr_abc = ((u_n - u_nm1) / dt) * v

        # Include absorbing layer damping term for HABCs
        if wave.abc_type == AbsorbingBCsType.HYBRID:
            weak_expr_habc = c_sqr_inv * weak_expr_abc
            le += wave.eta_mask * wave.eta_habc * weak_expr_habc * dx

        # Apply NRBCs (Higdon or Sommerfeld) at domain boundaries (w/o absorbing layer)
        if wave.abc_type == AbsorbingBCsType.NRBC or AbsorbingBCsType.HYBRID:
            # Cosine of incidence angle for Higdon BCs or 1 for Sommerfeld BCs
            if wave.abc_type == AbsorbingBCsType.NRBC:  # NRBCs
                cosHig = wave.nrbc_ops.cosHig
                forms_NRBCs = wave.nrbc_ops.forms_acoustic_NRBCs

            if wave.abc_type == AbsorbingBCsType.HYBRID:  # HABCs
                cosHig = wave.layer_ops.cosHig
                forms_NRBCs = wave.layer_ops.forms_acoustic_NRBCs

            # General weak form for NRBCs
            c_inv = 1.0 / c
            weak_expr_nrbc = cosHig * c_inv * weak_expr_abc

            # include NRBC terms in the load term
            le_nrbc = forms_NRBCs(wave, weak_expr_nrbc, bc_surf)
            le += le_nrbc

    # Dirichlet or Neumann BCs if desired (no ABCs)
    bc_fix = bc_bndr == BoundaryConditionsType.DIRICHLET and include_bcs
    fix_bnd = DirichletBC(V, 0.0, bc_surf) if bc_fix else None

    # Build variational forms
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    # TODO: Add citation
    form = m1 + a + le
    wave.rhs = rhs(form)
    wave.lhs = lhs(form)
    wave.source_function = Cofunction(V.dual())

    # Build solver
    lin_var = LinearVariationalProblem(
        wave.lhs,
        wave.rhs + wave.source_function,
        u_np1,
        bcs=fix_bnd,
        constant_jacobian=True,
    )
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    wave.solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
