"""Constructs Firedrake solver for the acosutic wave with typical BCs, NRBCs or HABCs."""

import firedrake as fire
from firedrake import ds, dx, dot, grad
from ..utils.typing import AbsorbingBCsType, BoundaryConditionsType


def construct_solver_or_matrix_no_pml(wave):
    """Builds solver operators for wave propagator with typical BCs, NRBCs or HABCs.

    Doesn't create mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """
    V = wave.function_space
    quad_rule = wave.quadrature_rule

    # typical CG FEM in 2d/3d
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    u_nm1 = fire.Function(V, name="pressure t-dt")
    u_n = fire.Function(V, name="pressure")
    u_np1 = fire.Function(V, name="pressure t+dt")
    wave.u_nm1 = u_nm1
    wave.u_n = u_n
    wave.u_np1 = u_np1

    wave.current_time = 0.0
    dt = wave.dt

    # -------------------------------------------------------
    m1 = (
        (1 / (wave.c * wave.c))
        * ((u - 2.0 * u_n + u_nm1) / dt**2)
        * v
        * dx(**quad_rule)
    )
    a = dot(grad(u_n), grad(v)) * dx(**quad_rule)  # explicit

    le = 0.0

    # Surfaces to apply boundary conditions (NRBCs or Traditional BCs)
    bc_surf = tuple([non_free_surf for non_free_surf, status in
                     wave.mesh_parameters.boundary_ids_map.items() if status])
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
            c_inv = 1. / c
            weak_expr_nrbc = cosHig * c_inv * weak_expr_abc

            # include NRBC terms in the load term
            le_nrbc = forms_NRBCs(wave, weak_expr_nrbc, bc_surf)
            le += le_nrbc

    # Dirichlet or Neumann BCs if desired (no ABCs)
    bc_fix = (bc_bndr == BoundaryConditionsType.DIRICHLET and include_bcs)
    fix_bnd = DirichletBC(V, 0., bc_surf) if bc_fix else None

    # Build variational forms
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    # TODO: Add citation
    form = m1 + a + le
    wave.rhs = fire.rhs(form)
    wave.lhs = fire.lhs(form)
    wave.source_function = fire.Cofunction(V.dual())

    lin_var = fire.LinearVariationalProblem(
        wave.lhs,
        wave.rhs + wave.source_function,
        u_np1,
        bcs=fix_bnd,
        constant_jacobian=True,
    )
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    wave.solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters,
    )
