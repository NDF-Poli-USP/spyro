"""Constructs Firedrake solver for the acosutic wave with typical BCs, NRBCs or HABCs."""

from firedrake import (Cofunction, dot, ds as fire_ds, ds_b as quad_dsbottom,
                       ds_t as quad_dstop, ds_v as quad_ds, dx as fire_dx,
                       Function, grad, lhs, LinearVariationalProblem,
                       LinearVariationalSolver, rhs, TestFunction, TrialFunction)
from ..utils.typing import AbsorbingBCsType, BoundaryConditionsType


def construct_solver_or_matrix_no_pml(wave):
    """Builds solver operators for wave propagator with typical BCs, NRBCs or HABCs.

    Doesn't create mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """

    # Simulation parameters for acoustic propagation
    V = wave.function_space
    dt = wave.dt
    c = wave.c
    c_sqr_inv = 1. / (c * c)
    quad_rule = wave.quadrature_rule
    quad_surf = wave.surface_quadrature_rule
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
    m1 = c_sqr_inv * ((u - 2. * u_n + u_nm1) / dt**2) * v * dx
    a = dot(grad(u_n), grad(v)) * dx

    # Load term for sources and ABCs
    le = 0.

    # Load term for sources
    q = wave.source_expression
    if q is not None:
        le += - q * v * dx

    # Surfaces to apply boundary conditions (NRBCs or Traditional BCs)
    bc_surf = tuple([non_free_surf for non_free_surf, status in
                     wave.mesh_parameters.boundary_ids_map.items() if status])
    bc_bndr = BoundaryConditionsType.NEUMANN  # TODO: include Dirichlet BCs
    # bc_bndr = wave.layer_ops.bc_boundary_habc  # TODO: Include this in the wave object

    # Load term for ABCs
    fix_bnd = None
    include_bcs = not wave.abc_get_ref_model
    if wave.abc_active and include_bcs:

        # General weak form for ABCs
        weak_expr_abc = ((u_n - u_nm1) / dt) * v

        # Include absorbing layer damping term for HABCs
        if wave.abc_type == AbsorbingBCsType.HYBRID:
            le += wave.eta_mask * c_sqr_inv * wave.eta_habc * weak_expr_abc * dx

        # Apply NRBCs (Higdon or Sommerfeld) at domain boundaries (w/o absorbing layer)
        if wave.abc_type == AbsorbingBCsType.NRBC or AbsorbingBCsType.HYBRID:

            # Cosine of incidence angle for Higdon BCs or 1 for Sommerfeld BCs
            cosHig = wave.nrbc_ops.cosHig if wave.abc_type == AbsorbingBCsType.NRBC \
                else wave.layer_ops.cosHig

            # General weak form for NRBCs
            weak_expr_nrbc = cosHig * (1 / c) * weak_expr_abc

            # exterior_markers = set(wave.mesh.exterior_facets.unique_markers)
            # print("Available boundary markers:", exterior_markers)

            if wave.mesh_parameters.quadrilateral:

                # Integer boundary IDs for quadrilaterals/hexahedra meshes
                int_ids = tuple(filter(lambda k: isinstance(k, int),
                                       wave.mesh_parameters.boundary_ids_map.keys()))

                # Top boundary for quadrilaterals/hexahedra meshes
                if wave.mesh_parameters.boundary_ids_map.get("top", False):
                    le += weak_expr_nrbc * quad_dstop  # (Do not support quadrature)

                # Bottom boundary for quadrilaterals/hexahedra meshes
                if wave.mesh_parameters.boundary_ids_map.get("bottom", False):
                    le += weak_expr_nrbc * quad_dsbottom  # (Do not support quadrature)

                ds = quad_ds(int_ids, **quad_surf) if quad_surf else quad_ds(int_ids)

            else:

                # Integration measure for triangles/tetrahedra
                ds = fire_ds(bc_surf, **quad_surf) if quad_surf else fire_ds(bc_surf)

            # NRBCs: Higdon or Sommerfeld
            le += weak_expr_nrbc * ds

    else:

        # Dirichlet or Neumann BCs if desired (no ABCs)
        fix_bnd = DirichletBC(V, 0., bc_surf) if \
            bc_bndr == BoundaryConditionsType.DIRICHLET and include_bcs else None

    # Build variational forms
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    # TODO: Add citation
    form = m1 + a + le
    wave.rhs = rhs(form)
    wave.lhs = lhs(form)
    wave.source_function = Cofunction(V.dual())

    # Build solver
    lin_var = LinearVariationalProblem(wave.lhs, wave.rhs + wave.source_function,
                                       u_np1, bcs=fix_bnd, constant_jacobian=True)
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    wave.solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
