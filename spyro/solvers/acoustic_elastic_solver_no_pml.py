import firedrake as fire

from firedrake import *
from spyro.solvers.acoustic_solver_construction_no_pml import build_acoustic_form
from spyro.solvers.elastic_wave.forms import build_elastic_form
from spyro.solvers.elastic_wave.local_abc import local_abc_form

import numpy as np
import ufl


def _drop_missing_facets(form):
    kept = []
    for itg in form.integrals():
        sid = itg.subdomain_id()
        if itg.integral_type() == "exterior_facet" and not isinstance(sid, str):
            markers = set(itg.ufl_domain().exterior_facets.unique_markers.tolist())
            if not set(np.atleast_1d(sid).tolist()) <= markers:
                continue
        kept.append(itg)
    return ufl.Form(kept)


def build_anisotropic_elastic_form(wave, u_trial, v_test, u_n, u_nm1, quad_rule, abc_kwargs=None):
    """Build the weak form of the anisotropic elastic wave equation using
    the precomputed stiffness tensor Elastic_C in Voigt notation.

    This is the anisotropic counterpart of build_elastic_form and has the
    same signature, so it can be used as a drop-in replacement inside
    construct_acoustic_elastic when wave.Elastic_C is set.

    Parameters
    ----------
    wave : AcousticElasticWave
        Wave object exposing rho, Elastic_C, dt, body_forces.
    u_trial, v_test, u_n, u_nm1 : firedrake Functions / TestFunction
        Trial/test functions and time-level fields.
    quad_rule : dict
        Quadrature rule for volume integration.
    abc_kwargs : dict, optional
        Extra keyword arguments forwarded to local_abc_form.

    Returns
    -------
    UFL form
    """
    dt = Constant(wave.dt)
    rho = wave.rho
    Elastic_C = wave.Elastic_C
    dim = v_test.function_space().mesh().topological_dimension

    def strain_vector(w):
        g = grad(w)
        if dim == 2:
            return as_vector([g[0, 0], g[1, 1], g[0, 1] + g[1, 0]])
        else:
            return as_vector([
                g[0, 0], g[1, 1], g[2, 2],
                g[1, 2] + g[2, 1],
                g[0, 2] + g[2, 0],
                g[0, 1] + g[1, 0],
            ])

    # Inertial term
    F_m = (rho / (dt**2)) * dot(u_trial - 2 * u_n + u_nm1, v_test) * dx(**quad_rule)

    # Stiffness term via Voigt notation
    e_n = strain_vector(u_n)
    e_v = strain_vector(v_test)
    F_k = dot(e_v, dot(Elastic_C, e_n)) * dx(**quad_rule)

    # Source term
    F_s = 0
    if getattr(wave, "body_forces", None) is not None:
        F_s += dot(wave.body_forces, v_test) * dx(**quad_rule)

    # Absorbing boundary
    F_t = local_abc_form(wave, v=v_test, u_n=u_n, u_nm1=u_nm1, **(abc_kwargs or {}))

    return F_m + F_k - F_s - F_t


def construct_acoustic_elastic(Wave_obj):
    p_nm1, u_nm1 = fire.split(Wave_obj.X_nm1)
    p_n, u_n = fire.split(Wave_obj.X_n)

    dt = Wave_obj.dt
    n_s = Wave_obj.n_s
    ds_int = Wave_obj.ds_int
    iface = Wave_obj.interface_id

    # -------------------------------------------------------------------------
    # Solid sub-problem
    # -------------------------------------------------------------------------
    u_trial = TrialFunction(Wave_obj.vector_function_space)
    v_s = TestFunction(Wave_obj.vector_function_space)

    abc_kwargs_solid = {
        "c_p": fire.sqrt((Wave_obj.lmbda + 2.0 * Wave_obj.mu) / Wave_obj.rho),
        "c_s": fire.sqrt(Wave_obj.mu / Wave_obj.rho),
        "qr_s": Wave_obj.surface_quadrature_rule_solid,
        "boundary_ids": {"bottom": 2, "top": None, "left": 3, "right": 4},
    }

    # Use anisotropic form when Elastic_C is available, isotropic otherwise
    if getattr(Wave_obj, "Elastic_C", None) is not None:
        F_elastic = build_anisotropic_elastic_form(
            Wave_obj, u_trial, v_s, u_n, u_nm1,
            Wave_obj.quadrature_rule_solid,
            abc_kwargs=abc_kwargs_solid,
        )
    else:
        F_elastic = build_elastic_form(
            Wave_obj, u_trial, v_s, u_n, u_nm1,
            Wave_obj.quadrature_rule_solid,
            abc_kwargs=abc_kwargs_solid,
        )

    F_interface_elastic = +p_n * dot(v_s, n_s) * ds_int(iface)
    F_solid = _drop_missing_facets(F_elastic + F_interface_elastic)

    solid_lhs = fire.lhs(F_solid)
    solid_rhs = fire.rhs(F_solid)

    solid_problem = LinearVariationalProblem(
        solid_lhs,
        solid_rhs,
        Wave_obj.X_np1.sub(1),
        constant_jacobian=True,
    )

    Wave_obj.solid_solver = LinearVariationalSolver(
        solid_problem,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "mat_type": "matfree",
        },
    )

    # -------------------------------------------------------------------------
    # Fluid sub-problem
    # -------------------------------------------------------------------------
    p_trial = TrialFunction(Wave_obj.scalar_function_space)
    v_f = TestFunction(Wave_obj.scalar_function_space)

    Wave_obj.surface_quadrature_rule = Wave_obj.surface_quadrature_rule_fluid

    F_acoustic = build_acoustic_form(
        Wave_obj,
        p_trial,
        v_f,
        p_n,
        p_nm1,
        Wave_obj.quadrature_rule_fluid,
    )

    u_tt = (Wave_obj.X_np1.sub(1) - 2.0 * u_n + u_nm1) / dt**2
    F_interface_acoustic = -dot(u_tt, n_s) * v_f * ds_int(iface)
    F_fluid = _drop_missing_facets(F_acoustic + F_interface_acoustic)

    fluid_lhs = fire.lhs(F_fluid)
    fluid_rhs = fire.rhs(F_fluid)

    Wave_obj.source_function = Cofunction(Wave_obj.function_space.dual())
    Wave_obj.source_function_fluid = Cofunction(Wave_obj.scalar_function_space.dual())

    fluid_problem = LinearVariationalProblem(
        fluid_lhs,
        fluid_rhs + Wave_obj.source_function_fluid,
        Wave_obj.X_np1.sub(0),
        constant_jacobian=True,
    )

    Wave_obj.fluid_solver = LinearVariationalSolver(
        fluid_problem,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "mat_type": "matfree",
        },
    )

    Wave_obj.solver = Wave_obj