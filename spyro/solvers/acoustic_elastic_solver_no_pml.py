import firedrake as fire

from firedrake import (
    Cofunction,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TestFunction,
    TrialFunction,
    dot,
)
from spyro.solvers.acoustic_solver_construction_no_pml import build_acoustic_form
from spyro.solvers.elastic_wave.forms import build_elastic_form

from matplotlib.ticker import MaxNLocator

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


def construct_acoustic_elastic(Wave_obj):
    p_nm1, u_nm1 = fire.split(Wave_obj.X_nm1)
    p_n, u_n = fire.split(Wave_obj.X_n)

    dt = Wave_obj.dt
    n_s = Wave_obj.n_s
    ds_int = Wave_obj.ds_int
    iface = Wave_obj.interface_id

    # Solid sub-problem:
    u_trial = TrialFunction(Wave_obj.vector_function_space)
    v_s = TestFunction(Wave_obj.vector_function_space)

    # F_elastic = build_elastic_form(
    #     Wave_obj, u_trial, v_s, u_n, u_nm1, Wave_obj.quadrature_rule_solid
    # )
    c_p = fire.sqrt((Wave_obj.lmbda + 2.0 * Wave_obj.mu) / Wave_obj.rho)
    c_s = fire.sqrt(Wave_obj.mu / Wave_obj.rho)
    F_elastic = build_elastic_form(
        Wave_obj, u_trial, v_s, u_n, u_nm1, Wave_obj.quadrature_rule_solid,
        abc_kwargs={"c_p": c_p, "c_s": c_s,
                    "qr_s": Wave_obj.surface_quadrature_rule_solid,
                    "boundary_ids": {"bottom": 2, "top": None, "left": 3, "right": 4}},
    )

    F_interface_elastic = +p_n * dot(v_s, n_s) * ds_int(iface)
    # F_solid = F_elastic + F_interface_elastic
    F_solid = _drop_missing_facets(F_elastic + F_interface_elastic)

    solid_lhs = fire.lhs(F_solid)
    solid_rhs = fire.rhs(F_solid)

    solid_problem = LinearVariationalProblem(
        solid_lhs,
        solid_rhs,
        Wave_obj.X_np1.sub(1),
        constant_jacobian=True,
    )

    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "jacobi",
        "mat_type": "matfree",
    }

    Wave_obj.solid_solver = LinearVariationalSolver(
        solid_problem, solver_parameters=solver_parameters
    )

    # Fluid sub-problem:
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
    # F_fluid = F_acoustic + F_interface_acoustic
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

    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "jacobi",
        "mat_type": "matfree",
    }

    Wave_obj.fluid_solver = LinearVariationalSolver(
        fluid_problem, solver_parameters=solver_parameters
    )

    Wave_obj.solver = Wave_obj
