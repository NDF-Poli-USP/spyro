import firedrake as fire
from firedrake import (Cofunction, LinearVariationalProblem,
                       LinearVariationalSolver, TestFunctions,
                       TrialFunctions, dot)

from spyro.solvers.acoustic_solver_construction_no_pml import build_acoustic_form
from spyro.solvers.elastic_wave.forms import build_elastic_form


def construct_acoustic_elastic_monolithic(Wave_obj):
    p_nm1, u_nm1 = fire.split(Wave_obj.X_nm1)
    p_n, u_n     = fire.split(Wave_obj.X_n)

    p_trial, u_trial = TrialFunctions(Wave_obj.function_space)
    v_f, v_s         = TestFunctions(Wave_obj.function_space)

    dt     = Wave_obj.dt
    n_s    = Wave_obj.n_s
    ds_int = Wave_obj.ds_int
    iface  = Wave_obj.interface_id

    F_acoustic = build_acoustic_form(
        Wave_obj, p_trial, v_f, p_n, p_nm1,
        Wave_obj.quadrature_rule_fluid, c=Wave_obj.c, implicit=False
    )
    F_elastic = build_elastic_form(
        Wave_obj, u_trial, v_s, u_n, u_nm1,
        Wave_obj.quadrature_rule_solid, implicit=False
    )

    u_tt_explicit = (u_trial - 2.0*u_n + u_nm1) / dt**2
    F_interface_acoustic = - dot(u_tt_explicit, n_s) * v_f * ds_int(iface)
    F_interface_elastic  = p_n * dot(v_s, n_s) * ds_int(iface) 

    F_total = F_acoustic + F_elastic + F_interface_acoustic + F_interface_elastic

    Wave_obj.lhs = fire.lhs(F_total)
    Wave_obj.rhs = fire.rhs(F_total)

    Wave_obj.source_function = Cofunction(Wave_obj.function_space.dual())

    lin_var_prob = LinearVariationalProblem(
        Wave_obj.lhs,
        Wave_obj.rhs + Wave_obj.source_function,
        Wave_obj.X_np1,
        constant_jacobian=True,
    )

    solver_parameters = dict(Wave_obj.solver_parameters)
    solver_parameters = {
        'mat_type': 'nest',
        'ksp_type': 'preonly',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
        'pc_fieldsplit_diag_use_amat': True,
        'pc_fieldsplit_0_fields': '1',
        'pc_fieldsplit_1_fields': '0',
        'fieldsplit_0_ksp_type': 'preonly',
        'fieldsplit_0_pc_type': 'jacobi',
        'fieldsplit_1_ksp_type': 'preonly',
        'fieldsplit_1_pc_type': 'jacobi',
    }
    Wave_obj.solver = LinearVariationalSolver(
        lin_var_prob, solver_parameters=solver_parameters
    )