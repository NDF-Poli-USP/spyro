import firedrake as fire

from firedrake import (Cofunction, LinearVariationalProblem,
                       LinearVariationalSolver, TestFunctions,
                       TrialFunctions, dot, lhs, rhs)
from spyro.solvers.acoustic_solver_construction_no_pml import build_acoustic_form
from spyro.solvers.elastic_wave.forms import build_elastic_form

def construct_acoustic_elastic(Wave_obj):
    p_nm1, u_nm1 = fire.split(Wave_obj.X_nm1)
    p_n, u_n     = fire.split(Wave_obj.X_n)

    p_trial, u_trial = TrialFunctions(Wave_obj.function_space)
    v_f, v_s         = TestFunctions(Wave_obj.functio_space)

    F_acoustic = build_acoustic_form(
        Wave_obj, p_trial, v_f, p_n, p_nm1,
        Wave_obj.quadratrure_rule_fluid, c=Wave_obj.c, implicit=True
    )

    F_elastic = build_elastic_form(
        Wave_obj, u_trial, v_s, u_n, u_nm1,
        Wave_obj.quadrature_rule_solid, implicit=True
    )

    dt     = Wave_obj.dt
    n_s    = Wave_obj.n_s
    ds_int = Wave_obj.ds_int
    iface  = Wave_obj.interface_id

    u_tt = (u_trial - 2.0*u_n + u_nm1) / dt**2

    F_interface_acoustic = - dot(u_tt, n_s) * v_f * ds_int(iface)
    F_interface_elastic  = + p_trial * dot(v_s, n_s) * ds_int(iface)

    F_total = F_acoustic + F_elastic + F_interface_acoustic + F_interface_elastic

    Wave_obj.lhs = lhs(F_total)
    Wave_obj.rhs = rhs(F_total)

    Wave_obj.source_function = Cofunction(Wave_obj.function_space.dual())

    lin_var_prob = LinearVariationalProblem(
        Wave_obj.lhs,
        Wave_obj.rhs + Wave_obj.source_function,
        Wave_obj.X_np1,
        constant_jacobian=True,
    )

    Wave_obj.solver = LinearVariationalSolver(
        lin_var_prob,
        solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        }
    )