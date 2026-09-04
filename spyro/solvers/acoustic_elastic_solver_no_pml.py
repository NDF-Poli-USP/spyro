import firedrake as fire

from firedrake import (Cofunction, LinearVariationalProblem,
                       LinearVariationalSolver, TestFunction,
                       TrialFunction, dot, lhs, rhs)
from spyro.solvers.acoustic_solver_construction_no_pml import build_acoustic_form
from spyro.solvers.elastic_wave.forms import build_elastic_form

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

    F_elastic = build_elastic_form(
        Wave_obj,
        u_trial,
        v_s,
        u_n,
        u_nm1,
        Wave_obj.quadrature_rule_solid
    )

    F_interface_elastic  = + p_n * dot(v_s, n_s) * ds_int(iface)
    F_solid = F_elastic + F_interface_elastic

    solid_lhs = fire.lhs(F_solid)
    solid_rhs = fire.rhs(F_solid)

    solid_problem = LinearVariationalProblem(
        solid_lhs,
        solid_rhs,
        Wave_obj.X_np1.sub(1),
        constant_jacobian=True,
    )

    solver_parameters = dict(Wave_obj.solver_parameters)
    solver_parameters["mat_type"] = "matfree"

    Wave_obj.solid_solver = LinearVariationalSolver(
        solid_problem, 
        solver_parameters=solver_parameters
    )

    # Wave_obj.solid_solver = LinearVariationalSolver(
    #     solid_problem,
    #     solver_parameters={
    #         'ksp_type': 'preonly', 'pc_type': 'lu',
    #         'pc_factor_mat_solver_type': 'mumps'
    #     }
    # )

    # Fluid sub-problem:
    p_trial = TrialFunction(Wave_obj.scalar_function_space)
    v_f = TestFunction(Wave_obj.scalar_function_space)

    F_acoustic = build_acoustic_form(
        Wave_obj, p_trial, v_f, p_n, p_nm1,
        Wave_obj.quadrature_rule_fluid, c=Wave_obj.c
    )

    u_tt = (Wave_obj.X_np1.sub(1) - 2.0*u_n + u_nm1) / dt**2
    F_interface_acoustic = - dot(u_tt, n_s) * v_f * ds_int(iface)
    F_fluid = F_acoustic + F_interface_acoustic

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

    solver_parameters = dict(Wave_obj.solver_parameters)
    solver_parameters["mat_type"] = "matfree"

    Wave_obj.fluid_solver = LinearVariationalSolver(
        fluid_problem, 
        solver_parameters=solver_parameters
    )

    # Wave_obj.fluid_solver = LinearVariationalSolver(
    #     fluid_problem,
    #     solver_parameters={
    #         'ksp_type': 'preonly', 'pc_type': 'lu',
    #         'pc_factor_mat_solver_type': 'mumps'
    #     }
    # )

    Wave_obj.solver = Wave_obj
