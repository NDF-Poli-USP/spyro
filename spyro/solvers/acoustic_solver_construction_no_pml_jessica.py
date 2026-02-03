import firedrake as fire
from firedrake import ds, dx, Constant, dot, grad

def construct_solver_or_matrix_no_pml_jessica(Wave_object):
    V = Wave_object.function_sapace

    quad_rule = Wave_object.quadrature_rule # Como mudar para a que implementei no Firedrake?

    u = fire.TrialFunction(V)
    v = fire.Test.Function(V)

    u_np1 = fire.Function(V)
    u_n = fire.Function(V)
    u_nm1 = fire.Function(V)

    Wave_object.u_np1 = u_np1
    Wave_object.u_n = u_n
    Wave_object.u_nm1 = u_nm1

    Wave_object.current_time = 0.0
    dt = Wave_object.dt

    m1 = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dx(**quad_rule)

    a = Wave_object.c * Wave_object.c * dot(grad(u_n), grad(v)) * dx(**quad_rule)
    
    le = 0.0
    q = Wave_object.source_expression
    if q is not None:
        le += -q * v * dx(**quad_rule) # O que está acontecendo aqui?
        
    B = fire.Cofunction(V.dual())

    form = m1 + a - le

    lhs = fire.lhs(form)
    rhs = fire.rhs(form)
    Wave_object.lhs = lhs

    A = fire.assemble(lhs)
    Wave_object.solver = fire.LinearSolver(A, solver_parameters=Wave_object.solver_parameters)

    Wave_object.rhs = rhs
    Wave_object.B = B
