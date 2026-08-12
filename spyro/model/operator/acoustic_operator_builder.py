
import firedrake as fire
from firedrake import ds, dx, Constant, dot, grad
from spyro.model.boundary import Boundary
from spyro.model.solver_input import SolverInput

class AcousticSolverOperatorBuilder(SolverOperatorBuilder):
    def build(wave: Wave, solver_input: SolverInput):
        V = wave.function_space
        quad_rule = wave.quadrature_rule

        u = fire.TrialFunction(V)
        v = fire.TestFunction(V)
        
        u_nm1 = fire.Function(V, name="pressure t-dt")
        u_n = fire.Function(V, name="pressure")
        u_np1 = fire.Function(V, name="pressure t+dt")

        dt = solver_input.time_axis.dt

        m1 = (
            (1 / (wave.c * wave.c))
            * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
            * v
            * dx(**quad_rule)
        )

        a = dot(grad(u_n), grad(v)) * dx(**quad_rule)

        le = 0.0

        if solver_input.boundary:
            weak_expr_abc = dot((u_n - u_nm1) / Constant(dt), v)

            f_abc = (1 / wave.c) * weak_expr_abc
            qr_s = wave.surface_quadrature_rule

            if wave.abc_boundary_layer_type == "hybrid":

                # NRBC
                le += Wave_object.cosHig * f_abc * ds(**qr_s)

                # Damping
                le += Wave_object.eta_mask * weak_expr_abc * \
                    (1 / (Wave_object.c * Wave_object.c)) * \
                    Wave_object.eta_habc * dx(**quad_rule)

            else:
                if Wave_object.absorb_top:
                    le += f_abc*ds(1, **qr_s)
                if Wave_object.absorb_bottom:
                    le += f_abc*ds(2, **qr_s)
                if Wave_object.absorb_right:
                    le += f_abc*ds(3, **qr_s)
                if Wave_object.absorb_left:
                    le += f_abc*ds(4, **qr_s)
                if Wave_object.dimension == 3:
                    if Wave_object.absorb_front:
                        le += f_abc*ds(5, **qr_s)
                    if Wave_object.absorb_back:
                        le += f_abc*ds(6, **qr_s)
        form = m1 + a + le

        lin_var = fire.LinearVariationalProblem(
            fire.rhs(form),
            fire.lhs(form) + fire.Cofunction(V.dual()),
            u_np1, constant_jacobian=True)
        
        return fire.LinearVariationalSolver(
            lin_var, solver_parameters= solver_input.solver_parameters | {"mat_type": "matfree"},
        )
