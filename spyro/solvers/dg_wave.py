import firedrake as fire
from firedrake import dot, grad, jump, avg, dx, dX, ds, dS, Constant
from spyro import Wave

fire.set_log_level(fire.ERROR)


class DG_Wave(Wave):
    def matrix_building(self):
        """Builds solver operators. Doesn't create mass matrices if matrix_free option is on,
        which it is by default.
        """
        V = self.function_space
        # Trial and test functions
        u = fire.TrialFunction(V)
        v = fire.TestFunction(V)

        ## Previous functions for time integration
        u_n = fire.Function(V)
        u_nm1 = fire.Function(V)
        self.u_nm1 = u_nm1
        self.u_n = u_n
        c = self.c

        self.current_time = 0.0
        dt = self.dt

        # Normal component, cell size and right-hand side
        h = fire.CellDiameter(self.mesh)
        h_avg = (h("+") + h("-")) / 2
        n = fire.FacetNormal(self.mesh)

        # Parameters
        alpha = 4.0
        gamma = 8.0

        # Bilinear form
        a = (
            dot(grad(v), grad(u)) * dx
            - dot(avg(grad(v)), jump(u, n)) * dS
            - dot(jump(v, n), avg(grad(u))) * dS
            + alpha / h_avg * dot(jump(v, n), jump(u, n)) * dS
            - dot(grad(v), u * n) * ds
            - dot(v * n, grad(u)) * ds
            + (gamma / h) * v * u * ds
            + ((u) / Constant(dt**2)) / c * v * dx
        )
        # Linear form
        b = ((2.0 * u_n - u_nm1) / Constant(dt**2)) / c * v * dx
        form = a - b

        lhs = fire.lhs(form)
        rhs = fire.rhs(form)

        A = fire.assemble(lhs)
        params = {"ksp_type": "gmres"}
        self.solver = fire.LinearSolver(A, solver_parameters=params)

        # lterar para como o thiago fez
        self.rhs = rhs
        B = fire.Function(V)
        self.B = B
