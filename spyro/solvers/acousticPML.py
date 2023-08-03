import firedrake as fire
from firedrake import Constant, dot, dx, grad
from CG_acoustic import AcousticWave
from ..domains.quadrature import quadrature_rules
from ..pml import damping


class AcousticWavePML(AcousticWave):
    def matrix_building_2d(self):
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        dt = self.dt
        V = self.function_space
        Z = self.vector_function_space
        quad_rule, k_rule, s_rule = quadrature_rules(V)
        self.quadrature_rule = quad_rule

        W = V * Z
        u, pp = fire.TrialFunctions(W)
        v, qq = fire.TestFunctions(W)
        z = self.mesh_z
        x = self.mesh_x
        # self.trial_function = u

        u_nm1, pp_nm1 = fire.Function(W).split()
        u_n, pp_n = fire.Function(W).split()
        self.u_nm1 = u_nm1
        self.u_n = u_n
        self.pp_nm1 = pp_nm1
        self.pp_n = pp_n

        # Getting PML parameters
        sigma_x, sigma_z = damping.functions(
                model, V, dim, x, x1, x2, a_pml, z, z1, z2, c_pml
            )

        # -------------------------------------------------------
        m1 = (
            (1 / (self.c * self.c))
            * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
            * v
            * dx(scheme=quad_rule)
        )
        a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)  # explicit

        B = fire.Function(V)

        form = m1 + a
        lhs = fire.lhs(form)
        rhs = fire.rhs(form)
        self.lhs = lhs

        A = fire.assemble(lhs, mat_type="matfree")
        self.solver = fire.LinearSolver(
            A, solver_parameters=self.solver_parameters
        )

        # lterar para como o thiago fez
        self.rhs = rhs
        self.B = B

    def matrix_building_3d(self):
        V = self.function_space
        Z = self.vector_function_space

        W = V * V * Z
        u, psi, pp = fire.TrialFunctions(W)
        v, phi, qq = fire.TestFunctions(W)
        # self.trial_function = u

        u_nm1, psi_nm1, pp_nm1 = fire.Function(W).split()
        u_n, psi_n, pp_n = fire.Function(W).split()
        self.u_nm1 = u_nm1
        self.u_n = u_n
        self.pp_nm1 = pp_nm1
        self.pp_n = pp_n
        self.psi_nm1 = psi_nm1
        self.psi_n = psi_n

    def matrix_building(self):
        self.current_time = 0.0

        V = self.function_space
        Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
        self.vector_function_space = Z
        if self.dimension == 2:
            self.matrix_building_2d()
        elif self.dimension == 3:
            self.matrix_building_3d()