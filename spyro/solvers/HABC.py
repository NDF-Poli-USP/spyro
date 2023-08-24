import firedrake as fire
from firedrake import Constant, dx, dot, grad, ds

from .acousticNoPML import AcousticWaveNoPML
from . import helpers
from .. import utils
from ..domains.quadrature import quadrature_rules


class HABC_wave(AcousticWaveNoPML):
    def __init__(self, dictionary=None, comm=None, eta=None, costet=None):
        super().__init__(dictionary=dictionary, comm=comm)
        self.eta = eta
        self.costet = costet

    def matrix_building(self):
        """Builds solver operators. Doesn't create mass matrices if matrix_free option is on,
        which it is by default.
        """
        eta = self.eta
        costet1 = self.costet
        c = self.c
        V = self.function_space
        quad_rule, k_rule, s_rule = quadrature_rules(V)

        # typical CG FEM in 2d/3d
        u = fire.TrialFunction(V)
        v = fire.TestFunction(V)

        u_nm1 = fire.Function(V)
        u_n = fire.Function(V)
        self.u_nm1 = u_nm1
        self.u_n = u_n

        self.current_time = 0.0
        dt = self.dt

        # -------------------------------------------------------
        m1 = (
            ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
            * v
            * dx(rule=quad_rule)
        )
        a = (
            self.c * self.c * dot(grad(u_n), grad(v)) * dx(rule=quad_rule)
        )  # explicit
        habc_form = eta * (u - u_n) / Constant(dt) * v * dx(rule=quad_rule)
        nf = costet1 * c * ((u_n - u_nm1) / dt) * v * ds(rule=s_rule)

        B = fire.Function(V)

        form = m1 + a + habc_form + nf
        lhs = fire.lhs(form)
        rhs = fire.rhs(form)

        A = fire.assemble(lhs, mat_type="matfree")
        self.solver = fire.LinearSolver(
            A, solver_parameters=self.solver_parameters
        )

        # lterar para como o thiago fez
        self.rhs = rhs
        self.B = B
