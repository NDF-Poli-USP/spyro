import firedrake as fire
from firedrake import Constant, dx, ds, dot, grad, inner
from firedrake.assemble import create_assembly_callable

from ..io.basicio import ensemble_propagator, parallel_print
from . import helpers
from .. import utils
from .acousticNoPML import AcousticWaveNoPML
from ..pml import damping
from ..domains.quadrature import quadrature_rules


class AcousticWavePML(AcousticWaveNoPML):
    def matrix_building(self):
        self.current_time = 0.0
        V = self.function_space
        Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
        self.vector_function_space = Z
        quad_rule, k_rule, s_rule = quadrature_rules(V)
        self.quadrature_rule = quad_rule
        self.stiffness_quadrature_rule = k_rule
        self.surface_quadrature_rule = s_rule
        if self.dimension == 2:
            self.matrix_building_2d()
        elif self.dimension == 3:
            self.matrix_building_3d()
        else:
            raise ValueError("Only 2D and 3D supported")

    def matrix_building_2d(self):
        dt = self.dt
        c = self.c

        V = self.function_space
        Z = self.vector_function_space
        W = V * Z
        dxlump = dx(scheme=self.quadrature_rule)
        dslump = ds(scheme=self.surface_quadrature_rule)

        u, pp = fire.TrialFunctions(W)
        v, qq = fire.TestFunctions(W)

        X = fire.Function(W)
        X_n = fire.Function(W)
        X_nm1 = fire.Function(W)

        u_n, pp_n = X_n.split()
        u_nm1, _ = X_nm1.split()

        self.u_n = u_n
        self.X = X
        self.X_n = X_n
        self.X_nm1 = X_nm1

        sigma_x, sigma_z = damping.functions(self)
        Gamma_1, Gamma_2 = damping.matrices_2D(sigma_z, sigma_x)
        pml1 = (
            (sigma_x + sigma_z)
            * ((u - u_nm1) / Constant(2.0 * dt))
            * v
            * dxlump
        )

        # typical CG FEM in 2d/3d

        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dxlump
        a = c * c * dot(grad(u_n), grad(v)) * dxlump  # explicit

        nf = c * ((u_n - u_nm1) / dt) * v * dslump

        FF = m1 + a + nf

        B = fire.Function(W)

        pml2 = sigma_x * sigma_z * u_n * v * dxlump
        pml3 = inner(pp_n, grad(v)) * dxlump
        FF += pml1 + pml2 + pml3
        # -------------------------------------------------------
        mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dxlump
        mm2 = inner(dot(Gamma_1, pp_n), qq) * dxlump
        dd = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dxlump
        FF += mm1 + mm2 + dd

        lhs_ = fire.lhs(FF)
        rhs_ = fire.rhs(FF)

        A = fire.assemble(lhs_, mat_type="matfree")
        solver = fire.LinearSolver(A, solver_parameters=self.solver_parameters)
        self.solver = solver
        self.rhs = rhs_
        self.B = B

        return

    def matrix_building_3d(self):
        dt = self.dt
        c = self.c

        V = self.function_space
        Z = self.vector_function_space
        W = V * V * Z
        dxlump = dx(scheme=self.quadrature_rule)
        dslump = ds(scheme=self.surface_quadrature_rule)

        u, psi, pp = fire.TrialFunctions(W)
        v, phi, qq = fire.TestFunctions(W)

        X = fire.Function(W)
        X_n = fire.Function(W)
        X_nm1 = fire.Function(W)

        u_n, psi_n, pp_n = X_n.split()
        u_nm1, psi_nm1, _ = X_nm1.split()

        self.u_n = u_n
        self.X = X
        self.X_n = X_n
        self.X_nm1 = X_nm1

        sigma_x, sigma_y, sigma_z = damping.functions(self)
        Gamma_1, Gamma_2, Gamma_3 = damping.matrices_3D(
            sigma_x, sigma_y, sigma_z
        )

        pml1 = (
            (sigma_x + sigma_y + sigma_z)
            * ((u - u_nm1) / Constant(2.0 * dt))
            * v
            * dxlump
        )

        pml2 = (
            (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
            * u_n
            * v
            * dxlump
        )

        pml3 = (sigma_x * sigma_y * sigma_z) * psi_n * v * dxlump
        pml4 = inner(pp_n, grad(v)) * dxlump

        # typical CG FEM in 2d/3d

        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dxlump
        a = c * c * dot(grad(u_n), grad(v)) * dxlump  # explicit

        nf = c * ((u_n - u_nm1) / dt) * v * dslump

        FF = m1 + a + nf

        B = fire.Function(W)

        FF += pml1 + pml2 + pml3 + pml4
        # -------------------------------------------------------
        mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dxlump
        mm2 = inner(dot(Gamma_1, pp_n), qq) * dxlump
        dd1 = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dxlump
        dd2 = -c * c * inner(grad(psi_n), dot(Gamma_3, qq)) * dxlump
        FF += mm1 + mm2 + dd1 + dd2

        mmm1 = (dot((psi - psi_n), phi) / Constant(dt)) * dxlump
        uuu1 = (-u_n * phi) * dxlump
        FF += mmm1 + uuu1

        lhs_ = fire.lhs(FF)
        rhs_ = fire.rhs(FF)

        A = fire.assemble(lhs_, mat_type="matfree")
        solver = fire.LinearSolver(A, solver_parameters=self.solver_parameters)
        self.solver = solver
        self.rhs = rhs_
        self.B = B

        return
