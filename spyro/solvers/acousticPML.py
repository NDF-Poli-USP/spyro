import firedrake as fire

from .acousticNoPML import AcousticWaveNoPML
from ..domains.quadrature import quadrature_rules
from .acoustic_solver_construction_with_pml import construct_solver_or_matrix_with_pml_2d
from .acoustic_solver_construction_with_pml import construct_solver_or_matrix_with_pml_3d


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

        self.u_n = None
        self.X = None
        self.X_n = None
        self.X_nm1 = None

        self.solver = None
        self.rhs = None
        self.B = None

        if self.dimension == 2:
            construct_solver_or_matrix_with_pml_2d(self)
        elif self.dimension == 3:
            construct_solver_or_matrix_with_pml_3d(self)
        else:
            raise ValueError("Only 2D and 3D supported")
