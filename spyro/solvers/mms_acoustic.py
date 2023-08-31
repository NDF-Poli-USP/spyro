import firedrake as fire
from .acoustic_wave import AcousticWave


class AcousticWaveMMS(AcousticWave):
    """Class for solving the acoustic wave equation in 2D or 3D using
    the finite element method. This class inherits from the AcousticWave class
    and overwrites the matrix_building method to use source propagated along
    the whole domain, which generates a known solution for comparison.
    """

    def matrix_building(self):
        super().matrix_building()
        lhs = self.lhs
        bcs = fire.DirichletBC(self.function_space, 0.0, "on_boundary")
        A = fire.assemble(lhs, bcs=bcs, mat_type="matfree")
        self.mms_source_in_space()
        self.solver = fire.LinearSolver(
            A, solver_parameters=self.solver_parameters
        )

    def mms_source_in_space(self):
        V = self.function_space
        self.q_xy = fire.Function(V)
        x = self.mesh_z
        y = self.mesh_x
        if self.dimension == 2:
            # xy = fire.project(sin(pi*x)*sin(pi*y), V)
            # self.q_xy.assign(xy)
            xy = fire.project((-(x**2) - x - y**2 + y), V)
            self.q_xy.assign(xy)
        elif self.dimension == 3:
            z = self.mesh_y
            # xyz = fire.project(sin(pi*x)*sin(pi*y)*sin(pi*z), V)
            # self.q_xy.assign(xyz)
            xyz = fire.project(
                (
                    -x * y * (x + 1) * (y - 1)
                    - x * z * (x + 1) * (z - 1)
                    - y * z * (y - 1) * (z - 1)
                ),
                V,
            )
            self.q_xy.assign(xyz)

        # self.q_xy.interpolate(sin(pi*x)*sin(pi*y))

    def mms_source_in_time(self, t):
        # return fire.Constant(2*pi**2*t**2 + 2.0)
        return fire.Constant(2 * t)

    def analytical_solution(self, t):
        self.analytical = fire.Function(self.function_space)
        x = self.mesh_z
        y = self.mesh_x
        # analytical = fire.project(sin(pi*x)*sin(pi*y)*t**2,
        # self.function_space)
        # self.analytical.interpolate(sin(pi*x)*sin(pi*y)*t**2)
        if self.dimension == 2:
            self.analytical.interpolate(x * (x + 1) * y * (y - 1) * t)
        elif self.dimension == 3:
            z = self.mesh_y
            self.analytical.interpolate(
                x * (x + 1) * y * (y - 1) * z * (z - 1) * t
            )
        # self.analytical.assign(analytical)

        return self.analytical
