import firedrake as fire
import ufl
from .acoustic_wave import AcousticWave
from ..utils.typing import override


class AcousticWaveMMS(AcousticWave):
    """Class for solving the acoustic wave equation in 2D or 3D using
    the finite element method. This class inherits from the AcousticWave class
    and overwrites the matrix_building method to use source propagated along
    the whole domain, which generates a known solution for comparison.
    """

    @override
    def matrix_building(self):
        self.mms_source_in_space()
        # The manufactured solution is linear in time, so the source it
        # needs is ``2 t q(x)``; written in the time coefficient, both the
        # central-difference loop and the Irksome stages evaluate it at
        # their own times.
        self.source_expression = 2 * self.time * self.q_xy
        self.bcs = [fire.DirichletBC(self.function_space, 0.0, "on_boundary")]

        super().matrix_building()

        if self.uses_irksome:
            time = ufl.variable(self.time)
            solution = self.analytical_expression(time)
            self.u_n.interpolate(solution)
            self.u_t.interpolate(ufl.diff(solution, time))
        else:
            dt = self.dt
            t = self.current_time
            self.u_nm1.assign(self.analytical_solution(t - 2 * dt))
            self.u_n.assign(self.analytical_solution(t - dt))

    def mms_source_in_space(self):
        V = self.function_space
        self.q_xy = fire.Function(V)
        x = self.mesh_z
        y = self.mesh_x
        if self.dimension == 2:
            # xy = fire.project(sin(pi*x)*sin(pi*y), V)
            # self.q_xy.assign(xy)
            self.q_xy.interpolate(-(x**2) - x - y**2 + y)
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

    def analytical_expression(self, t):
        """Return the manufactured solution as a UFL expression.

        Parameters
        ----------
        t : float or ufl.core.expr.Expr
            The time, a number or a UFL expression such as a
            ``firedrake.Constant``.

        Returns
        -------
        ufl.core.expr.Expr
            The manufactured pressure at time ``t``.
        """
        x = self.mesh_z
        y = self.mesh_x
        # analytical = fire.project(sin(pi*x)*sin(pi*y)*t**2,
        # self.function_space)
        # self.analytical.interpolate(sin(pi*x)*sin(pi*y)*t**2)
        if self.dimension == 2:
            return x * (x + 1) * y * (y - 1) * t
        z = self.mesh_y
        return x * (x + 1) * y * (y - 1) * z * (z - 1) * t

    def analytical_solution(self, t):
        self.analytical = fire.Function(self.function_space)
        self.analytical.interpolate(self.analytical_expression(t))
        # self.analytical.assign(analytical)

        return self.analytical

    @override
    def update_source_expression(self, t):
        self.time.assign(t)
