"""Constructs Firedrake solver for the acosutic wave with typical BCs, NRBCs or HABCs."""

import firedrake as fire
from firedrake import ds, dx, dot, grad
from ..utils.typing import AbsorbingBCsType


class CentralDifferenceAccelerationSolver:
    r"""Central-difference step split into an acceleration solve and an update.

    One call to :meth:`solve` advances the pressure by one time step,

    .. math::

        M(c)\,a^n = f^n - K u^n - g(u^n, u^{n-1}), \qquad
        u^{n+1} = 2u^n - u^{n-1} + \Delta t^2 a^n,

    where :math:`g` gathers the explicit absorbing terms. This is the same
    discretization as solving
    :math:`M(c)(u^{n+1} - 2u^n + u^{n-1})/\Delta t^2 = f^n - K u^n - g` for
    :math:`u^{n+1}`; only the unknown of the linear solve changes.

    Parameters
    ----------
    acceleration_solver : firedrake.LinearVariationalSolver
        Solver for :math:`a^n`, whose solution is ``acceleration``.
    u_np1 : firedrake.Function
        Pressure at the next time level, written by :meth:`solve`.
    u_n : firedrake.Function
        Pressure at the current time level.
    u_nm1 : firedrake.Function
        Pressure at the previous time level.
    acceleration : firedrake.Function
        Second time derivative :math:`a^n` of the pressure.
    dt : float
        Time step.

    Attributes
    ----------
    acceleration_solver : firedrake.LinearVariationalSolver
        Solver for :math:`a^n`.
    u_np1 : firedrake.Function
        Pressure at the next time level.
    update : ufl.core.expr.Expr
        The explicit update :math:`2u^n - u^{n-1} + \Delta t^2 a^n`.
    """

    def __init__(self, acceleration_solver: fire.LinearVariationalSolver,
                 u_np1: fire.Function, u_n: fire.Function,
                 u_nm1: fire.Function, acceleration: fire.Function,
                 dt: float) -> None:
        self.acceleration_solver = acceleration_solver
        self.u_np1 = u_np1
        self.update = 2.0 * u_n - u_nm1 + dt**2 * acceleration

    def solve(self) -> None:
        """Advance the pressure from ``u_n`` to ``u_np1``.

        Returns
        -------
        None
            ``u_np1`` and the acceleration are updated in place.
        """
        self.acceleration_solver.solve()
        self.u_np1.assign(self.update)


def build_acceleration_solver(
    wave: object, bcs: fire.DirichletBC | None = None,
) -> CentralDifferenceAccelerationSolver:
    """Build the time-step solver from the forms set up on ``wave``.

    Parameters
    ----------
    wave : spyro.AcousticWave
        Wave object whose ``lhs``, ``rhs``, ``source_function`` and pressure
        levels were built by :func:`construct_solver_or_matrix_no_pml`.
    bcs : firedrake.DirichletBC, optional
        Boundary conditions on the acceleration. A homogeneous condition on
        the acceleration keeps a time-independent Dirichlet value of the
        pressure.

    Returns
    -------
    CentralDifferenceAccelerationSolver
        The solver advancing ``wave.u_n`` to ``wave.u_np1``.
    """
    lin_var = fire.LinearVariationalProblem(
        wave.lhs,
        wave.rhs + wave.source_function,
        wave.acceleration, bcs=bcs, constant_jacobian=True)
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    acceleration_solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters,
    )
    return CentralDifferenceAccelerationSolver(
        acceleration_solver, wave.u_np1, wave.u_n, wave.u_nm1,
        wave.acceleration, wave.dt,
    )


def construct_solver_or_matrix_no_pml(wave):
    """Builds solver operators for wave propagator with typical BCs, NRBCs or HABCs.

    Doesn't create mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """
    V = wave.function_space
    quad_rule = wave.quadrature_rule

    # typical CG FEM in 2d/3d
    a_trial = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    u_nm1 = fire.Function(V, name="pressure t-dt")
    u_n = fire.Function(V, name="pressure")
    u_np1 = fire.Function(V, name="pressure t+dt")
    wave.u_nm1 = u_nm1
    wave.u_n = u_n
    wave.u_np1 = u_np1
    wave.acceleration = fire.Function(V, name="pressure acceleration")

    wave.current_time = 0.0
    dt = wave.dt

    # -------------------------------------------------------
    # The time step solves for the acceleration a = (u^{n+1} - 2u^n +
    # u^{n-1})/dt^2, every other term being explicit.
    m1 = (1 / (wave.c * wave.c)) * a_trial * v * dx(**quad_rule)
    a = dot(grad(u_n), grad(v)) * dx(**quad_rule)  # explicit

    le = 0.0
    q = wave.source_expression
    if q is not None:
        le += - q * v * dx(**quad_rule)

    if wave.abc_active and not wave.abc_get_ref_model:
        weak_expr_abc = dot((u_n - u_nm1) / dt, v)

        f_abc = (1 / wave.c) * weak_expr_abc
        qr_s = wave.surface_quadrature_rule

        if wave.abc_type == AbsorbingBCsType.HYBRID:

            # NRBC
            le += wave.cosHig * f_abc * ds(**qr_s)

            # Damping
            le += wave.eta_mask * weak_expr_abc * \
                (1 / (wave.c * wave.c)) * \
                wave.eta_habc * dx(**quad_rule)

        else:
            if wave.absorb_top:
                le += f_abc*ds(1, **qr_s)
            if wave.absorb_bottom:
                le += f_abc*ds(2, **qr_s)
            if wave.absorb_right:
                le += f_abc*ds(3, **qr_s)
            if wave.absorb_left:
                le += f_abc*ds(4, **qr_s)
            if wave.dimension == 3:
                if wave.absorb_front:
                    le += f_abc*ds(5, **qr_s)
                if wave.absorb_back:
                    le += f_abc*ds(6, **qr_s)

    # form = m1 + a - le
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    # TODO: Add citation
    form = m1 + a + le
    wave.rhs = fire.rhs(form)
    wave.lhs = fire.lhs(form)
    wave.source_function = fire.Cofunction(V.dual())
    wave.solver = build_acceleration_solver(wave)
