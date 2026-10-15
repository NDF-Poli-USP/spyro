"""Checkpointed tape replayed before its first adjoint gives a wrong gradient.

Pure Firedrake/pyadjoint reproducer, no spyro. A central-difference wave
equation is solved for :math:`u^{n+1}` and taped with a checkpoint schedule;
each time step is closed right after the solve, as spyro does. Compared with
a tape without checkpointing:

- the gradient taken right after recording is exact;
- calling the reduced functional (even at the recorded control) before the
  first ``derivative()`` makes that derivative wrong;
- once one adjoint has run, later replays and derivatives are exact.

TAO calls ``rf(x0)`` before its first gradient, so every TAO inversion over a
checkpointed tape starts from a wrong gradient. Run with::

    python repro_replay_before_first_adjoint.py
"""
import numpy as np
import firedrake as fire
from firedrake.adjoint import (
    Control, ReducedFunctional, Tape, continue_annotation, get_working_tape,
    pause_annotation, set_working_tape,
)
from checkpoint_schedules import MixedCheckpointSchedule, StorageType

mesh = fire.UnitSquareMesh(6, 6)
V = fire.FunctionSpace(mesh, "CG", 1)
NT, DT = 40, 0.01


def record(c_values: np.ndarray, snapshots: int | None) -> ReducedFunctional:
    """Tape a forward solve at the velocity ``c_values``.

    Parameters
    ----------
    c_values : numpy.ndarray
        Nodal values of the velocity, the control.
    snapshots : int or None
        Checkpoints of a mixed schedule, or ``None`` for a plain tape.

    Returns
    -------
    pyadjoint.ReducedFunctional
        Reduced functional of the summed squared pressure.
    """
    set_working_tape(Tape())
    tape = get_working_tape()
    if snapshots:
        tape.enable_checkpointing(
            MixedCheckpointSchedule(NT, snapshots, storage=StorageType.RAM))
    continue_annotation()
    c = fire.Function(V)
    c.dat.data[:] = c_values
    u_nm1, u_n, u_np1 = (fire.Function(V) for _ in range(3))
    u, v = fire.TrialFunction(V), fire.TestFunction(V)
    x, y = fire.SpatialCoordinate(mesh)
    f = fire.Function(V).interpolate(fire.exp(-50 * ((x - .5)**2 + (y - .5)**2)))
    amp = fire.Function(fire.FunctionSpace(mesh, "R", 0))
    F = (1 / c**2) * (u - 2 * u_n + u_nm1) / DT**2 * v * fire.dx \
        + fire.inner(fire.grad(u_n), fire.grad(v)) * fire.dx \
        - amp * f * v * fire.dx
    problem = fire.LinearVariationalProblem(
        fire.lhs(F), fire.rhs(F), u_np1, constant_jacobian=True)
    solver = fire.LinearVariationalSolver(problem)
    J = 0.0
    for k in range(NT):
        amp.assign(np.sin(10 * k * DT))
        solver.solve()
        if snapshots and k < NT - 1:
            tape.end_timestep()
        u_nm1.assign(u_n)
        u_n.assign(u_np1)
        J += fire.assemble(u_n**2 * fire.dx)
    pause_annotation()
    return ReducedFunctional(J, Control(c))


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Relative difference of ``a`` from the nonzero ``b``.

    Parameters
    ----------
    a, b : numpy.ndarray
        Arrays to compare.

    Returns
    -------
    float
        ``||a - b|| / ||b||``.
    """
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def gradient(rf: ReducedFunctional) -> np.ndarray:
    """Return a copy of the derivative of ``rf``.

    Parameters
    ----------
    rf : pyadjoint.ReducedFunctional
        Reduced functional evaluated at the point of interest.

    Returns
    -------
    numpy.ndarray
        Values of the derivative.
    """
    return np.array(rf.derivative().dat.data_ro, copy=True)


rng = np.random.default_rng(0)
c0 = 1.0 + 0.2 * rng.random(V.dim())
c1 = c0 + 0.1 * rng.random(V.dim())


def control(values: np.ndarray) -> fire.Function:
    """Wrap nodal values as a control value.

    Parameters
    ----------
    values : numpy.ndarray
        Nodal values.

    Returns
    -------
    firedrake.Function
        Function holding ``values``.
    """
    function = fire.Function(V)
    function.dat.data[:] = values
    return function


g_c0 = gradient(record(c0, None))
g_c1 = gradient(record(c1, None))

rf = record(c0, 4)
print(f"derivative right after recording:        {relative(gradient(rf), g_c0):.1e}")
rf = record(c0, 4)
rf(control(c0))
print(f"rf(c0) then first derivative:            {relative(gradient(rf), g_c0):.1e}")
rf = record(c0, 4)
rf(control(c1))
print(f"rf(c1) then first derivative:            {relative(gradient(rf), g_c1):.1e}")
rf = record(c0, 4)
gradient(rf)
rf(control(c1))
print(f"derivative, rf(c1), derivative (2nd):    {relative(gradient(rf), g_c1):.1e}")
