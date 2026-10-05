"""The lumped L2 change of variables spyro's TAO inversions run in.

``minimize_with_tao`` hands TAO the controls ``z = M_L^{1/2} m``, in which
the lumped L2 inner product is the Euclidean one. These tests check the
transformation itself, and that TAO, running its own Euclidean quasi-Newton
method on ``z``, behaves as an L2 method on ``m``: on an L2 least-squares
problem its first step lands on the minimizer, on any mesh.
"""
import firedrake as fire
import numpy as np
import pytest
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    set_working_tape,
    taylor_test,
)
from pyadjoint import MinimizationProblem, TAOSolver, Tape
from pyadjoint.optimization.tao_solver import TAOConvergenceError

from spyro.domains.quadrature import quadrature_rules
from spyro.tools.optimization import (
    LumpedL2TransformedFunctional,
    minimize_with_tao,
)


@pytest.fixture(autouse=True)
def fresh_tape():
    """Record each test on a tape of its own."""
    tape = Tape()
    set_working_tape(tape)
    continue_annotation()
    yield
    pause_annotation()
    tape.clear_tape()


def kmv_space(n: int) -> fire.FunctionSpace:
    """Return a mass-lumped quadratic space on an ``n`` by ``n`` mesh.

    Parameters
    ----------
    n : int
        Cells per side.

    Returns
    -------
    firedrake.FunctionSpace
        KMV space of degree 2, whose lumped mass varies between vertex, edge
        and interior nodes, so that it is far from a multiple of the identity.
    """
    return fire.FunctionSpace(fire.UnitSquareMesh(n, n), "KMV", 2)


def reference(space: fire.FunctionSpace) -> fire.Function:
    """Return a smooth field peaking at 1.8 around (0.4, 0.6).

    Parameters
    ----------
    space : firedrake.FunctionSpace
        Space to interpolate into.

    Returns
    -------
    firedrake.Function
        The field.
    """
    x, y = fire.SpatialCoordinate(space.mesh())
    return fire.Function(space).interpolate(
        1.0 + 0.8 * fire.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.02)
    )


def lumped_misfit(controls, targets, power: int = 1) -> ReducedFunctional:
    r"""Return :math:`\sum_i \frac12 \|m_i^p - t_i\|^2` in the lumped metric.

    Parameters
    ----------
    controls : list of firedrake.Function
        The controls :math:`m_i`.
    targets : list of firedrake.Function
        The targets :math:`t_i`.
    power : int, optional
        The power :math:`p`; 1 makes this an L2 least-squares problem.

    Returns
    -------
    pyadjoint.ReducedFunctional
        The functional of the controls.
    """
    functional = 0.0
    for control, target in zip(controls, targets):
        quadrature, _, _ = quadrature_rules(control.function_space())
        measure = fire.dx(**quadrature) if quadrature else fire.dx
        functional += fire.assemble(
            0.5 * (control ** power - target) ** 2 * measure
        )
    controls = [Control(control) for control in controls]
    return ReducedFunctional(functional, controls[0] if len(controls) == 1 else controls)


def test_transformed_functional_matches_the_model_functional():
    space = kmv_space(4)
    m = fire.Function(space).assign(1.2)
    reduced_functional = lumped_misfit([m], [reference(space)], power=2)
    transformed = LumpedL2TransformedFunctional(reduced_functional)
    z = transformed.controls[0].control

    assert np.isclose(float(transformed(z)), float(reduced_functional(m)))
    (m_back,) = transformed.map_result(z)
    assert np.allclose(m_back.dat.data_ro, m.dat.data_ro)

    direction = fire.Function(space).interpolate(
        fire.sin(3 * fire.SpatialCoordinate(space.mesh())[0])
    )
    assert taylor_test(transformed, z, direction) > 1.9


@pytest.mark.parametrize("n", [4, 8])
def test_l2_least_squares_takes_one_step(n):
    """The first BQNLS step solves an L2 least-squares problem, on any mesh.

    PETSc scales the first quasi-Newton step by :math:`2|f|/\\|g\\|^2`. In
    :math:`z` that is exactly the step to the minimizer of
    :math:`\\frac12\\|z - z^*\\|^2`; on the untransformed coefficients, where
    the gradient is :math:`M_L (m - m^*)`, it is not.
    """
    space = kmv_space(n)
    target = reference(space)
    options = {"tao_type": "bqnls", "tao_gatol": 1e-10, "tao_grtol": 0.0,
               "tao_gttol": 0.0, "tao_max_it": 20}

    iterations = []
    reduced_functional = lumped_misfit([fire.Function(space).assign(1.0)], [target])
    (result,) = minimize_with_tao(
        reduced_functional, options=options,
        record=lambda iteration, functional, controls: iterations.append(iteration),
    )
    assert iterations == [1]
    assert np.allclose(result.dat.data_ro, target.dat.data_ro, atol=1e-8)

    # The coefficients themselves, in the Euclidean metric, need more steps:
    # here more than the whole budget, which TAO reports as an error.
    reduced_functional = lumped_misfit([fire.Function(space).assign(1.0)], [target])
    plain = TAOSolver(MinimizationProblem(reduced_functional), options)
    with pytest.raises(TAOConvergenceError, match="DIVERGED_MAXITS"):
        plain.solve()


def test_bounds_are_projected_exactly():
    """With the metric diagonal, the bounded minimizer is the clipped target.

    Two controls in different spaces, each with bounds of its own.
    """
    space = kmv_space(6)
    cells = fire.FunctionSpace(space.mesh(), "DG", 0)
    targets = [reference(space), fire.Function(cells).interpolate(reference(space))]
    controls = [fire.Function(space, name="kmv").assign(1.0),
                fire.Function(cells, name="dg").assign(1.0)]
    bounds = [(0.9, 1.5), (1.1, 1.6)]

    result = minimize_with_tao(
        lumped_misfit(controls, targets), bounds=bounds,
        options={"tao_type": "bqnls", "tao_gatol": 1e-8, "tao_grtol": 0.0,
                 "tao_gttol": 0.0, "tao_max_it": 20},
    )

    assert [control.name() for control in result] == ["kmv", "dg"]
    for control, target, (lower, upper) in zip(result, targets, bounds):
        assert np.allclose(control.dat.data_ro,
                           np.clip(target.dat.data_ro, lower, upper),
                           rtol=0.0, atol=1e-7)


def test_a_mass_that_does_not_lump_is_rejected():
    """Quadratic Lagrange vertex functions integrate to zero on triangles."""
    space = fire.FunctionSpace(fire.UnitSquareMesh(2, 2), "CG", 2)
    reduced_functional = lumped_misfit(
        [fire.Function(space).assign(1.0)], [fire.Function(space)],
    )
    with pytest.raises(ValueError, match="lumped mass"):
        LumpedL2TransformedFunctional(reduced_functional)
