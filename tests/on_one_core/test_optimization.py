"""Minimizing with TAO in the lumped L2 metric.

``minimize_with_tao`` runs TAO over ``LumpedL2ReducedFunctional``. These
tests check that TAO, running its own Euclidean quasi-Newton method on
``z = M_L^{1/2} m``, behaves as an L2 method on ``m``: on an L2 least-squares
problem its first step lands on the minimizer, on any mesh.
"""
import warnings

import firedrake as fire
import numpy as np
import pytest
from pyadjoint import MinimizationProblem, TAOSolver
from pyadjoint.optimization.tao_solver import TAOConvergenceError

from spyro.tools.optimization import minimize_with_tao

from .test_lumped_l2_functional import (  # noqa: F401 - fresh_tape is autouse
    fresh_tape,
    kmv_space,
    lumped_misfit,
    reference,
)


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


def test_bqnls_is_the_default():
    """Left to itself TAO would pick LMVM, which warns and ignores bounds."""
    space = kmv_space(4)
    reduced_functional = lumped_misfit(
        [fire.Function(space).assign(1.0)], [reference(space)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        minimize_with_tao(
            reduced_functional, bounds=[(0.5, 2.0)],
            options={"tao_gatol": 1e-10, "tao_grtol": 0.0, "tao_gttol": 0.0,
                     "tao_max_it": 20},
        )


@pytest.mark.parametrize("tao_type, bounds, reason", [
    ("lmvm", None, "fixed initial Hessian"),
    ("blmvm", None, "fixed initial Hessian"),
    ("blmvm", None, "unit step"),
    ("lmvm", [(0.5, 2.0)], "ignores the bounds"),
])
def test_types_that_are_not_recommended_warn(tao_type, bounds, reason):
    """LMVM and BLMVM warn, with the reasons that apply to each."""
    space = kmv_space(4)
    reduced_functional = lumped_misfit(
        [fire.Function(space).assign(1.0)], [reference(space)],
    )
    with pytest.warns(UserWarning, match=reason):
        minimize_with_tao(reduced_functional, bounds=bounds,
                          options={"tao_type": tao_type, "tao_max_it": 1})
