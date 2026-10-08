"""Regression coverage for the latent map and the TAO failure contract."""

import firedrake as fire
import numpy as np
import pytest
from petsc4py import PETSc

from .test_lumped_l2_functional import (  # noqa: F401
    _space, fresh_tape, lumped_misfit, reference,
)

pytestmark = pytest.mark.newer_firedrake


@pytest.mark.parametrize("bounds", [[], [(1, 0)], [(0, np.inf)], [(np.nan, 1)]])
def test_invalid_box_bounds(bounds: list) -> None:
    """Reject missing, reversed or nonfinite boxes in the latent map.

    Parameters
    ----------
    bounds : list
        Missing, reversed or nonfinite bounds.
    """
    from spyro.reduced_functionals import LatentReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    base = lumped_misfit(fire.Function(space).assign(0.4), reference(space))
    with pytest.raises(ValueError):
        LatentReducedFunctional(base, bounds)


@pytest.mark.parametrize("reason", [
    PETSc.TAO.Reason.DIVERGED_LS_FAILURE, PETSc.TAO.Reason.DIVERGED_NAN,
])
def test_tao_failures_propagate(monkeypatch: pytest.MonkeyPatch, reason: int) -> None:
    """Do not turn fatal TAO termination into a successful result.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Inject a termination reason into the real TAO adapter.
    reason : int
        PETSc termination code that must propagate as an exception.
    """
    from spyro.tools import optimization

    def fail(solver: object) -> None:
        """Simulate a failed solve.

        Parameters
        ----------
        solver : pyadjoint.TAOSolver
            Solver whose PETSc termination reason is set.

        Raises
        ------
        TAOConvergenceError
            Always, to exercise the failure path.
        """
        solver.tao.setConvergedReason(reason)
        raise optimization.TAOConvergenceError("injected failure")

    monkeypatch.setattr(optimization.TAOSolver, "solve", fail)
    space = _space("mass_lumped_triangle", 2)
    base = lumped_misfit(fire.Function(space).assign(0.4), reference(space))
    with pytest.raises(optimization.TAOConvergenceError, match="injected failure"):
        optimization.minimize_with_tao(base)
