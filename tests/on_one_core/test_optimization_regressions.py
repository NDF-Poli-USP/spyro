"""Regression coverage for latent and proximal solver contracts."""

from functools import partial
from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest
from firedrake.adjoint import Control, ReducedFunctional, taylor_test
from petsc4py import PETSc

import spyro
from .test_lumped_l2_functional import (  # noqa: F401
    _space, fresh_tape, lumped_misfit, reference,
)

pytestmark = pytest.mark.newer_firedrake


@pytest.mark.parametrize("kind", ["l2", "bregman"])
@pytest.mark.parametrize("seed", [0.0, 2.5, -1.0])
def test_proximal_adjoint_seed(kind: str, seed: float) -> None:
    """Scale the entire derivative by the adjoint seed.

    Parameters
    ----------
    kind : str
        Proximal divergence.
    seed : float
        Adjoint seed applied to both parts of the functional.
    """
    from spyro.reduced_functionals import ProximalReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    model = fire.Function(space).assign(0.4)
    term = ProximalReducedFunctional(
        lumped_misfit(model, reference(space)), kind,
        bounds=[(0.0, 1.0)], scales=[2.0], step=0.7,
    )
    term(fire.Function(space).assign(0.6))
    expected = seed * term.derivative().dat.data_ro.copy()
    assert np.allclose(term.derivative(adj_input=seed).dat.data_ro, expected)


@pytest.mark.parametrize("point", [0.0, 2e-13, 1e-12, 0.3, 1 - 2e-13, 1.0])
def test_bregman_endpoint_derivative(point: float) -> None:
    """Keep the Bregman value and derivative consistent near both endpoints.

    Parameters
    ----------
    point : float
        Constant model at which to compare a directional derivative.
    """
    from spyro.reduced_functionals import ProximalReducedFunctional

    space = fire.FunctionSpace(fire.UnitSquareMesh(2, 2), "DG", 0)
    model = fire.Function(space).assign(0.4)
    term = ProximalReducedFunctional(
        lumped_misfit(model, fire.Function(space)), "bregman",
        bounds=[(0.0, 1.0)],
    )
    value = fire.Function(space).assign(point)
    term(value)
    slope = (term.derivative().dat.data_ro
             - term._functional.derivative().dat.data_ro).sum()
    step = 1e-14 if point < 1e-10 or point > 1 - 1e-10 else 1e-6
    plus = fire.Function(space).assign(point + step)
    minus = fire.Function(space).assign(point - step)
    # The smooth entropy continuation is also defined just outside the box.
    width = plus.dat.data_ro[0] - minus.dat.data_ro[0]
    numerical = (term.proximal_value(plus) - term.proximal_value(minus)) / width
    assert np.isfinite(slope)
    assert np.isclose(slope, numerical, rtol=5e-3, atol=1e-5)


@pytest.mark.parametrize("kind", ["l2", "bregman"])
def test_latent_proximal_composition(kind: str) -> None:
    """Verify the derivative of the compositions used by the inversion driver.

    Parameters
    ----------
    kind : str
        L2 in latent coordinates or Bregman in physical coordinates.
    """
    from spyro.reduced_functionals import (
        LatentReducedFunctional, ProximalReducedFunctional,
    )

    space = _space("mass_lumped_triangle", 2)
    model = fire.Function(space).assign(0.4)
    base = lumped_misfit(model, reference(space))
    if kind == "l2":
        mapped = LatentReducedFunctional(base, [(0.0, 1.0)])
        functional = ProximalReducedFunctional(mapped, kind)
    else:
        term = ProximalReducedFunctional(base, kind, bounds=[(0.0, 1.0)])
        functional = LatentReducedFunctional(term, [(0.0, 1.0)])
    point = functional.controls[0].control.copy(deepcopy=True)
    point.dat.data[:] += 0.3
    assert taylor_test(functional, point, fire.Function(space).assign(0.7)) > 1.9


@pytest.mark.parametrize("settings", [
    {"step": 0}, {"step": -1}, {"step": np.nan}, {"step": np.inf},
    {"scales": [-1]}, {"scales": [np.nan]}, {"scales": 1},
    {"outer_iterations": 1.9}, {"outer_iterations": 0},
    {"outer_iterations": True}, {"bound_margin": 0.5},
    {"bound_margin": -0.1}, {"bound_margin": np.nan},
])
def test_invalid_proximal_settings(settings: dict) -> None:
    """Reject invalid settings before starting the inversion.

    Parameters
    ----------
    settings : dict
        Invalid overrides of the default L2 proximal configuration.
    """
    with pytest.raises(ValueError):
        spyro.FullWaveformInversion._proximal({"kind": "l2", **settings})


@pytest.mark.parametrize("bounds", [[], [(1, 0)], [(0, np.inf)], [(np.nan, 1)]])
@pytest.mark.parametrize("latent", [False, True])
def test_invalid_box_bounds(bounds: list, latent: bool) -> None:
    """Reject invalid boxes in both wrappers.

    Parameters
    ----------
    bounds : list
        Missing, reversed or nonfinite bounds.
    latent : bool
        Whether to construct the latent wrapper instead of Bregman.
    """
    from spyro.reduced_functionals import (
        LatentReducedFunctional, ProximalReducedFunctional,
    )
    space = _space("mass_lumped_triangle", 2)
    base = lumped_misfit(fire.Function(space).assign(0.4), reference(space))
    with pytest.raises(ValueError):
        if latent:
            LatentReducedFunctional(base, bounds)
        else:
            ProximalReducedFunctional(base, "bregman", bounds=bounds)


@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("kind", ["l2", "bregman"])
def test_stage_weights_follow_active_controls(
    monkeypatch: pytest.MonkeyPatch, latent: bool, kind: str,
) -> None:
    """Select each parameter's weight when dispatching partial functionals.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replace TAO with an observer of its input functional.
    latent : bool
        Whether to compose the proximal term with the latent map.
    kind : str
        Proximal divergence.
    """
    from spyro.reduced_functionals import LatentReducedFunctional
    from spyro.tools import optimization

    space = _space("mass_lumped_triangle", 2)
    vp = fire.Function(space, name="vp").assign(0.4)
    vs = fire.Function(space, name="vs").assign(0.5)
    objective = fire.assemble((vp ** 2 + vs ** 2) * fire.dx)
    P, S = (spyro.ElasticMaterialParameter.P_WAVE_VELOCITY,
            spyro.ElasticMaterialParameter.S_WAVE_VELOCITY)
    fields = {P: vp, S: vs}
    adjoint = SimpleNamespace(
        controls=[vp, vs], control_parameter_names=[P, S],
        create_partial_reduced_functional=lambda obj, names: ReducedFunctional(
            obj, [Control(fields[name]) for name in names]),
    )
    driver = SimpleNamespace(
        get_functional=lambda: None,
        wave=SimpleNamespace(automated_adjoint=adjoint, functional_value=objective,
                             comm=SimpleNamespace(comm=space.mesh().comm)),
        _record_iterate=lambda *args: None,
        _shrink_bounds=spyro.FullWaveformInversion._shrink_bounds,
        _minimize=None,
    )
    driver._minimize = partial(spyro.FullWaveformInversion._minimize, driver)
    observed = []

    def observe(functional: object, **kwargs: object) -> list:
        """Capture the penalty weights and return the initial iterate.

        Parameters
        ----------
        functional : object
            Functional supplied to the minimizer.
        **kwargs : object
            Bounds, options and callbacks supplied by the dispatcher.

        Returns
        -------
        list
            Starting values in the minimizer's control coordinates.
        """
        term = functional._functional if isinstance(functional, LatentReducedFunctional) else functional
        observed.append(term._scales)
        return [control.tape_value().copy(deepcopy=True) for control in functional.controls]

    monkeypatch.setattr(optimization, "minimize_with_tao", observe)
    settings = spyro.FullWaveformInversion._proximal(
        {"kind": kind, "scales": [2.0, 7.0], "outer_iterations": 1})
    spyro.FullWaveformInversion._run_fwi_tao(
        driver, {"vmin": [0.0, 0.0], "vmax": [1.0, 1.0]},
        stages=[({S}, 5), ({P}, 5)], latent=latent, proximal=settings,
    )
    assert observed == [[7.0], [2.0]]
    assert settings["scales"] == [2.0, 7.0]


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
