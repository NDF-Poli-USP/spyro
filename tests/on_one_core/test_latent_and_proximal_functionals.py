"""The latent map and the proximal term, as reduced functionals.

``LatentReducedFunctional`` optimizes over ``psi``, with the model
``m = lower + (upper - lower) * sigmoid(psi)`` inside its bounds.
``ProximalReducedFunctional`` adds ``(1/alpha) D(v, anchor)`` to a
functional, with ``D`` the L2 distance or the Bregman divergence of the box
entropy.
"""
import firedrake as fire
import numpy as np
import pytest
from firedrake.adjoint import taylor_test

from .test_lumped_l2_functional import (  # noqa: F401 - fresh_tape is autouse
    _space,
    fresh_tape,
    lumped_misfit,
    reference,
)

# The functionals need pyadjoint's AbstractReducedFunctional, which older
# Firedrake releases do not ship; they are imported inside each test.
pytestmark = pytest.mark.newer_firedrake

LOWER, UPPER = 1.1, 1.6


def test_latent_functional_matches_the_model_functional():
    """Same value at the starting point, a way back to m, and a Taylor test."""
    from spyro.reduced_functionals import LatentReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    m = fire.Function(space).assign(1.3)
    reduced_functional = lumped_misfit(m, reference(space), power=2)
    latent = LatentReducedFunctional(reduced_functional, [(LOWER, UPPER)])
    psi = latent.controls[0].control

    assert np.isclose(float(latent(psi)), float(reduced_functional(m)))
    (m_back,) = latent.map_result(psi)
    assert np.allclose(m_back.dat.data_ro, m.dat.data_ro)

    direction = fire.Function(space).interpolate(
        fire.sin(3 * fire.SpatialCoordinate(space.mesh())[0])
    )
    assert taylor_test(latent, psi, direction) > 1.9


def test_latent_optimization_stays_within_the_bounds():
    """Without bounds on psi, the model never leaves [lower, upper].

    The target lies below the lower bound in most of the domain, so psi goes
    to minus infinity there and the model reaches the bound in floating point.
    """
    from spyro.reduced_functionals import LatentReducedFunctional
    from spyro.tools.optimization import minimize_with_tao

    space = _space("mass_lumped_triangle", 2)
    target = reference(space)
    m = fire.Function(space).assign(1.3)
    latent = LatentReducedFunctional(lumped_misfit(m, target), [(LOWER, UPPER)])
    psi = minimize_with_tao(latent, options={"tao_max_it": 50})
    (model,) = latent.map_result(psi)

    values = model.dat.data_ro
    assert values.min() >= LOWER and values.max() <= UPPER
    interior = (target.dat.data_ro > LOWER + 0.05) & (target.dat.data_ro < UPPER - 0.05)
    assert np.allclose(values[interior], target.dat.data_ro[interior], atol=1e-3)


@pytest.mark.parametrize("kind", ["l2", "bregman"])
def test_proximal_functional_derivative(kind):
    """The proximal term vanishes at the anchor, and its derivative is right."""
    from spyro.reduced_functionals import ProximalReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    m = fire.Function(space).assign(1.3)
    reduced_functional = lumped_misfit(m, reference(space))
    proximal = ProximalReducedFunctional(
        reduced_functional, kind, step=0.5, bounds=[(LOWER, UPPER)],
    )
    assert np.isclose(float(proximal(m)), float(reduced_functional(m)))

    x = fire.SpatialCoordinate(space.mesh())[0]
    away = fire.Function(space).interpolate(1.3 + 0.1 * fire.sin(2 * x))
    direction = fire.Function(space).interpolate(fire.cos(3 * x))
    assert proximal.proximal_value(away) > 0.0
    assert taylor_test(proximal, away, direction) > 1.9


def test_l2_proximal_step_has_the_closed_form_minimizer():
    """min 1/2|v - t|^2 + 1/(2 alpha)|v - a|^2 is v = (alpha t + a)/(alpha + 1)."""
    from spyro.reduced_functionals import ProximalReducedFunctional
    from spyro.tools.optimization import minimize_with_tao

    alpha = 0.5
    space = _space("mass_lumped_triangle", 2)
    target = reference(space)
    m = fire.Function(space).assign(1.3)
    proximal = ProximalReducedFunctional(lumped_misfit(m, target), "l2", step=alpha)
    (v,) = minimize_with_tao(proximal, options={
        "tao_gatol": 1e-10, "tao_grtol": 0.0, "tao_gttol": 0.0, "tao_max_it": 20,
    })
    expected = (alpha * target.dat.data_ro + 1.3) / (alpha + 1.0)
    assert np.allclose(v.dat.data_ro, expected, rtol=0.0, atol=1e-8)
