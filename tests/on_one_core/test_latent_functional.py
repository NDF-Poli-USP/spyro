"""The latent map, as a reduced functional.

``LatentReducedFunctional`` optimizes over ``psi``, with the model
``m = lower + (upper - lower) * sigmoid(psi)`` inside its bounds.
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
    from spyro.tools.optimization import minimize_with_tao

    space = _space("mass_lumped_triangle", 2)
    target = reference(space)
    m = fire.Function(space).assign(1.3)
    (model,) = minimize_with_tao(
        lumped_misfit(m, target), bounds=[(LOWER, UPPER)],
        options={"tao_max_it": 50}, latent=True,
    )

    values = model.dat.data_ro
    assert values.min() >= LOWER and values.max() <= UPPER
    interior = (target.dat.data_ro > LOWER + 0.05) & (target.dat.data_ro < UPPER - 0.05)
    assert np.allclose(values[interior], target.dat.data_ro[interior], atol=1e-3)
