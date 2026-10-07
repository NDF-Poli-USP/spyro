"""The lumped L2 change of variables of the reduced functional.

``LumpedL2ReducedFunctional`` replaces the controls ``m`` by
``m_tilde = M_L^{1/2} m``, in which the lumped L2 inner product is the Euclidean
one. These tests check the transformation itself; ``test_optimization`` checks
what TAO does with it.
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
from pyadjoint import Tape

from spyro.domains.quadrature import quadrature_rules
from spyro.reduced_functionals import LumpedL2ReducedFunctional


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
    transformed = LumpedL2ReducedFunctional(reduced_functional)
    m_tilde = transformed.controls[0].control

    assert np.isclose(float(transformed(m_tilde)), float(reduced_functional(m)))
    (m_back,) = transformed.map_result(m_tilde)
    assert np.allclose(m_back.dat.data_ro, m.dat.data_ro)

    direction = fire.Function(space).interpolate(
        fire.sin(3 * fire.SpatialCoordinate(space.mesh())[0])
    )
    assert taylor_test(transformed, m_tilde, direction) > 1.9


def test_a_mass_that_does_not_lump_is_rejected():
    """Quadratic Lagrange vertex functions integrate to zero on triangles."""
    space = fire.FunctionSpace(fire.UnitSquareMesh(2, 2), "CG", 2)
    reduced_functional = lumped_misfit(
        [fire.Function(space).assign(1.0)], [fire.Function(space)],
    )
    with pytest.raises(ValueError, match="integrate to zero or less"):
        LumpedL2ReducedFunctional(reduced_functional)
