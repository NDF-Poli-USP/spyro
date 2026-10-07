"""The lumped L2 change of variables of the reduced functional.

``LumpedL2ReducedFunctional`` replaces the controls ``m`` by
``m_tilde = M_L^{1/2} m``, in which the lumped L2 inner product is the Euclidean
one.
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
from spyro.domains.space import create_function_space
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


def _space(method: str, degree: int) -> fire.FunctionSpace:
    """Return a spyro function space on a small mesh of the method's cells.

    Parameters
    ----------
    method : str
        Spyro method name, see ``spyro.domains.space.create_function_space``.
    degree : int
        Polynomial degree.

    Returns
    -------
    firedrake.FunctionSpace
        The space.
    """
    quadrilateral = "quadrilateral" in method or method == "DQ"
    mesh = fire.UnitSquareMesh(3, 3, quadrilateral=quadrilateral)
    return create_function_space(mesh, method, degree)


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


def lumped_misfit(control: fire.Function, target: fire.Function,
                  power: int = 1) -> ReducedFunctional:
    r"""Return :math:`\frac12 \|m^p - t\|^2` in the lumped metric.

    Parameters
    ----------
    control : firedrake.Function
        The control :math:`m`.
    target : firedrake.Function
        The target :math:`t`.
    power : int, optional
        The power :math:`p`; 1 makes this an L2 least-squares problem.

    Returns
    -------
    pyadjoint.ReducedFunctional
        The functional of the control.
    """
    quadrature, _, _ = quadrature_rules(control.function_space())
    measure = fire.dx(**quadrature) if quadrature else fire.dx
    functional = fire.assemble(0.5 * (control ** power - target) ** 2 * measure)
    return ReducedFunctional(functional, Control(control))


def test_transformed_functional_matches_the_model_functional():
    space = _space("mass_lumped_triangle", 2)
    m = fire.Function(space).assign(1.2)
    reduced_functional = lumped_misfit(m, reference(space), power=2)
    lumped_functional = LumpedL2ReducedFunctional(reduced_functional)
    m_tilde = lumped_functional.controls[0].control

    assert np.isclose(float(lumped_functional(m_tilde)), float(reduced_functional(m)))
    (m_back,) = lumped_functional.map_result(m_tilde)
    assert np.allclose(m_back.dat.data_ro, m.dat.data_ro)

    direction = fire.Function(space).interpolate(
        fire.sin(3 * fire.SpatialCoordinate(space.mesh())[0])
    )
    assert taylor_test(lumped_functional, m_tilde, direction) > 1.9


@pytest.mark.parametrize("method, degree", [
    ("CG_triangle", 1), ("CG_triangle", 2), ("DG_triangle", 1), ("DQ", 2),
])
def test_non_diagonal_mass_spaces_are_rejected(method, degree):
    """Spaces whose mass, with spyro's quadrature, is not diagonal fail."""
    space = _space(method, degree)
    reduced_functional = lumped_misfit(
        fire.Function(space).assign(1.0), fire.Function(space),
    )
    with pytest.raises(ValueError, match="is diagonal"):
        LumpedL2ReducedFunctional(reduced_functional)
