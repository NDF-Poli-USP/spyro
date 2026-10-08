"""The lumped L2 change of variables of the reduced functional.

``LumpedL2ReducedFunctional`` replaces the controls ``m`` by
``m_tilde = M_L^{1/2} m``, in which the lumped L2 inner product is the Euclidean
one. With ``latent`` it uses ``psi_tilde = M_L^{1/2} psi`` instead, with
the latent control ``psi`` of ``m = lower + (upper - lower) * sigmoid(psi)``.
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


# LumpedL2ReducedFunctional needs pyadjoint's AbstractReducedFunctional, which
# older Firedrake releases do not ship; it is imported inside each test so
# that collecting this module does not fail there.
pytestmark = pytest.mark.newer_firedrake

LOWER, UPPER = 1.1, 1.6


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


@pytest.mark.parametrize("latent", [False, True], ids=["physical", "latent"])
def test_lumped_functional_matches_the_model_functional(latent):
    """The lumped functional is the same problem, written in m_tilde or psi_tilde.

    Checks that it gives the same value as the original functional, that
    ``map_result`` brings m_tilde or psi_tilde back to m, and that its
    derivative passes a Taylor test.
    """
    from spyro.reduced_functionals import LumpedL2ReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    m = fire.Function(space).assign(1.2)
    reduced_functional = lumped_misfit(m, reference(space), power=2)
    lumped_functional = LumpedL2ReducedFunctional(
        reduced_functional, bounds=[(LOWER, UPPER)], latent=latent,
    )
    start = lumped_functional.controls[0].control

    assert np.isclose(float(lumped_functional(start)), float(reduced_functional(m)))
    (m_back,) = lumped_functional.map_result(start)
    assert np.allclose(m_back.dat.data_ro, m.dat.data_ro)

    direction = fire.Function(space).interpolate(
        fire.sin(3 * fire.SpatialCoordinate(space.mesh())[0])
    )
    assert taylor_test(lumped_functional, start, direction) > 1.9


def test_lumped_functional_starts_from_the_tape_value():
    """Between stages the controls move through Control.update.

    That changes the tape value of a control but not its Function, so the
    lumped functional has to start from the tape value.
    """
    from spyro.reduced_functionals import LumpedL2ReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    m = fire.Function(space).assign(1.0)
    reduced_functional = lumped_misfit(m, reference(space))
    reduced_functional.controls[0].update(fire.Function(space).assign(2.0))
    lumped_functional = LumpedL2ReducedFunctional(reduced_functional)
    (start,) = lumped_functional.map_result(lumped_functional.controls[0].control)
    assert np.allclose(start.dat.data_ro, 2.0)


@pytest.mark.parametrize("method, degree", [
    ("CG_triangle", 1), ("CG_triangle", 2), ("DG_triangle", 1), ("DQ", 2),
])
def test_non_diagonal_mass_spaces_are_rejected(method, degree):
    """Spaces whose mass, with spyro's quadrature, is not diagonal fail."""
    from spyro.reduced_functionals import LumpedL2ReducedFunctional

    space = _space(method, degree)
    reduced_functional = lumped_misfit(
        fire.Function(space).assign(1.0), fire.Function(space),
    )
    with pytest.raises(ValueError, match="is diagonal"):
        LumpedL2ReducedFunctional(reduced_functional)


def test_latent_optimization_stays_within_the_bounds():
    """Without bounds on the latent controls, the model never leaves [lower, upper].

    The target lies below the lower bound in most of the domain, so the
    latent control goes to minus infinity there and the model reaches the
    bound in floating point.
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
