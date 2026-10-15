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


# LumpedL2ReducedFunctional needs pyadjoint's AbstractReducedFunctional, which
# older Firedrake releases do not ship; it is imported inside each test so
# that collecting this module does not fail there.
pytestmark = pytest.mark.newer_firedrake


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


def test_lumped_functional_matches_the_model_functional():
    """The lumped functional is the same problem, written in m_tilde.

    Checks that it gives the same value as the original functional, that
    ``map_result`` brings m_tilde back to m, and that its derivative passes
    a Taylor test.
    """
    from spyro.reduced_functionals import LumpedL2ReducedFunctional

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


def _scale_field(space: fire.FunctionSpace) -> fire.Function:
    """Return a positive scale that varies over the domain.

    Parameters
    ----------
    space : firedrake.FunctionSpace
        Space of the control.

    Returns
    -------
    firedrake.Function
        Values between 0.5 and 1.5.
    """
    x, _ = fire.SpatialCoordinate(space.mesh())
    return fire.Function(space).interpolate(0.5 + x)


@pytest.mark.parametrize("scale", ["number", "field"])
def test_scaled_functional_matches_the_model_functional(scale):
    """With a scale, the same problem: value, the way back to m, and a Taylor test.

    Parameters
    ----------
    scale : str
        Whether the scale is a number or a field.
    """
    from spyro.reduced_functionals import LumpedL2ReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    m = fire.Function(space).assign(1.2)
    reduced_functional = lumped_misfit(m, reference(space), power=2)
    s = 0.6 if scale == "number" else _scale_field(space)
    lumped_functional = LumpedL2ReducedFunctional(reduced_functional, scales=[s])
    m_tilde = lumped_functional.controls[0].control
    plain = LumpedL2ReducedFunctional(reduced_functional).controls[0].control
    s_values = s if scale == "number" else s.dat.data_ro
    assert np.allclose(m_tilde.dat.data_ro, plain.dat.data_ro / s_values)
    assert np.isclose(float(lumped_functional(m_tilde)), float(reduced_functional(m)))
    (m_back,) = lumped_functional.map_result(m_tilde)
    assert np.allclose(m_back.dat.data_ro, m.dat.data_ro)
    direction = fire.Function(space).interpolate(
        fire.sin(3 * fire.SpatialCoordinate(space.mesh())[0])
    )
    assert taylor_test(lumped_functional, m_tilde, direction) > 1.9


@pytest.mark.parametrize("scales", [[0.0], [-1.0], [np.inf], [1.0, 1.0]])
def test_invalid_scales_are_rejected(scales):
    """A scale is finite and positive, one per control.

    Parameters
    ----------
    scales : list
        Invalid scales.
    """
    from spyro.reduced_functionals import LumpedL2ReducedFunctional

    space = _space("mass_lumped_triangle", 2)
    reduced_functional = lumped_misfit(fire.Function(space).assign(1.2), reference(space))
    with pytest.raises(ValueError):
        LumpedL2ReducedFunctional(reduced_functional, scales=scales)


TIGHT = {"tao_gatol": 1e-12, "tao_grtol": 0.0, "tao_gttol": 0.0, "tao_max_it": 100}


def test_normalized_minimization_reaches_the_same_solution():
    """Measured in units of the box or not, BQNLS finds the projection of the target."""
    from spyro.tools.optimization import minimize_with_tao

    lower, upper = 1.1, 1.6
    solutions = []
    for normalized in (False, True):
        space = _space("mass_lumped_triangle", 2)
        m = fire.Function(space, name="vp").assign(1.3)
        reduced_functional = lumped_misfit(m, reference(space))
        (solution,) = minimize_with_tao(reduced_functional, bounds=[(lower, upper)],
                                        options=TIGHT, normalized=normalized)
        assert solution.name() == "vp"
        solutions.append(solution.dat.data_ro.copy())
    expected = np.clip(reference(space).dat.data_ro, lower, upper)
    # The target is below the box at most nodes: the lower bound is active.
    assert np.any(expected == lower)
    for values in solutions:
        assert np.allclose(values, expected, atol=1e-6)


def test_normalized_keeps_degrees_of_freedom_without_width_fixed():
    """Where the box has no width, the scale is 1 and the bounds hold the control."""
    from spyro.tools.optimization import _box_widths, minimize_with_tao

    space = _space("mass_lumped_triangle", 2)
    m = fire.Function(space, name="vp").assign(1.3)
    reduced_functional = lumped_misfit(m, reference(space))
    x, _ = fire.SpatialCoordinate(space.mesh())
    upper = fire.Function(space).interpolate(fire.conditional(x < 0.5, 1.3, 1.8))
    (width,) = _box_widths([(1.3, upper)], reduced_functional.controls)
    fixed = upper.dat.data_ro == 1.3
    assert np.any(fixed) and np.all(width.dat.data_ro[fixed] == 1.0)
    assert np.allclose(width.dat.data_ro[~fixed], 0.5)
    (solution,) = minimize_with_tao(reduced_functional, bounds=[(1.3, upper)],
                                    options=TIGHT, normalized=True)
    assert np.allclose(solution.dat.data_ro[fixed], 1.3)


@pytest.mark.parametrize("bounds", [None, [(None, 1.6)], [(1.6, 1.1)], [(1.1, np.inf)]])
def test_normalized_needs_finite_ordered_bounds(bounds):
    """Without a finite box there is no width to measure the control by.

    Parameters
    ----------
    bounds : list or None
        Invalid bounds.
    """
    from spyro.tools.optimization import minimize_with_tao

    space = _space("mass_lumped_triangle", 2)
    reduced_functional = lumped_misfit(fire.Function(space).assign(1.3), reference(space))
    with pytest.raises(ValueError):
        minimize_with_tao(reduced_functional, bounds=bounds, options=TIGHT,
                          normalized=True)
