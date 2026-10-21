"""Tests for the cells-per-wavelength line searches and threshold discretization."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from spyro.tools.cells_per_wavelength_calculator.line_search import (
    MAX_CPW,
    backward_line_search,
    forward_line_search,
)
from spyro.tools.cells_per_wavelength_calculator.threshold_discretization import (
    ThresholdDiscretization,
    _parse_threshold_discretization,
)


def make_error_function(
    critical_cpw: float,
) -> tuple[Callable[[float], float], list[float]]:
    """Build a monotonically decreasing error that crosses 1 at ``critical_cpw``.

    Parameters
    ----------
    critical_cpw : float
        Cells-per-wavelength where the error equals the threshold of 1.

    Returns
    -------
    calculate_error : Callable[[float], float]
        Error as a function of the cells-per-wavelength.
    calls : list of float
        Every value ``calculate_error`` was called with, in order.
    """
    calls = []

    def calculate_error(cpw: float) -> float:
        calls.append(cpw)
        return (critical_cpw / cpw) ** 2

    return calculate_error, calls


@pytest.mark.parametrize(
    "starting_cpw, critical_cpw, expected",
    [
        (6.0, 3.14, 3.2),
        (6.0, 3.2, 3.2),
        (3.25, 2.0, 2.0),
        (2.35, 2.31, 2.35),
        (2.35, 2.25, 2.3),
    ],
)
def test_backward_line_search_finds_smallest_passing_cpw(
    starting_cpw: float, critical_cpw: float, expected: float,
) -> None:
    """The search returns the smallest grid value at or above the crossing."""
    calculate_error, calls = make_error_function(critical_cpw)

    cpw = backward_line_search(
        calculate_error, starting_cpw, 1.0, 0.1, max_cpw=10.0
    )

    assert np.isclose(cpw, expected)
    assert len(calls) == len(set(calls)), "A cpw value was evaluated twice"


def test_backward_line_search_coarse_phase_saves_evaluations() -> None:
    """Coarse steps use far fewer solves than stepping by the accuracy."""
    calculate_error, calls = make_error_function(3.0)

    backward_line_search(calculate_error, 10.0, 1.0, 0.1, max_cpw=10.0)

    assert len(calls) < 25


def test_backward_line_search_rejects_failing_start() -> None:
    """A starting value above the threshold is an input error."""
    calculate_error, _ = make_error_function(3.0)

    with pytest.raises(ValueError, match="Start from a larger value"):
        backward_line_search(calculate_error, 2.0, 1.0, 0.1)


@pytest.mark.parametrize(
    "starting_cpw, critical_cpw, expected",
    [
        (2.0, 3.14, 3.2),
        (2.0, 3.2, 3.2),
        (1.0, 2.0, 2.0),
        (2.25, 2.31, 2.4),
        (2.25, 2.29, 2.3),
        (1.0, 4.0, 4.0),
        (4.0, 4.97, 5.0),
    ],
)
def test_forward_line_search_finds_smallest_passing_cpw(
    starting_cpw: float, critical_cpw: float, expected: float,
) -> None:
    """The search returns the smallest grid value at or above the crossing."""
    calculate_error, calls = make_error_function(critical_cpw)

    cpw = forward_line_search(calculate_error, starting_cpw, 1.0, 0.1)

    assert np.isclose(cpw, expected)
    assert len(calls) == len(set(calls)), "A cpw value was evaluated twice"


@pytest.mark.parametrize("starting_cpw", [1.0, 2.0, 2.55])
def test_forward_and_backward_line_searches_agree(starting_cpw: float) -> None:
    """Both directions find the same minimum cells-per-wavelength."""
    critical_cpw = 2.57
    calculate_error, _ = make_error_function(critical_cpw)

    forward = forward_line_search(calculate_error, starting_cpw, 1.0, 0.1)
    backward = backward_line_search(
        calculate_error, 10.0, 1.0, 0.1, max_cpw=10.0
    )

    assert np.isclose(forward, backward)


def test_forward_line_search_coarse_phase_saves_evaluations() -> None:
    """Coarse steps use far fewer solves than stepping by the accuracy."""
    calculate_error, calls = make_error_function(4.5)

    forward_line_search(calculate_error, 1.0, 1.0, 0.1)

    assert len(calls) < 25


def test_forward_line_search_rejects_passing_start() -> None:
    """A starting value within the threshold is an input error."""
    calculate_error, _ = make_error_function(3.0)

    with pytest.raises(ValueError, match="Start from a smaller value"):
        forward_line_search(calculate_error, 4.0, 1.0, 0.1)


def test_forward_line_search_stops_at_max_cpw() -> None:
    """A value above the cap is never evaluated, and failing it raises."""
    calculate_error, calls = make_error_function(6.0)

    with pytest.raises(ValueError, match="Maximum cells-per-wavelength"):
        forward_line_search(calculate_error, 1.0, 1.0, 0.1, max_cpw=5.0)

    assert max(calls) == 5.0


def test_forward_line_search_rejects_start_above_max_cpw() -> None:
    """A starting value at or above the cap is an input error."""
    calculate_error, _ = make_error_function(6.0)

    with pytest.raises(ValueError, match="not below the maximum"):
        forward_line_search(calculate_error, 5.0, 1.0, 0.1, max_cpw=5.0)


@pytest.mark.parametrize("line_search", [backward_line_search, forward_line_search])
@pytest.mark.parametrize(
    "starting_cpw, accuracy, coarse_step_fraction, name",
    [
        (0.0, 0.1, 0.1, "starting_cpw"),
        (2.0, 0.0, 0.1, "accuracy"),
        (2.0, 0.1, -0.1, "coarse_step_fraction"),
    ],
)
def test_line_searches_reject_non_positive_settings(
    line_search: Callable[..., float],
    starting_cpw: float,
    accuracy: float,
    coarse_step_fraction: float,
    name: str,
) -> None:
    """Non-positive settings are rejected before any error is evaluated."""
    calculate_error, calls = make_error_function(3.0)

    with pytest.raises(ValueError, match=f"{name} must be positive"):
        line_search(
            calculate_error, starting_cpw, 1.0, accuracy,
            coarse_step_fraction=coarse_step_fraction,
        )

    assert calls == []


def test_backward_line_search_rejects_start_above_max_cpw() -> None:
    """A starting value above the cap is rejected before any evaluation."""
    calculate_error, calls = make_error_function(3.0)

    with pytest.raises(ValueError, match="above the maximum"):
        backward_line_search(calculate_error, 6.0, 1.0, 0.1, max_cpw=5.0)

    assert calls == []


def test_backward_line_search_accepts_start_at_max_cpw() -> None:
    """The cap itself is a valid starting value."""
    calculate_error, _ = make_error_function(3.0)

    cpw = backward_line_search(calculate_error, 5.0, 1.0, 0.1, max_cpw=5.0)

    assert np.isclose(cpw, 3.0)


@pytest.mark.parametrize("critical_cpw", np.linspace(1.03, 4.97, 41))
def test_line_searches_match_exhaustive_search(critical_cpw: float) -> None:
    """Both searches return the smallest passing value an exhaustive scan finds."""
    starting_cpw = 0.95
    grid = [round(0.1 * i, 10) for i in range(10, 51)]
    calculate_error, _ = make_error_function(critical_cpw)
    expected = min(cpw for cpw in grid if calculate_error(cpw) <= 1.0)

    forward = forward_line_search(calculate_error, starting_cpw, 1.0, 0.1)
    backward = backward_line_search(calculate_error, 5.0, 1.0, 0.1)

    assert np.isclose(forward, expected)
    assert np.isclose(backward, expected)


def test_fine_phase_bisects() -> None:
    """The fine phase needs a logarithmic number of solves in its bracket."""
    calculate_error, calls = make_error_function(1.01)

    cpw = backward_line_search(
        calculate_error, 5.0, 1.0, 0.01, coarse_step_fraction=0.5
    )

    assert np.isclose(cpw, 1.01)
    # Coarse phase: 5.0, 2.5, 1.25, 0.62; then bisecting 62 grid values
    assert len(calls) <= 4 + math.ceil(math.log2(62))


def test_backward_line_search_uses_default_max_cpw() -> None:
    """Without an explicit cap, a start above MAX_CPW is rejected."""
    calculate_error, calls = make_error_function(3.0)

    with pytest.raises(ValueError, match="above the maximum"):
        backward_line_search(calculate_error, MAX_CPW + 1.0, 1.0, 0.1)

    assert calls == []


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (False, None),
        (True, ThresholdDiscretization("spectral_quadrilateral", 4, 2.0)),
        ({"cpw": 3.0}, ThresholdDiscretization("spectral_quadrilateral", 4, 3.0)),
        (
            ThresholdDiscretization("mass_lumped_triangle", 2, 5.0),
            ThresholdDiscretization("mass_lumped_triangle", 2, 5.0),
        ),
    ],
)
def test_parse_threshold_discretization(
    value: bool | dict | ThresholdDiscretization | None,
    expected: ThresholdDiscretization | None,
) -> None:
    """The threshold discretization defaults to P4 quadrilaterals at cpw 2."""
    assert _parse_threshold_discretization(value) == expected


@pytest.mark.parametrize("value", [{"order": 4}, 4.0])
def test_parse_threshold_discretization_rejects_bad_values(
    value: dict | float,
) -> None:
    """Unknown keys and unsupported types are rejected."""
    with pytest.raises(TypeError):
        _parse_threshold_discretization(value)
