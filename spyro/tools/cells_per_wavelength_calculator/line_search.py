"""Line searches for the minimum cells-per-wavelength meeting an error threshold."""

import math
from collections.abc import Callable

from ...utils.typing import CpwSearchDirection

#: Fraction of the current cpw added or removed per coarse search step.
COARSE_STEP_FRACTION = 0.1

#: Default largest cells-per-wavelength evaluated by the line searches.
MAX_CPW = 5.0


def _floor_to_accuracy(value: float, accuracy: float) -> float:
    """Floor a value onto the grid of multiples of ``accuracy``.

    Parameters
    ----------
    value : float
        Value to floor.
    accuracy : float
        Grid spacing.

    Returns
    -------
    float
        Largest multiple of ``accuracy`` not above ``value``.
    """
    # The small tolerance keeps values already on the grid in place
    return round(math.floor(value / accuracy + 1e-9) * accuracy, 10)


def _ceil_to_accuracy(value: float, accuracy: float) -> float:
    """Ceil a value onto the grid of multiples of ``accuracy``.

    Parameters
    ----------
    value : float
        Value to ceil.
    accuracy : float
        Grid spacing.

    Returns
    -------
    float
        Smallest multiple of ``accuracy`` not below ``value``.
    """
    # The small tolerance keeps values already on the grid in place
    return round(math.ceil(value / accuracy - 1e-9) * accuracy, 10)


class _CachedThresholdCheck:
    """Memoized check of whether a cells-per-wavelength meets a threshold.

    Each forward solve is expensive, so every value is evaluated at most
    once.

    Attributes
    ----------
    calculate_error : Callable[[float], float]
        Returns the error of a cells-per-wavelength value.
    threshold : float
        Largest accepted error.
    errors : dict[float, float]
        Error of every value evaluated so far.
    """

    def __init__(
        self, calculate_error: Callable[[float], float], threshold: float
    ) -> None:
        """Store the error function and the threshold.

        Parameters
        ----------
        calculate_error : Callable[[float], float]
            Returns the error of a cells-per-wavelength value.
        threshold : float
            Largest accepted error.
        """
        self.calculate_error = calculate_error
        self.threshold = threshold
        self.errors: dict[float, float] = {}

    def __call__(self, cpw: float) -> bool:
        """Check a value, evaluating its error only the first time.

        Parameters
        ----------
        cpw : float
            Cells-per-wavelength to check.

        Returns
        -------
        bool
            Whether the error of ``cpw`` meets the threshold.
        """
        if cpw not in self.errors:
            self.errors[cpw] = self.calculate_error(cpw)
        return self.errors[cpw] <= self.threshold


def _check_search_settings(
    starting_cpw: float, accuracy: float, coarse_step_fraction: float
) -> None:
    """Check the settings shared by the line searches.

    Parameters
    ----------
    starting_cpw : float
        First value evaluated.
    accuracy : float
        Resolution of the result.
    coarse_step_fraction : float
        Fraction of the current value moved per coarse step.

    Raises
    ------
    ValueError
        If any setting is not positive.
    """
    for name, value in (
        ("starting_cpw", starting_cpw),
        ("accuracy", accuracy),
        ("coarse_step_fraction", coarse_step_fraction),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")


def _bisect_grid(
    passes: _CachedThresholdCheck,
    failing: float,
    passing: float,
    accuracy: float,
) -> float:
    """Bisect the grid for the smallest value meeting the threshold.

    The error is assumed to decrease as the cells-per-wavelength grows, so
    every grid value up to ``failing`` fails and every value from
    ``passing`` on passes. Only grid values strictly between them are
    evaluated.

    Parameters
    ----------
    passes : _CachedThresholdCheck
        Threshold check of a cells-per-wavelength value.
    failing : float
        A value known to fail, or 0.0 when none is known.
    passing : float
        A value known to pass. It may be off the grid.
    accuracy : float
        Grid spacing.

    Returns
    -------
    float
        The smallest grid value in ``(failing, passing)`` that passes, or
        ``passing`` if none does.
    """
    # Grid indices: values at or below low fail, values at or above high pass
    low = math.floor(failing / accuracy + 1e-9)
    high = math.ceil(passing / accuracy - 1e-9)
    smallest_passing = passing
    while high - low > 1:
        middle = (low + high) // 2
        candidate = round(middle * accuracy, 10)
        if passes(candidate):
            high = middle
            smallest_passing = candidate
        else:
            low = middle
    return smallest_passing


def backward_line_search(
    calculate_error: Callable[[float], float],
    starting_cpw: float,
    threshold: float,
    accuracy: float,
    coarse_step_fraction: float = COARSE_STEP_FRACTION,
    max_cpw: float = MAX_CPW,
) -> float:
    """Search downward for the smallest cells-per-wavelength meeting a threshold.

    The error is assumed to decrease as the cells-per-wavelength grows.
    Starting from a passing value, a coarse phase removes
    ``coarse_step_fraction`` of the current value per step, but at least
    ``accuracy``, until the error first exceeds the threshold. A fine phase
    then bisects the grid between the last passing and the first failing
    value. Each value is evaluated at most once.

    Parameters
    ----------
    calculate_error : Callable[[float], float]
        Returns the error of a cells-per-wavelength value.
    starting_cpw : float
        First value evaluated. Its error must meet ``threshold``.
    threshold : float
        Largest accepted error.
    accuracy : float
        Resolution of the result. Candidates after the start are multiples
        of it.
    coarse_step_fraction : float, optional
        Fraction of the current value removed per coarse step. Default is
        :data:`COARSE_STEP_FRACTION`.
    max_cpw : float, optional
        Largest value evaluated. Default is :data:`MAX_CPW`.

    Returns
    -------
    float
        The smallest evaluated value whose error meets ``threshold``.

    Raises
    ------
    ValueError
        If a setting is not positive, ``starting_cpw`` is above
        ``max_cpw``, or ``starting_cpw`` does not meet ``threshold``.
    """
    _check_search_settings(starting_cpw, accuracy, coarse_step_fraction)
    if starting_cpw > max_cpw:
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} is above "
            f"the maximum {max_cpw}."
        )
    passes = _CachedThresholdCheck(calculate_error, threshold)

    if not passes(starting_cpw):
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} has error "
            f"{passes.errors[starting_cpw]} above the threshold {threshold}. "
            "Start from a larger value."
        )
    last_passing = starting_cpw
    first_failing = 0.0

    # Coarse phase: large relative steps until the first failure
    while True:
        step = max(coarse_step_fraction * last_passing, accuracy)
        candidate = _floor_to_accuracy(last_passing - step, accuracy)
        if candidate <= 0.0:
            break
        if not passes(candidate):
            first_failing = candidate
            break
        last_passing = candidate

    return _bisect_grid(passes, first_failing, last_passing, accuracy)


def forward_line_search(
    calculate_error: Callable[[float], float],
    starting_cpw: float,
    threshold: float,
    accuracy: float,
    coarse_step_fraction: float = COARSE_STEP_FRACTION,
    max_cpw: float = MAX_CPW,
) -> float:
    """Search upward for the smallest cells-per-wavelength meeting a threshold.

    The error is assumed to decrease as the cells-per-wavelength grows.
    Starting from a failing value, a coarse phase adds
    ``coarse_step_fraction`` of the current value per step, but at least
    ``accuracy``, until the error first meets the threshold. A fine phase
    then bisects the grid between the last failing and the first passing
    value. No value above ``max_cpw`` is evaluated, and each value is
    evaluated at most once.

    Parameters
    ----------
    calculate_error : Callable[[float], float]
        Returns the error of a cells-per-wavelength value.
    starting_cpw : float
        First value evaluated. Its error must exceed ``threshold``.
    threshold : float
        Largest accepted error.
    accuracy : float
        Resolution of the result. Candidates after the start are multiples
        of it.
    coarse_step_fraction : float, optional
        Fraction of the current value added per coarse step. Default is
        :data:`COARSE_STEP_FRACTION`.
    max_cpw : float, optional
        Largest value evaluated. Default is :data:`MAX_CPW`.

    Returns
    -------
    float
        The smallest evaluated value whose error meets ``threshold``.

    Raises
    ------
    ValueError
        If a setting is not positive, ``starting_cpw`` already meets
        ``threshold`` or is not below ``max_cpw``, or ``max_cpw`` does not
        meet ``threshold``.
    """
    _check_search_settings(starting_cpw, accuracy, coarse_step_fraction)
    if starting_cpw >= max_cpw:
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} is not below "
            f"the maximum {max_cpw}."
        )
    passes = _CachedThresholdCheck(calculate_error, threshold)

    if passes(starting_cpw):
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} has error "
            f"{passes.errors[starting_cpw]} within the threshold {threshold}. "
            "Start from a smaller value."
        )
    last_failing = starting_cpw

    # Coarse phase: large relative steps until the first success
    while True:
        step = max(coarse_step_fraction * last_failing, accuracy)
        candidate = min(_ceil_to_accuracy(last_failing + step, accuracy), max_cpw)
        if passes(candidate):
            first_passing = candidate
            break
        if candidate >= max_cpw:
            raise ValueError(
                f"Maximum cells-per-wavelength {max_cpw} has error "
                f"{passes.errors[candidate]} above the threshold {threshold}."
            )
        last_failing = candidate

    return _bisect_grid(passes, last_failing, first_passing, accuracy)


#: Line search run by :meth:`MeshingParameterCalculator.find_minimum`,
#: selected by the ``"search_direction"`` parameter.
LINE_SEARCHES: dict[CpwSearchDirection, Callable[..., float]] = {
    CpwSearchDirection.BACKWARD: backward_line_search,
    CpwSearchDirection.FORWARD: forward_line_search,
}
