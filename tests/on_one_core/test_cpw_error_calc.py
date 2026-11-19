"""Tests for the time alignment in `cells_per_wavelength_calculator.error_calc`."""

import numpy as np

from spyro.tools.cells_per_wavelength_calculator import error_calc


class _RecordingMeasure:
    """Stand-in for `MeasureError` that keeps the reference it receives.

    Whatever error-measure method `error_calc` calls, it stores the
    reference resampled onto the receiver times and returns zero errors.

    Attributes
    ----------
    reference : numpy.ndarray or None
        Reference passed to the last error-measure call.
    """

    def __init__(self) -> None:
        self.reference = None

    def __getattr__(self, name: str):
        """Return a recorder for any error-measure method.

        Parameters
        ----------
        name : str
            Name of the requested method.

        Returns
        -------
        callable
            Function that stores its second argument and returns zero errors.
        """

        def record(receivers, reference, *args, **kwargs):
            self.reference = reference
            return [None, None, None, 0.0]

        return record


def test_error_calc_uses_time_vectors() -> None:
    """Resample a reference with another timestep at its true sample times.

    The receivers are sampled every 0.01 up to 1.0, the reference every
    0.004 up to 1.004. Given both time vectors, the resampled reference must
    match a linear signal exactly at the receiver times, instead of being
    stretched over the receivers' time span.
    """
    dt = 0.01
    receiver_times = np.linspace(0.0, 1.0, 101)
    reference_times = np.linspace(0.0, 1.004, 252)
    receivers = np.column_stack([receiver_times, -receiver_times])
    reference = np.column_stack([2.0 * reference_times + 1.0, -reference_times + 3.0])

    measure = _RecordingMeasure()
    error_calc(
        measure,
        receivers,
        reference,
        dt,
        time_vector=receiver_times,
        analytical_time_vector=reference_times,
    )

    expected = np.column_stack([2.0 * receiver_times + 1.0, -receiver_times + 3.0])
    assert np.allclose(measure.reference, expected, rtol=0.0, atol=1e-12)
