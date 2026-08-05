"""Test time series interpolation utilities (NumPy docstring style).

This module contains tests for the `interpolate_time_series` function
from `spyro.io.time_io`. The tests verify interpolation behaviour when
the target time step is twice the source time step (half the temporal
resolution).
"""

import numpy as np
from spyro.io.time_io import interpolate_time_series


def test_interpolate_time_series_three_receivers_half_dt():
    """Interpolate a 3-receiver time series to a coarser temporal grid.

    Verifies that linearly-varying signals for three receivers are correctly
    interpolated when the `target_dt` is twice the `source_dt`. Uses a simple
    analytic expression so expected values are known exactly.

    Notes
    -----
    The source time vector is generated with `np.linspace` between
    `initial_time` and `final_time` with spacing `source_dt`. The
    interpolation should produce values matching the analytic expressions
    evaluated on the coarser time grid.
    """
    initial_time = 0.0
    final_time = 4.0
    source_dt = 0.5
    num_dt = int(np.round((final_time-initial_time) / source_dt) + 1)
    target_dt = source_dt * 2.0

    time_vector = np.linspace(initial_time, final_time, num_dt)

    values = np.column_stack(
        [
            2.0 * time_vector + 1.0,
            -time_vector + 3.0,
            0.5 * time_vector - 2.0,
        ]
    )

    interpolated = interpolate_time_series(
        values,
        target_dt=target_dt,
        initial_time=initial_time,
        final_time=final_time,
    )

    target_num_dt = int(np.round((final_time-initial_time) / target_dt) + 1)
    target_time_vector = np.linspace(initial_time, final_time, target_num_dt)
    expected = np.column_stack(
        [
            2.0 * target_time_vector + 1.0,
            -target_time_vector + 3.0,
            0.5 * target_time_vector - 2.0,
        ]
    )

    assert interpolated.shape == expected.shape
    assert np.allclose(interpolated, expected)


if __name__ == "__main__":
    test_interpolate_time_series_three_receivers_half_dt()
