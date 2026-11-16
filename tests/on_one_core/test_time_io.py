"""Test time series interpolation utilities (NumPy docstring style).

This module contains tests for the `interpolate_time_series` function
from `spyro.io.time_io`. The tests verify interpolation behaviour when
the target time step is twice the source time step (half the temporal
resolution).
"""

import numpy as np
import pytest

import spyro
from spyro.io.time_io import Read_time_axis, interpolate_time_series


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


def test_interpolate_time_series_displacement_shape() -> None:
    """Interpolate ``(time, receiver, direction)`` data and keep its shape.

    Each receiver and direction holds a different linear signal, so the
    expected values on the coarser grid are known exactly.
    """
    initial_time = 0.0
    final_time = 4.0
    time_vector = np.linspace(initial_time, final_time, 9)
    target_time_vector = np.linspace(initial_time, final_time, 5)

    slopes = np.arange(1.0, 7.0).reshape(3, 2)
    values = time_vector[:, None, None] * slopes[None, :, :]
    expected = target_time_vector[:, None, None] * slopes[None, :, :]

    interpolated = interpolate_time_series(
        values,
        target_dt=1.0,
        initial_time=initial_time,
        final_time=final_time,
    )

    assert interpolated.shape == (5, 3, 2)
    assert np.allclose(interpolated, expected)


def test_interpolate_time_series_time_shift() -> None:
    """Shift the target grid and clamp samples past the end of the data.

    A linear signal sampled with ``time_shift`` should match the signal
    evaluated at the shifted times, and target times beyond
    ``final_time`` should take the last data value.
    """
    initial_time = 0.0
    final_time = 4.0
    time_vector = np.linspace(initial_time, final_time, 9)
    values = 2.0 * time_vector + 1.0
    time_shift = 0.25

    interpolated = interpolate_time_series(
        values,
        target_dt=1.0,
        initial_time=initial_time,
        final_time=final_time,
        time_shift=time_shift,
    )

    shifted_times = np.minimum(np.arange(5) * 1.0 + time_shift, final_time)
    expected = 2.0 * shifted_times + 1.0

    assert interpolated.shape == (5,)
    assert np.allclose(interpolated, expected)


def test_interpolate_time_series_source_times() -> None:
    """Use the true sample times when they do not span `final_time`.

    A solver with `dt = 0.3` samples at 0.0, 0.3, 0.6 and 0.9. Passing
    those times as `source_times`, with `final_time = 1.0`, must recover the
    linear signal exactly up to the last sample, instead of spreading the
    samples over [0, 1].
    """
    final_time = 1.0
    source_times = np.arange(4) * 0.3
    values = 2.0 * source_times + 1.0

    interpolated = interpolate_time_series(
        values,
        target_dt=0.25,
        initial_time=0.0,
        final_time=final_time,
        source_times=source_times,
    )

    target_times = np.minimum(np.arange(5) * 0.25, source_times[-1])
    assert np.allclose(interpolated, 2.0 * target_times + 1.0)

    with pytest.raises(ValueError, match="source_times"):
        interpolate_time_series(
            values,
            target_dt=0.25,
            initial_time=0.0,
            final_time=final_time,
            source_times=source_times[:-1],
        )


class _TimeAxis(Read_time_axis):
    """Bare `Read_time_axis` built from a `time_axis` dictionary.

    Parameters
    ----------
    time_axis : dict
        The `time_axis` entry of a spyro input dictionary.
    """

    def __init__(self, time_axis: dict) -> None:
        self.input_dictionary = {"time_axis": time_axis}
        super().__init__()


@pytest.mark.parametrize(
    "final_time, dt", [(0.0049, 1e-4), (1.0, 0.3), (1.0, 0.0005), (0.7, 0.4)]
)
def test_time_axis_snaps_final_time(final_time: float, dt: float) -> None:
    """Snap `final_time` onto the last time level and keep the axis in sync.

    Parameters
    ----------
    final_time : float
        Requested final time, not necessarily a multiple of `dt`.
    dt : float
        Timestep.
    """
    axis = _TimeAxis({"final_time": final_time, "dt": dt})
    nt = axis.number_of_timesteps

    assert nt == round(final_time / dt) + 1
    assert axis.final_time == (nt - 1) * dt
    assert abs(axis.final_time - final_time) <= dt / 2
    assert len(axis.time_vector) == nt
    assert axis.time_vector[-1] == axis.final_time
    assert np.allclose(axis.time_vector, np.arange(nt) * dt, rtol=0.0, atol=1e-15)


def test_time_axis_without_dt() -> None:
    """Leave the time levels undefined while `dt` is not set."""
    axis = _TimeAxis({"final_time": 1.0})

    assert axis.final_time == 1.0
    assert axis.number_of_timesteps is None
    assert axis.time_vector is None


def test_forward_receivers_match_time_vector() -> None:
    """Record one receiver row per entry of `time_vector`.

    `final_time = 0.0123` is not a multiple of `dt = 5e-4`, so it is snapped
    to 0.0125 and the solver must stop exactly there.
    """
    dictionary = {
        "options": {"cell_type": "T", "variant": "lumped", "degree": 2, "dimension": 2},
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.2, 0.5)],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": spyro.create_transect((-0.8, 0.2), (-0.8, 0.8), 3),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.0123,
            "dt": 5e-4,
            "amplitude": 1,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {"forward_output": False, "debug_output": False},
    }
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=1.5)
    wave.forward_solve()

    time_vector = wave.time_vector
    assert len(time_vector) == 26
    assert wave.forward_solution_receivers.shape[0] == len(time_vector)
    assert wave.current_time == time_vector[-1] == wave.final_time


if __name__ == "__main__":
    test_interpolate_time_series_three_receivers_half_dt()
    test_interpolate_time_series_displacement_shape()
    test_interpolate_time_series_time_shift()
    test_interpolate_time_series_source_times()
    test_time_axis_without_dt()
