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
    "final_time, dt", [(1.0, 0.3), (1.0, 0.00056), (0.7, 0.41)]
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


@pytest.mark.parametrize("final_time, dt", [(0.3, 1e-5), (0.9, 0.3), (0.013, 5e-4)])
def test_time_axis_keeps_exact_final_time(final_time: float, dt: float) -> None:
    """Keep a `final_time` that is already a multiple of `dt` unchanged.

    For these pairs `(nt - 1) * dt` differs from `final_time` by one ulp,
    so snapping would move a value the user gave exactly.

    Parameters
    ----------
    final_time : float
        Requested final time, a multiple of `dt`.
    dt : float
        Timestep.
    """
    axis = _TimeAxis({"final_time": final_time, "dt": dt})
    nt = axis.number_of_timesteps

    # The chosen pairs must reproduce the one-ulp shift this test guards.
    assert (nt - 1) * dt != final_time
    assert nt == round(final_time / dt) + 1
    assert axis.final_time == final_time
    assert axis.time_vector[-1] == final_time


def test_time_axis_without_dt() -> None:
    """Leave the time levels undefined while `dt` is not set."""
    axis = _TimeAxis({"final_time": 1.0})

    assert axis.final_time == 1.0
    assert axis.number_of_timesteps is None
    assert axis.time_vector is None


@pytest.mark.parametrize("new_final_time", [2.0, 0.95])
def test_time_axis_follows_final_time(new_final_time: float) -> None:
    """Re-derive the time discretization when `final_time` is reset.

    The axis starts at `final_time = 0.9` with `dt = 0.3`. A value far from
    it (2.0) must extend the time levels, and one within a timestep of it
    (0.95) must snap back onto it without recursing between the setters.

    Parameters
    ----------
    new_final_time : float
        Final time set after the axis is built.
    """
    dt = 0.3
    axis = _TimeAxis({"final_time": 0.9, "dt": dt})
    axis.final_time = new_final_time
    nt = axis.number_of_timesteps

    assert nt == round(new_final_time / dt) + 1
    assert abs(axis.final_time - new_final_time) <= dt / 2
    assert len(axis.time_vector) == nt
    assert axis.time_vector[-1] == axis.final_time


def test_time_axis_snaps_proposed_final_time_on_dt_change() -> None:
    """Snap the proposed `final_time` again, not the snapped one, on new `dt`.

    `final_time = 1.0` with `dt = 0.3` snaps to 0.9. Setting `dt = 0.2`
    must then end at 1.0, as if it had been given first. Snapping the
    already-snapped 0.9 would give `round(4.5) + 1 = 5` time levels ending
    at 0.8.
    """
    axis = _TimeAxis({"final_time": 1.0, "dt": 0.3})
    assert np.isclose(axis.final_time, 0.9, rtol=0.0, atol=1e-15)

    axis.dt = 0.2

    assert axis.number_of_timesteps == 6
    assert axis.final_time == 1.0
    assert axis.time_vector[-1] == 1.0


def _small_acoustic_wave(final_time: float, dt: float) -> spyro.AcousticWave:
    """Build a small 2D acoustic wave with one source and three receivers.

    Parameters
    ----------
    final_time : float
        Final time of the simulation.
    dt : float
        Timestep.

    Returns
    -------
    spyro.AcousticWave
        Wave with its mesh and a constant velocity model of 1.5 set.
    """
    dictionary = {
        "options": {"cell_type": "T", "variant": "lumped", "degree": 4, "dimension": 2},
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
            "final_time": final_time,
            "dt": dt,
            "amplitude": 1,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {"forward_output": False, "debug_output": False},
    }
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=1.5)
    return wave


def test_forward_receivers_match_time_vector() -> None:
    """Record one receiver row per entry of `time_vector`.

    `final_time = 0.0123` is not a multiple of `dt = 5e-4`, so it is snapped
    to 0.0125 and the solver must stop exactly there.
    """
    wave = _small_acoustic_wave(final_time=0.0123, dt=5e-4)
    wave.forward_solve()

    time_vector = wave.time_vector
    assert len(time_vector) == 26
    assert wave.forward_solution_receivers.shape[0] == len(time_vector)
    assert wave.current_time == time_vector[-1] == wave.final_time


def test_wave_propagator_snaps_final_time_to_new_dt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snap a `final_time` passed with a new `dt` onto that `dt` only.

    `wave_propagator` sets `final_time` before `dt`, and with the old
    `dt = 5e-4` it snaps 0.0124 to 0.0125. The new `dt = 1e-4` must then
    give 125 time levels ending at the proposed 0.0124, not the 126 of the
    value snapped to the old timestep. The source wavelet must be rebuilt
    to cover every time level.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the time integrator, since only the time axis is checked.
    """
    monkeypatch.setattr(
        "spyro.solvers.wave._forward_time_integrator",
        lambda wave, source_nums: None,
    )
    wave = _small_acoustic_wave(final_time=0.0123, dt=5e-4)
    wave.wave_propagator(dt=1e-4, final_time=0.0124)

    assert wave.number_of_timesteps == 125
    assert wave.final_time == 0.0124
    assert len(wave.sources.wavelet) == wave.number_of_timesteps


def test_wave_propagator_extends_final_time() -> None:
    """Propagate up to a later `final_time` given to `wave_propagator`.

    Changing only `final_time` must rebuild the time levels and the source
    wavelet, so the integrator runs and records every new time level.
    """
    wave = _small_acoustic_wave(final_time=0.0123, dt=5e-4)
    wave.forward_solve()
    wave.wave_propagator(final_time=0.02)

    nt = wave.number_of_timesteps
    assert nt == 41
    assert len(wave.sources.wavelet) == nt
    assert wave.forward_solution_receivers.shape[0] == nt
    assert np.isclose(wave.current_time, wave.final_time, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    test_interpolate_time_series_three_receivers_half_dt()
    test_interpolate_time_series_displacement_shape()
    test_interpolate_time_series_time_shift()
    test_interpolate_time_series_source_times()
    test_time_axis_without_dt()
