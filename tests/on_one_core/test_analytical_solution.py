from types import SimpleNamespace

import numpy as np
import pytest
import spyro

from spyro.io.basicio import parallel_print as pprint
from spyro.receivers.Receivers import RECEIVER_STEP_OFFSET
from spyro.tools.error_measure import MeasureError
from spyro.utils.analytical_solution_nodal import (
    analytical_explosive_source,
    analytical_force_source_2d,
    analytical_force_source_3d,
)


@pytest.mark.parametrize("use_vertex_only_mesh", [False, True])
def test_analytical_solution(use_vertex_only_mesh):
    frequency = 5.0
    offset = 0.5
    c_value = 1.5
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "status": False,
        "abc_type": None,
        "exponent": None,
        "cmax": None,
        "R": None,
        "pad_length": None,
    }
    dictionary["mesh"] = {
        "length_z": 3.0,  # depth in km - always positive
        "length_x": 3.0,  # width in km - always positive
    }
    dictionary["acquisition"] = {
        "delay_type": "time",
        "frequency": frequency,
        "delay": c_value / frequency,
        "source_locations": [(-1.5, 1.5)],
        "receiver_locations": [(-1.5 - offset, 1.5)],
        "use_vertex_only_mesh": use_vertex_only_mesh,
    }
    wave = spyro.examples.Rectangle_acoustic(dictionary=dictionary, periodic=True)
    wave.set_initial_velocity_model(constant=c_value)
    analytical_p = spyro.utils.nodal_homogeneous_analytical(wave, offset, c_value)

    wave.forward_solve()
    numerical_p = wave.forward_solution_receivers
    numerical_p = numerical_p.flatten()

    # Computing errors
    peak_error = MeasureError.calculate_peak_error(numerical_p, analytical_p)[0]
    integral_error = MeasureError.calculate_integral_error(
        numerical_p, analytical_p, wave.dt
    )
    normalized_l2_error = MeasureError.calculate_normalized_L2_error(
        numerical_p, analytical_p
    )

    vom_label = "VOM" if use_vertex_only_mesh else "NO VOM"
    pprint(
        f"Normalized L2 Error ({vom_label}) = {normalized_l2_error:.4e}", comm=wave.comm
    )
    pprint(f"Integral Error ({vom_label}) = {integral_error:.4e}", comm=wave.comm)
    pprint(f"Peak Error ({vom_label}) = {peak_error:.4e}", comm=wave.comm)

    assert normalized_l2_error < 1e-3 and integral_error < 1e-3 and peak_error < 1e-3


def test_analytical_force_source_3d_offset_symmetry():
    """Tests analytical force source.

    It should have the same result with the same abs distance from
    the source.
    """
    time_vector = np.linspace(0.0, 3.0, 101)

    common_parameters = dict(
        time_vector=time_vector,
        p_wave_velocity=1.5,
        s_wave_velocity=1.0,
        density=0.1,
        amplitude=1.0,
        frequency=5.0,
        time_delay=0.2,
        force_direction=0,
        displacement_direction=0,
    )

    result_positive = analytical_force_source_3d(
        offsets=np.array([1.0, 0.5, 0.3]),
        **common_parameters,
    )
    assert np.max(np.abs(result_positive)) > 1e-12

    result_negative = analytical_force_source_3d(
        offsets=np.array([-1.0, -0.5, -0.3]),
        **common_parameters,
    )

    np.testing.assert_allclose(
        result_negative,
        result_positive,
    )


def test_analytical_explosive_source_direction():
    """Tests if explosive source is zero outsie of the main direction."""
    offsets = np.array([1.0, 0.0, 0.0])
    time_vector = np.linspace(0.0, 2.0, 201)

    common_parameters = dict(
        offsets=offsets,
        time_vector=time_vector,
        p_wave_velocity=1.0,
        density=1.0,
        amplitude=1.0,
        frequency=5.0,
        time_delay=0.2,
    )

    ux = analytical_explosive_source(
        displacement_direction=0,
        **common_parameters,
    )

    uy = analytical_explosive_source(
        displacement_direction=1,
        **common_parameters,
    )

    uz = analytical_explosive_source(
        displacement_direction=2,
        **common_parameters,
    )

    assert np.any(np.abs(ux) > 1e-5)

    np.testing.assert_allclose(uy, 0.0)
    np.testing.assert_allclose(uz, 0.0)


@pytest.mark.parametrize(
    "delay, delay_type, expected",
    [
        (1.5, "multiples_of_minimum", 1.5 * np.sqrt(6.0) / (np.pi * 5.0)),
        (0.3, "time", 0.3),
    ],
)
def test_ricker_time_delay(delay: float, delay_type: str, expected: float) -> None:
    """A delay in multiples of the minima distance is converted to seconds.

    Parameters
    ----------
    delay : float
        Delay in the units of `delay_type`.
    delay_type : str
        Unit of `delay`.
    expected : float
        Delay in seconds.
    """
    assert np.isclose(spyro.sources.ricker_time_delay(delay, 5.0, delay_type), expected)


def test_ricker_time_delay_rejects_unknown_type() -> None:
    """An unknown delay type is rejected instead of left undefined."""
    with pytest.raises(ValueError, match="delay_type"):
        spyro.sources.ricker_time_delay(1.5, 5.0, "cycles")


def test_nodal_analytical_shift_is_one_time_step_for_any_delay_type() -> None:
    """Both delay types advance the wavelet by exactly one time step.

    The same delay, in multiples of the minima distance or in seconds, gives
    the same receiver data. The one-step shift was once subtracted from the
    delay before converting it, which shifted multiples by about 0.16 dt.
    """
    frequency = 5.0
    multiples = 1.5
    common = {"dt": 1e-3, "final_time": 1.0, "frequency": frequency}
    wave_multiples = SimpleNamespace(
        delay=multiples, delay_type="multiples_of_minimum", **common
    )
    wave_time = SimpleNamespace(
        delay=multiples * np.sqrt(6.0) / (np.pi * frequency),
        delay_type="time",
        **common,
    )

    from_multiples = spyro.utils.nodal_homogeneous_analytical(
        wave_multiples, 0.5, 1.5, n_extra=20
    )
    from_time = spyro.utils.nodal_homogeneous_analytical(
        wave_time, 0.5, 1.5, n_extra=20
    )

    assert np.any(np.abs(from_time) > 1e-3)
    np.testing.assert_allclose(from_multiples, from_time, rtol=0.0, atol=1e-12)


def make_analytical_wave() -> SimpleNamespace:
    """Build a stand-in wave for the nodal analytical solution.

    The delay is long enough for the wavelet to vanish at time 0, so a
    shifted wavelet is a shifted copy of the unshifted one. With the default
    delay of 1.5, the cut wavelet start changes the solution by about 1e-5
    of its peak.

    Returns
    -------
    SimpleNamespace
        Time axis and Ricker settings, with a delay in multiples.
    """
    return SimpleNamespace(
        dt=1e-3,
        final_time=1.0,
        frequency=5.0,
        delay=3.0,
        delay_type="multiples_of_minimum",
    )


def test_nodal_analytical_default_shift_is_receiver_offset() -> None:
    """With no time shift, the solution leads by the receiver offset."""
    wave = make_analytical_wave()

    default = spyro.utils.nodal_homogeneous_analytical(wave, 0.5, 1.5, n_extra=20)
    explicit = spyro.utils.nodal_homogeneous_analytical(
        wave, 0.5, 1.5, n_extra=20, time_shift=RECEIVER_STEP_OFFSET * wave.dt
    )

    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("steps", [1, 2])
def test_nodal_analytical_time_shift_advances_samples(steps: int) -> None:
    """A time shift of whole time steps moves the samples by as many indices.

    Parameters
    ----------
    steps : int
        Time shift, in time steps.
    """
    wave = make_analytical_wave()

    unshifted = spyro.utils.nodal_homogeneous_analytical(
        wave, 0.5, 1.5, n_extra=20, time_shift=0.0
    )
    shifted = spyro.utils.nodal_homogeneous_analytical(
        wave, 0.5, 1.5, n_extra=20, time_shift=steps * wave.dt
    )

    assert np.any(np.abs(unshifted) > 1e-3)
    np.testing.assert_allclose(
        shifted[:-steps], unshifted[steps:], rtol=0.0, atol=1e-12
    )


def force_source_2d(**kwargs) -> np.ndarray:
    """Compute a 2D point-force displacement with small, fixed settings.

    The time axis holds the S-wave arrival, and the extension factor is
    small to keep the test fast.

    Parameters
    ----------
    **kwargs
        Keyword arguments of :func:`analytical_force_source_2d`, such as
        `time_shift`.

    Returns
    -------
    np.ndarray
        The horizontal displacement under a horizontal force.
    """
    return analytical_force_source_2d(
        offsets=np.array([-1.0, 0.5]),
        time_vector=np.arange(301) * 2e-3,
        vp=5.6,
        vs=2.75,
        rho=2.0,
        amplitude=1.0,
        frequency=10.0,
        time_delay=0.1,
        force_direction=0,
        displacement_direction=0,
        n_extra=4,
        **kwargs,
    )


def test_force_source_2d_default_shift_is_receiver_offset() -> None:
    """With no time shift, the 2D elastic solution leads by the receiver offset."""
    default = force_source_2d()
    explicit = force_source_2d(time_shift=RECEIVER_STEP_OFFSET * 2e-3)

    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("steps", [1, 2])
def test_force_source_2d_time_shift_advances_samples(steps: int) -> None:
    """A 2D elastic shift of whole time steps moves the samples as many indices.

    Parameters
    ----------
    steps : int
        Time shift, in time steps.
    """
    unshifted = force_source_2d(time_shift=0.0)
    shifted = force_source_2d(time_shift=steps * 2e-3)

    peak = np.max(np.abs(unshifted))
    assert peak > 0.0
    np.testing.assert_allclose(
        shifted[:-steps], unshifted[steps:], rtol=0.0, atol=1e-12 * peak
    )


@pytest.mark.parametrize("source_type", ["force_source", "explosive_source"])
def test_elastic_3d_rejects_time_shift(source_type: str) -> None:
    """The 3D elastic solutions hold no receiver offset, so none is accepted.

    Parameters
    ----------
    source_type : str
        Type of source.
    """
    with pytest.raises(ValueError, match="time_shift"):
        spyro.utils.analytical_solution_elastic(
            source_type,
            np.array([1.0, 0.5, 0.3]),
            p_wave_velocity=1.5,
            s_wave_velocity=1.0,
            density=1.0,
            amplitude=1.0,
            frequency=5.0,
            time_delay=0.2,
            final_time=1.0,
            dt=1e-2,
            force_direction=0,
            time_shift=0.0,
        )


if __name__ == "__main__":
    test_analytical_solution(False)
    test_analytical_solution(True)
    test_analytical_force_source_3d_offset_symmetry()
    test_analytical_explosive_source_direction()
