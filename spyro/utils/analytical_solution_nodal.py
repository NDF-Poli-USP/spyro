"""Analytical solvers for nodal acoustic wavefield validation.

This module provides utilities to build reference traces for homogeneous
media, mainly for comparisons against numerical solutions.
"""

import numpy as np
from scipy.special import hankel2
from ..sources import full_ricker_wavelet


def nodal_homogeneous_analytical(Wave_object, offset, c_value, n_extra=5000):
    """Compute an analytical nodal trace for a homogeneous medium.

    The solution considers a single source-receiver pair and extends the
    source wavelet duration before truncating the final result to the original
    simulation time window.

    Parameters
    ----------
    Wave_object : spyro.Wave
        Wave object containing temporal and source parameters (`dt`,
        `final_time`, `frequency`, `delay`, and `delay_type`).
    offset : float
        Offset between source and receiver.
    c_value : float
        Velocity of the homogeneous medium.
    n_extra : int, default=5000
        Multiplicative factor used to extend ``final_time`` when generating
        the source wavelet and analytical response.

    Returns
    -------
    numpy.ndarray
        Analytical displacement/pressure time series restricted to the
        original number of time samples.
    """
    # Generating extended ricker wavelet
    dt = Wave_object.dt
    final_time = Wave_object.final_time
    num_t = int(final_time / dt + 1)

    extended_final_time = n_extra * final_time

    frequency = Wave_object.frequency
    delay = Wave_object.delay
    delay_type = Wave_object.delay_type

    ricker_wavelet = full_ricker_wavelet(
        dt=dt,
        final_time=extended_final_time,
        frequency=frequency,
        delay=delay - dt,
        delay_type=delay_type,
    )

    full_u_analytical = analytical_solution(
        ricker_wavelet, c_value, extended_final_time, offset
    )

    u_analytical = full_u_analytical[:num_t]

    return u_analytical


def analytical_solution(ricker_wavelet, c_value, final_time, offset):
    """Compute the analytical response from a source wavelet in frequency space.

    This routine applies a 2D Green's function in the Fourier domain using a
    zeroth-order Hankel function of the second kind.

    Parameters
    ----------
    ricker_wavelet : numpy.ndarray
        Time-domain source wavelet samples.
    c_value : float
        Propagation velocity of the homogeneous medium.
    final_time : float
        Total modeled time associated with ``ricker_wavelet``.
    offset : float
        Source-receiver distance.

    Returns
    -------
    numpy.ndarray
        Real-valued analytical time series with the same length as
        ``ricker_wavelet``.
    """
    num_t = len(ricker_wavelet)

    # Fourier-domain frequencies.
    nf = int(num_t / 2 + 1)
    frequency_axis = (1.0 / final_time) * np.arange(nf)

    # Fourier transform of the source wavelet (positive frequencies).
    fft_rw = np.fft.fft(ricker_wavelet)
    fft_rw = fft_rw[0:nf]

    U_a = np.zeros((nf), dtype=complex)
    for a in range(1, nf - 1):
        k = 2 * np.pi * frequency_axis[a] / c_value
        tmp = k * offset
        U_a[a] = -1j * np.pi * hankel2(0.0, tmp) * fft_rw[a]

    U_t = 1.0 / (2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], num_t))

    return np.real(U_t)
