"""Utilities for calculating the frequency response of a signal."""

from numpy import abs, concatenate, linspace, zeros
from scipy.fft import fft


def freq_response(signal, f_Nyq, fpad=4, get_dominant_freq=False):
    """Calculate the response in frequency domain of a time signal via FFT.

    Parameters
    ----------
    signal : `array`
        Signal data
    f_Nyq : `float`
        Nyquist frequency according to the time step. f_Nyq = 1 / (2 * dt)
    fpad : `int`, optional
        Padding factor for FFT. Default is 4
    get_dominant_freq : `bool`, optional
        If True, return only the dominant frequency of the spectrum. Default is False

    Returns
    -------
    norm_magnitude : `array`
        Normalized frequency spectrum with respect to the maximum magnitude
    dominant_freq : `float`, optional
        Dominant frequency of the spectrum
    """

    # Check if the signal is empty
    if signal.size == 0:
        raise ValueError("Input signal is empty. Cannot compute frequency response.")

    # Check if the Nyquist frequency is positive
    if f_Nyq <= 0:
        raise ValueError("Nyquist frequency is invalid. "
                         "Cannot compute frequency response.")

    # Zero padding for increasing smoothing in FFT
    signal_with_padding = concatenate([zeros(fpad * len(signal)), signal])

    # Number of sample points
    N_samples = len(signal_with_padding)

    # Determine the number of samples of the spectrum
    samples_fft = N_samples // 2 + N_samples % 2

    # Calculate the response in frequency domain of the signal (FFT)
    norm_magnitude = abs(fft(signal_with_padding)[0:samples_fft])
    del signal_with_padding

    # Frequency vector
    xf = linspace(0.0, f_Nyq, samples_fft)

    # Get the Dominant frequency of the spectrum
    dominant_freq = xf[norm_magnitude.argmax()]

    if get_dominant_freq:

        # Return the Dominant frequency only
        return dominant_freq
    else:

        # Normalized frequency spectrum
        norm_magnitude *= (1 / norm_magnitude.max())

        # Return the normalized spectrum
        return norm_magnitude
