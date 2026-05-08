# This file contains methods for calculating the frequency response of a signal

import numpy as np
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
    yf : `array`
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
    yt = np.concatenate([np.zeros(fpad * len(signal)), signal])

    # Number of sample points
    N_samples = len(yt)

    # Determine the number of samples of the spectrum
    pfft = N_samples // 2 + N_samples % 2

    # Calculate the response in frequency domain of the signal (FFT)
    yf = np.abs(fft(yt)[0:pfft])
    del yt

    # Frequency vector
    xf = np.linspace(0.0, f_Nyq, pfft)

    # Get the Dominant frequency of the spectrum
    dominant_freq = xf[yf.argmax()]

    if get_dominant_freq:

        # Return the Dominant frequency only
        return dominant_freq
    else:

        # Normalized frequency spectrum
        yf *= (1 / yf.max())

        # Return the normalized spectrum
        return yf
