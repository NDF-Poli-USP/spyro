"""Utilities for calculating the frequency response of a signal."""

from numpy import abs, empty, hanning, mean, pad
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from ..utils.error_management import validate_data_structure, validate_numeric


def ensure_even_length(signal):
    """Ensure that the signal has an even length by trimming the last sample if necessary.

    Parameters
    ----------
    signal : `array`
        Input signal data.

    Returns
    -------
    signal_even : `array`
        Signal with even length.
    """
    if len(signal) % 2 != 0:
        signal = signal[:-1]  # Trim the last sample to ensure even length
    return signal


def freq_response(signal, freq_Nyquist, fpad=0, get_dominant_freq=False):
    """Calculate the response in frequency domain of a time signal via FFT.

    Parameters
    ----------
    signal : `array`
        Signal data.
    freq_Nyquist : `float`
        Nyquist frequency according to the time step. freq_Nyquist = 1 / (2 * dt).
    fpad : `int`, optional
        Padding factor for FFT. Default is 0, which means no padding.
    get_dominant_freq : `bool`, optional
        If `True`, return only the dominant frequency of the spectrum. Default is `False`.

    Returns
    -------
    norm_magnitude : `array`
        Normalized frequency spectrum with respect to the maximum magnitude.
    dominant_freq : `float`, optional
        Dominant frequency of the spectrum.
    """

    # Check if the signal is empty
    if signal.size == 0:
        raise ValueError("Input signal is empty. Cannot compute frequency response.")

    # Check if the Nyquist frequency is positive
    validate_numeric("freq_Nyquist", freq_Nyquist, lower_bound=0.)

    # Remove DC offset
    signal -= mean(signal)

    # Remove linear trend so ends are closer to zero
    signal = detrend(signal)

    # Apply window to taper ends to zero
    window = hanning(len(signal))
    signal *= window

    # Zero padding for increasing smoothing in FFT
    signal = pad(signal, (0, fpad * len(signal)), 'constant')

    # Ensure even number of samples for FFT
    signal = ensure_even_length(signal)

    # Number of sample points for FFT
    N_samples = len(signal)

    # Calculate the response in frequency domain of the signal (FFT)
    #  N // 2 + 1 samples for real input, where N is the length of the input array.
    norm_magnitude = abs(rfft(signal))

    if get_dominant_freq:

        # Sample spacing
        d_sample = 1. / (2. * freq_Nyquist)

        # Frequency vector
        frequencies = rfftfreq(N_samples, d=d_sample)

        # Get the Dominant frequency of the spectrum
        dominant_freq = frequencies[norm_magnitude.argmax()]

        # Return the dominant frequency only
        return dominant_freq

    else:

        # Normalized frequency spectrum
        norm_magnitude *= (1 / norm_magnitude.max())

        # Return the normalized spectrum
        return norm_magnitude


def fft_at_receivers(number_of_receivers, forward_solution_receivers, freq_Nyquist):
    """Compute the FFT for output signals at receivers.

    Parameters
    ----------
    number_of_receivers : `int`
        Number of receivers.
    forward_solution_receivers : `array`
        Receiver waveform data acquired from forward proeblem.
    freq_Nyquist : `float`
        Nyquist frequency according to the time step. freq_Nyquist = 1 / (2 * dt).

    Returns
    -------
    receivers_out_fft : `array`
        Frequency response magnitude of the computed receiver data. The first
        dimension corresponds to frequency response and the second to receivers.
    """

    # Check the input parameters
    validate_numeric("number_of_receivers", number_of_receivers,
                     float_num=False, integer_num=True, lower_bound=0.)
    validate_data_structure("forward_solution_receivers", forward_solution_receivers,
                            "array2D", expected_shape=(None, number_of_receivers))

    # Compute the length of the FFT output
    length_fft = (forward_solution_receivers.shape[0]
                  - forward_solution_receivers.shape[0] % 2) // 2 + 1

    # Compute FFT for output signal at receivers
    receivers_out_fft = empty((length_fft, number_of_receivers))
    for rec in range(number_of_receivers):
        signal = forward_solution_receivers[:, rec]
        receivers_out_fft[:, rec] = freq_response(signal, freq_Nyquist)

    return receivers_out_fft
