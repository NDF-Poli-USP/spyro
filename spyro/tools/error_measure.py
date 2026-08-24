from os import getcwd
import numpy as np
from numpy import inf, load, pad, save, savetxt, trapezoid
from numpy.linalg import norm
from pathlib import Path
from scipy.signal import find_peaks
from ..io.basicio import parallel_print as pprint
from ..utils.error_management import (mutually_exclusive_parameter_error,
                                      validate_data_structure, validate_file,
                                      validate_numeric, validate_string)
from ..utils.freq_tools import freq_response
# from ..plots.plots_habc import plot_hist_receivers, plot_rfft_receivers, plot_xCR_opt
# from ..utils.error_management import validate_parameter

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# TODO: add citation
# With additions by Alexandre Olender


class MeasureError():
    """Manage reference data and error calculations for wave simulations.

    This class is responsible primarily for managing output paths,
    reference signals, MPI communication, and orchestration of error
    calculations. Individual error measures are implemented as standalone
    functions so that they can also be used independently.

    Attributes
    ----------
    path_reference : `str`
        Path to save the reference signal.
    path_save_error : `str`
        Path to save data.
    path_save_err_case : `str`
        Path to save data for the current case study.

    Methods
    -------
    error_measures()
        Compute the error measures at the receivers for comparison between models.
    get_reference_signal()
        Acquire the reference signal for comparison between models.
    integral_error()
        Compute the integral error between the model and reference signals.
    normalized_root_mean_square_error()
        Compute the normalized RMS error between the model and reference signals.
    peak_error()
        Compute the peak error between the model and reference signals.
    save_reference_signal()
        Save the reference signal for comparison between models.
    comparison_plots()
        Plot the comparison between the HABC scheme and the reference model
    get_xCR_candidates()
        Get the heuristic factor candidates for the quadratic regression
    get_xCR_optimal()
        Get the optimal heuristic factor for the quadratic damping
    """

    def __init__(
        self,
        output_folder: Path | str | None = None,
        output_case: Path | str | None = None,
        comm=None,
    ):
        """Initialize the MeasureError class.

        Parameters
        ----------
        output_folder : pathlib.Path or `str`, optional
            The folder where output data will be saved. Default is `None`.
        output_case : pathlib.Path or `str`, optional
            The folder for the current case study. Default is `None`.
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Path to save data
        if output_folder is None:
            output_folder = Path(getcwd()) / "output"
        elif isinstance(output_folder, str):
            output_folder = Path(
                validate_string("output_folder", output_folder)
            )
        # Path to save data
        if output_case is None:
            output_case = output_folder
        elif isinstance(output_case, str):
            output_case = Path(
                validate_string("output_case", output_case)
            )

        self.path_save_error = output_folder
        self.path_save_err_case = output_case
        self.path_reference = output_folder / "preamble"
        self.comm = comm

    def save_reference_signal(
        self,
        receiver_locations,
        forward_solution_receivers: np.ndarray,
        number_of_receivers,
        nyquist_frequency: float,
        output_file="reference",
    ):
        """Save the reference signal for comparison between models.

        Parameters
        ----------
        receiver_locations: `list`
            List of receiver locations.
        forward_solution_receivers : `array`
            Receiver waveform data acquired from forward proeblem.
        number_of_receivers: `int`
            Number of receivers used in the simulation.
        freq_Nyquist : `float`
            Nyquist frequency according to the time step. freq_Nyquist = 1 / (2 * dt).
        output_file : `str`, optional
            Name of the file to save the reference signal without any extension.
            Default is "ref_rec.npy".

        Returns
        -------
        None
        """

        # Check the input parameters
        validate_numeric("number_of_receivers", number_of_receivers,
                         float_num=False, integer_num=True, lower_bound=0.)
        validate_data_structure("receiver_locations", receiver_locations, "list",
                                expected_type_element="tuple",
                                expected_length=number_of_receivers)
        validate_data_structure("forward_solution_receivers", forward_solution_receivers,
                                "array2D", expected_type_element="float",
                                expected_shape=(None, number_of_receivers))
        validate_numeric("nyquist_frequency", nyquist_frequency, float_num=True,
                         integer_num=True, lower_bound=0.)

        pprint("\nSaving Reference Output", comm=self.comm)

        # File name for saving the reference signal
        self.path_reference.mkdir(parents=True, exist_ok=True)

        output_file_prefix = self.path_reference / f"{output_file}_"

        save(
            str(output_file_prefix) + "time.npy",
            forward_solution_receivers,
        )

        # Computing and saving FFT of the reference signal at receivers
        receivers_ref_fft = []
        for rec in range(number_of_receivers):
            signal = forward_solution_receivers[:, rec]
            yf = freq_response(signal, nyquist_frequency)
            receivers_ref_fft.append(yf)
            save(
                str(output_file_prefix) + "fft.npy",
                receivers_ref_fft,
            )

    def get_reference_signal(self):
        """Acquire the reference signal for comparison between models.

        Parameters
        ----------
        None

        Returns
        -------
        receivers_reference : `array`
            Receiver waveform data in the reference model
        receivers_ref_fft : `array`
          Frequency response at the receivers in the reference model.
        """

        pprint("\nLoading Reference Signal from Reference Model", comm=self.comm)

        # Path to the reference data folder with reference signals
        pth_str = self.path_reference + self.output_file + "_"

        # Time domain signal
        receivers_reference_file = validate_file("reference time file",
                                                 pth_str + "time.npy", [".npy"],
                                                 check_file_existence=True)
        receivers_reference = load(receivers_reference_file)

        # Frequency domain signal
        receivers_reference_fft_file = validate_file("reference fft file",
                                                     pth_str + "fft.npy", [".npy"],
                                                     check_file_existence=True)
        receivers_ref_fft = load(receivers_reference_fft_file).T

        return receivers_reference, receivers_ref_fft

    def calculate_error_measures(
        self,
        forward_solution_receivers: np.ndarray,
        receivers_reference: np.ndarray,
        dt: float,
        number_of_receivers: int,
        error_if_different_length: bool =True,
        final_energy: float =None,
        final_energy_reference: float =None,
        save_file: bool =True,
        save_in_case_folder: bool =True,
        start_padding: bool =False,
        end_padding: bool =False,
    ):
        """Compute the error measures at the receivers for comparison between models.

        Error measures used in Salas et al. (2022) Sec. 2.5.
        Hybrid absorbing scheme based on hyperelliptical layers with non-reflecting
        boundary conditions in scalar wave equations. Applied Mathematical Modelling.
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: add citation

        Parameters
        ----------
        forward_solution_receivers : `array`
            Receiver waveform data acquired from forward proeblem.
        receivers_reference : `array`
            Receiver waveform data in the reference model
        dt : `float`
            Time step used in the simulation.
        number_of_receivers: `int`
            Number of receivers used in the simulation.
        error_if_different_length: `bool`, optional
            If `True`, raise an error if the lengths of the model and reference
            signals are different. Default is `True`.
        final_energy : `float`, optional
            Energy of the model in the last time step. Default is `None`.
        final_energy_reference : `float`, optional
            Energy of the reference model in the last time step. Default is `None`.
        save_file : `bool`, optional
            If `True`, save the error measures in a text file. Default is `True`.
        save_in_case_folder : `bool`, optional
            If `True`, save the error measures in the current case folder. Otherwise,
            save the error measures in the reference folder. Default is `True`.
        start_padding: `bool`, optional
            If `True`, pad the shorter signal with zeros at the start to match
            the length of the other signal. Default is `False`.
        end_padding: `bool`, optional
            If `True`, pad the shorter signal with zeros at the end to match the
            length of the other signal. Default is `False`.

        Returns
        -------
        error_measures : `list`
            Error measures at the receivers with respect to a reference model,
            in the following order:
            Structure: [integral_error, peak_errors,...]
            - integral_error : `list`
                Integral error.
            - peak_error : `list`
                Peak error.
            - maximum_reference_peak : `list`
                Maximum reference peak.
            - maximum_integral_error : `float`
                Maximum integral error.
            - maximum_peak_error : `float`
                Maximum peak error
            - final_energy : `float`
                Final energy of the model. Only available if `final_energy` is provided.
            - dissipated_energy : `float`
                Total energy dissipated with respect to a reference model.
                Only available if `final_energy_reference` is provided.

        Notes
        -----
        - An error during execution in `find_peaks` means that the simulation
            transient time should be increased in order to observe a peak.
        - The `final_energy` value correspond to the mechanical energy in the
            last step of the simulation. If the model has an ABC scheme, the
            value should be close to zero. Otherwise, the value is constant
            during the simulation due to the law of conservation of energy.
        - The total energy dissipated by an ABC scheme can be calculated as the
            difference of the final energies with respect to an infinite model.
        """

        # Check the input parameters
        validate_numeric("number_of_receivers", number_of_receivers,
                         float_num=False, integer_num=True, lower_bound=0.)
        validate_data_structure("forward_solution_receivers", forward_solution_receivers,
                                "array2D", expected_shape=(None, number_of_receivers))
        validate_data_structure("receivers_reference", receivers_reference, "array2D",
                                expected_shape=(None, number_of_receivers))
        validate_numeric("dt", dt, float_num=True, integer_num=True, lower_bound=0.)
        validate_numeric("final_energy", final_energy, float_num=True,
                         integer_num=False, lower_bound=0.)
        validate_numeric("final_energy_reference", final_energy_reference,
                         float_num=True, integer_num=False, lower_bound=0.)

        pprint("\nComputing Error Measures", comm=self.comm)

        # Initializing error measures
        reference_peak_values = []
        peak_errors = []
        integral_errors = []

        for i in range(number_of_receivers):

            # Transient response at receiver
            model_receiver_signal = forward_solution_receivers[:, i]
            reference_receiver_signal = receivers_reference[:, i]

            # Peak error and Maximum peak
            peak_error, reference_peak = calculate_peak_error(
                model_receiver_signal,
                reference_receiver_signal,
            )
            reference_peak_values.append(reference_peak)
            peak_errors.append(peak_error)

            # Integral error
            integral_error = calculate_integral_error(
                model_receiver_signal,
                reference_receiver_signal,
                dt,
                error_if_different_length=error_if_different_length,
                start_padding=start_padding,
                end_padding=end_padding,
            )
            integral_errors.append(integral_error)

        # Receiver error measures
        error_measures = [
            integral_errors,
            peak_errors,
            reference_peak_values,
        ]
        maximum_integral_error = max(integral_errors)
        maximum_peak_error = max(peak_errors)
        scalar_values = [
            maximum_integral_error,
            maximum_peak_error,
        ]
        pprint(
            f"Maximum Integral Error: {maximum_integral_error:.2%}",
            comm=self.comm,
        )
        pprint(
            f"Maximum Peak Error: {maximum_peak_error:.2%}",
            comm=self.comm,
        )
        # Final energy
        if final_energy is not None:
            scalar_values.append(final_energy)
            pprint(
                f"Final Energy (J): {final_energy:.2e}",
                comm=self.comm,
            )

            # Dissipated energy
            if final_energy_reference is not None:
                dissipated_energy = (
                    1 - final_energy / final_energy_reference
                )
                scalar_values.append(dissipated_energy)
                pprint(
                    f"Dissipated Energy: {dissipated_energy:.2%}",
                    comm=self.comm,
                )

        error_measures.extend(scalar_values)

        # Save error measures
        if save_file:
            if save_in_case_folder:
                output_directory = self.path_save_err_case
            else:
                output_directory = self.path_reference
            output_directory.mkdir(parents=True, exist_ok=True)
            error_file = output_directory / "measure_errs.txt"

            savetxt(
                error_file,
                error_measures[:3],
                delimiter="\t",
            )
            with error_file.open("a") as file_handle:
                savetxt(
                    file_handle,
                    scalar_values,
                    delimiter="\t",
                )

        return error_measures


def pad_signal_lengths(
    signal_model: np.ndarray,
    signal_reference: np.ndarray,
    error_if_different_length: bool = True,
    start_padding: bool = False,
    end_padding: bool = False,
):
    """Equalize the signal lengths in comparison by padding with zeros.

    Parameters
    ----------
    signal_model : `np.ndarray`
        Transient response at the receiver for the model.
    signal_reference : `np.ndarray`
        Transient response at the receiver for the reference model.
    error_if_different_length : `bool`, optional
        If `True`, raise an error if the lengths of the model and reference
        signals are different. Default is `True`.
    start_padding : `bool`, optional
        If `True`, pad the shorter signal with zeros at the start to match
        the length of the other signal. Default is `False`.
    end_padding : `bool`, optional
        If `True`, pad the shorter signal with zeros at the end to match the
        length of the other signal. Default is `False`.

    Returns
    -------
    signal_model : `np.ndarray`
        Transient response at the receiver for the model, modified with a zero pad
        if shorter than the reference.
    signal_reference : `np.ndarray`
        Transient response at the receiver for the reference model, modified with
        a zero pad if shorter than the model signal.

    Raises
    ------
    ValueError
        If the signals have different lengths and
        `error_if_different_length` is `True`.
        Also raised if both or neither of `start_padding` and
        `end_padding` are `True` when padding is required.
    """

    # Raise an error if signal lengths must be verified and are different
    if error_if_different_length and len(signal_model) != len(signal_reference):
        raise ValueError("The lengths of the model and reference signals "
                            "are different. Please check the simulation time "
                            " or the time step used in the simulations.")

    if len(signal_model) == len(signal_reference):
        return signal_model, signal_reference

    def _pad_signal(signal, delta_len, padding_type):
        """Pad the signal with zeros to match the length of the other signal.

        Parameters
        ----------
        signal : `array`
            Transient signal that is the shorter of the two signals to compare.
        delta_len : `int`
            Difference in length between the two signals to compare.
        padding_type : `str`
            Type of padding to apply. Options: "end" or "start".

        Returns
        -------
        modified_signal : `array`
            Transient signal modified with zero padding at the start or end
            to match the length of the other signal to compare.
        """

        pad_distribution = (0, delta_len) if padding_type == "end" else (delta_len, 0)
        return pad(signal, pad_distribution, 'constant', constant_values=0)

    # Pad the shorter signal with zeros if the lengths are different

    # Check if both start and end padding are requested, which is not allowed
    if not (start_padding ^ end_padding):  # Not XOR: both True or both False
        mutually_exclusive_parameter_error(["end_padding", "start_padding"],
                                            [end_padding, start_padding])

    # Getting the maximum length
    max_length = max(len(signal_model), len(signal_reference))

    # Type of padding to apply
    padding_type = "end" if end_padding else "start"

    # Completing with zeros if arrays lengths are different
    if len(signal_model) < max_length:
        delta_len = max_length - len(signal_model)
        signal_model = _pad_signal(signal_model, delta_len, padding_type)
    elif len(signal_reference) < max_length:
        delta_len = max_length - len(signal_reference)
        signal_reference = _pad_signal(signal_reference, delta_len, padding_type)

    return signal_model, signal_reference


def calculate_peak_error(
    signal_model: np.ndarray,
    signal_reference: np.ndarray,
):
    """Compute the peak error between the model and reference signals.

    Error measures used in Salas et al. (2022) Sec. 2.5.
    Hybrid absorbing scheme based on hyperelliptical layers with non-reflecting
    boundary conditions in scalar wave equations. Applied Mathematical Modelling.
    doi: https://doi.org/10.1016/j.apm.2022.09.014
    TODO: add citation

    Parameters
    ----------
    signal_model : `np.ndarray`
        Transient response ar the receiver for the model.
    signal_reference : `np.ndarray`
        Transient response at the receiver for the reference model.

    Returns
    -------
    peak_error : `float`
        Peak error between the model and reference signals.
    peak_reference : `float`
        Maximum peak value of the reference signal.
    """

    # Check the input parameters
    validate_data_structure(
        "signal_model",
        signal_model,
        "array",
        expected_type_element="float",
    )
    validate_data_structure(
        "signal_reference",
        signal_reference,
        "array",
        expected_type_element="float",
    )

    # Finding peaks in transient response
    peaks_in_signal = find_peaks(signal_model)
    if peaks_in_signal[0].size == 0:
        UserWarning("No peak observed in the transient response. "
                    "Increase the transient time of the simulation.")

    # Maximum peak value
    peak_model = max(abs(signal_model))
    peak_reference = max(abs(signal_reference))

    # Peak error
    peak_error = abs(peak_model / peak_reference - 1)

    return peak_error, peak_reference


def calculate_integral_error(
    signal_model: np.ndarray,
    signal_reference: np.ndarray, 
    dt: float,
    error_if_different_length: bool = True,
    start_padding: bool = False,
    end_padding: bool = False,
):
    """Compute the integral error between the model and reference signals.

    Error measures used in Salas et al. (2022) Sec. 2.5.
    Hybrid absorbing scheme based on hyperelliptical layers with non-reflecting
    boundary conditions in scalar wave equations. Applied Mathematical Modelling.
    doi: https://doi.org/10.1016/j.apm.2022.09.014
    TODO: add citation

    Parameters
    ----------
    signal_model : `np.ndarray`
        Transient response at the receiver for the model.
    signal_reference : `np.ndarray`
        Transient response at the receiver for the reference model.
    dt : `float`
        Time step used in the simulation.
    error_if_different_length : `bool`, optional
        If `True`, raise an error if the lengths of the model and reference
        signals are different. Default is `True`.
    start_padding : `bool`, optional
        If `True`, pad the shorter signal with zeros at the start to match
        the length of the other signal. Default is `False`.
    end_padding : `bool`, optional
        If `True`, pad the shorter signal with zeros at the end to match the
        length of the other signal. Default is `False`.

    Returns
    -------
    integral_error : `float`
        Integral error between the model and reference signals.
    """

    # Check the input parameters
    validate_data_structure("signal_model", signal_model, "array",
                            expected_type_element="float")
    validate_data_structure("signal_reference", signal_reference, "array",
                            expected_type_element="float")
    validate_numeric("dt", dt, float_num=True, integer_num=True, lower_bound=0.)

    # Padding with zeros if arrays lengths are different
    signal_model, signal_reference = pad_signal_lengths(
        signal_model, signal_reference,
        error_if_different_length=error_if_different_length,
        start_padding=start_padding, end_padding=end_padding)

    # Integral error
    numerator = trapezoid((signal_model - signal_reference)**2, dx=dt)
    denominator = trapezoid(signal_reference**2, dx=dt)
    integral_error = numerator / denominator if denominator != 0 else inf

    return integral_error


def calculate_normalized_L2_error(
    signal_model: np.ndarray,
    signal_reference: np.ndarray,
    error_if_different_length=True,
    start_padding=False,
    end_padding=False,
):
    """Compute the normalized L2 error between the model and reference signals.

    Takem from https://www.statisticshowto.com/nrmse/
    TODO: add citation

    Parameters
    ----------
    signal_model : `array`
        Transient response at the receiver for the model.
    signal_reference : `array`
        Transient response at the receiver for the reference model.
    error_if_different_length: `bool`, optional
        If `True`, raise an error if the lengths of the model and reference
        signals are different. Default is `True`.
    start_padding: `bool`, optional
        If `True`, pad the shorter signal with zeros at the start to match
        the length of the other signal. Default is `False`.
    end_padding: `bool`, optional
        If `True`, pad the shorter signal with zeros at the end to match the
        length of the other signal. Default is `False`.

    Returns
    -------
    nrms_error : `float`
        Normalized L2 error between the model and reference signals.
    """

    # Check the input parameters
    validate_data_structure("signal_model", signal_model, "array",
                            expected_type_element="float")
    validate_data_structure("signal_reference", signal_reference, "array",
                            expected_type_element="float")

    # Padding with zeros if arrays lengths are different
    signal_model, signal_reference = pad_signal_lengths(
        signal_model, signal_reference,
        error_if_different_length=error_if_different_length,
        start_padding=start_padding, end_padding=end_padding)

    # Normalized RMS error
    numerator = norm(signal_model - signal_reference)
    denominator = norm(signal_reference)
    nrms_error = numerator / denominator if denominator != 0 else inf

    return nrms_error
