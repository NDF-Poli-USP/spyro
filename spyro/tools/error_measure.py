from os import getcwd
from numpy import concatenate, inf, load, save, savetxt, trapezoid, zeros
from scipy.signal import find_peaks
from ..io.basicio import parallel_print as pprint
from ..utils.error_management import (validate_data_structure, validate_file,
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
    """Class for the error calculation for comparison purposes between models.

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
    error_measures_habc()
        Compute the error measures at the receivers for the HABC scheme
    get_reference_signal()
        Acquire the reference signal to compare with the HABC scheme
    save_reference_signal()
        Save the reference signal for the HABC scheme
    comparison_plots()
        Plot the comparison between the HABC scheme and the reference model
    get_xCR_candidates()
        Get the heuristic factor candidates for the quadratic regression
    get_xCR_optimal()
        Get the optimal heuristic factor for the quadratic damping
    """

    def __init__(self, output_folder=None, output_case=None, comm=None):
        """Initialize the MeasureError class.

        Parameters
        ----------
        output_folder : `str`, optional
            The folder where output data will be saved. Default is `None`.
        output_case : `str`, optional
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
            self.path_save_error = getcwd() + "/output/"
        else:
            self.path_save_error = validate_string("output_folder", output_folder)

        # Path to save data
        if output_case is None:
            self.path_save_err_case = self.path_save_error
        else:
            self.path_save_err_case = validate_string("output_case", output_case)

        # Path to save the reference signal
        self.path_reference = self.path_save_error + "preamble/"

        # Communicator MPI
        self.comm = comm

    def save_reference_signal(self, receiver_locations, forward_solution_receivers,
                              number_of_receivers, freq_Nyquist, output_file="reference"):
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
        validate_numeric("freq_Nyquist", freq_Nyquist, float_num=True,
                         integer_num=True, lower_bound=0.)

        pprint("\nSaving Reference Output", comm=self.comm)

        # File name for saving the reference signal
        self.output_file = validate_string("output_file", output_file)

        # Path to the reference data folder with reference signals
        pth_str = self.path_reference + self.output_file + "_"

        # Saving reference signal
        save(pth_str + "time.npy", forward_solution_receivers)

        # Computing and saving FFT of the reference signal at receivers
        receivers_ref_fft = []
        for rec in range(number_of_receivers):
            signal = forward_solution_receivers[:, rec]
            yf = freq_response(signal, freq_Nyquist)
            receivers_ref_fft.append(yf)
            save(pth_str + "fft.npy", receivers_ref_fft)

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
        receivers_reference_file = validate_file(
            "reference time file", pth_str + "time.npy", [".npy"], check_file_existence=True,
        )
        receivers_reference = load(receivers_reference_file)

        # Frequency domain signal
        receivers_reference_fft_file = validate_file(
            "reference fft file", pth_str + "fft.npy", [".npy"], check_file_existence=True,
        )
        receivers_ref_fft = load(receivers_reference_fft_file).T

        return receivers_reference, receivers_ref_fft

    @staticmethod
    def peak_error(signal_model, signal_reference):
        """Compute the peak error between the model and reference signals.

        Error measures used in Salas et al. (2022) Sec. 2.5.
        Hybrid absorbing scheme based on hyperelliptical layers with non-reflecting
        boundary conditions in scalar wave equations. Applied Mathematical Modelling.
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: add citation

        Parameters
        ----------
        signal_model : `array`
            Transient response ar the receiver for the model.
        signal_reference : `array`
            Transient response at the receiver for the reference model.

        Returns
        -------
        peak_error : `float`
            Peak error between the model and reference signals.
        peak_reference : `float`
            Maximum peak value of the reference signal.
        """

        # Check the input parameters
        validate_data_structure("signal_model", signal_model, "array",
                                expected_type_element="float")
        validate_data_structure("signal_reference", signal_reference, "array",
                                expected_type_element="float")

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

    @staticmethod
    def integral_error(signal_model, signal_reference, dt):
        """Compute the integral error between the model and reference signals.

        Error measures used in Salas et al. (2022) Sec. 2.5.
        Hybrid absorbing scheme based on hyperelliptical layers with non-reflecting
        boundary conditions in scalar wave equations. Applied Mathematical Modelling.
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: add citation

        Parameters
        ----------
        signal_model : `array`
            Transient response at the receiver for the model.
        signal_reference : `array`
            Transient response at the receiver for the reference model.
        dt : `float`
            Time step used in the simulation.

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

        # Completing with zeros if arrays lengths are different
        model_len = len(signal_model)
        reference_len = len(signal_reference)
        delta_len = abs(model_len - reference_len)
        if reference_len < model_len:
            signal_reference = concatenate([signal_reference, zeros(delta_len)])
        elif reference_len > model_len:
            signal_model = concatenate([signal_model, zeros(delta_len)])

        # Integral error
        numerator = trapezoid((signal_model - signal_reference)**2, dx=dt)
        denominator = trapezoid(signal_reference**2, dx=dt)
        integral_error = numerator / denominator if denominator != 0 else inf

        return integral_error

    def error_measures(self, forward_solution_receivers, receivers_reference, dt,
                       number_of_receivers, final_energy=None,
                       final_energy_reference=None, save_in_case_folder=True):
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
        final_energy : `float`, optional
            Energy of the model in the last time step. Default is `None`.
        final_energy_reference : `float`, optional
            Energy of the reference model in the last time step. Default is `None`.
        save_in_case_folder : `bool`, optional
            If `True`, save the error measures in the current case folder. Otherwise,
            save the error measures in the reference folder. Default is `True`.

        Returns
        -------
        error_measures : `list`
            Error measures at the receivers with respect to a reference model.
            Structure: [errIt, errPk, pkMax, max_errIt, max_errPK, final_ener, dsspt_ener]
            - errIt : `list`
                Integral error.
            - errPk : `list`
                Peak error.
            - pkMax : `list`
                Maximum reference peak.
            - max_errIt : `float`
                Maximum integral error.
            - max_errPK : `float`
                Maximum peak error
            - final_ener : `float`
                Final energy of the model. Only available if `final_energy` is provided.
            - dsspt_ener : `float`
                Total energy dissipated with respect to a reference model.
                Only available if `final_energy_reference` is provided.

        Notes
        -----
        - An error during execution in `find_peaks` means that the simulation
            transient time should be increased in order to observe a peak.
        - The `final_ener` value correspond to the mechanical energy in the
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
        pkMax = []  # Maximum reference peak
        errPk = []  # Peak error
        errIt = []  # Integral error

        for i in range(number_of_receivers):

            # Transient response at receiver
            u_abc = forward_solution_receivers[:, i]
            u_ref = receivers_reference[:, i]

            # Peak error and Maximum peak
            peak_error, peak_reference = self.peak_error(u_abc, u_ref)
            pkMax.append(peak_reference)
            errPk.append(peak_error)

            # Integral error
            integral_error = self.integral_error(u_abc, u_ref, dt)
            errIt.append(integral_error)

        # Receiver error measures
        error_measures = [errIt, errPk, pkMax]
        max_errIt = max(errIt)
        max_errPK = max(errPk)
        scalar_values = [max_errIt, max_errPK]
        pprint(f"Maximum Integral Error: {max_errIt:.2%}", comm=self.comm)
        pprint(f"Maximum Peak Error: {max_errPK:.2%}", comm=self.comm)

        # Save error measures
        pth_str = self.path_save_err_case if save_in_case_folder else self.path_reference
        err_str = pth_str + "measure_errs.txt"
        savetxt(err_str, error_measures, delimiter='\t')

        # Final energy
        if final_energy is not None:
            scalar_values.append(final_energy)
            pprint(f"Final Energy (J): {final_energy:.2e}", comm=self.comm)

            # Dissipated energy
            if final_energy_reference is not None:
                dsspt_ener = 1 - final_energy / final_energy_reference
                scalar_values.append(dsspt_ener)
                pprint(f"Dissipated Energy: {dsspt_ener:.2%}", comm=self.comm)

        error_measures.extend(scalar_values)

        # Append scalar values to the error measures list
        with open(err_str, 'a') as f:
            savetxt(f, scalar_values, delimiter='\t')

        return error_measures

    # def comparison_plots(self, regression_xCR=False, data_regr_xCR=None):
    #     """
    #     Plot the comparison between the HABC scheme and the reference model.

    #     Parameters
    #     ----------
    #     regression_xCR : `bool`, optional
    #         If True, Plot the regression for the error measure vs xCR
    #         Default is False.
    #     data_regr_xCR: `list`
    #         Data for the regression of the parameter xCR.
    #         Structure: [xCR, max_errIt, max_errPK, crit_opt]
    #         - xCR: Values of xCR used in the regression.
    #           The last value IS the optimal xCR
    #         - max_errIt: Values of the maximum integral error.
    #           The last value corresponds to the optimal xCR
    #         - max_errPK: Values of the maximum peak error.
    #           The last value corresponds to the optimal xCR
    #         - crit_opt : Criterion for the optimal heuristic factor.
    #           * 'err_difference' : Difference between integral and peak errors
    #           * 'err_integral' : Minimum integral error

    #     Returns
    #     -------
    #     None
    #     """

    #     # Time domain comparison
    #     plot_hist_receivers(self)

    #     forward_solution_receivers: `array`
    #     Receiver waveform data in the HABC scheme
    #     receivers_out_fft: `array`
    #     Frequency response at the receivers in the HABC scheme

    #     # Compute FFT for output signal at receivers
    #     self.receivers_out_fft = []
    #     for rec in range(self.number_of_receivers):
    #         signal = self.forward_solution_receivers[:, rec]
    #         yf = freq_response(signal, self.freq_Nyquist)
    #         self.receivers_out_fft.append(yf)
    #     self.receivers_out_fft = np.asarray(self.receivers_out_fft).T

    #     # Frequency domain comparison
    #     plot_rfft_receivers(self)

    #     # Plot the error measures
    #     if regression_xCR:
    #         plot_xCR_opt(self, data_regr_xCR)

    # def get_xCR_candidates(self, n_pts=3):
    #     """
    #     Get the heuristic factor candidates for the quadratic regression.

    #     Parameters
    #     ----------
    #     n_pts : `int`, optional
    #         Number of candidates for the heuristic factor xCR.
    #         Default is 3. Must be an odd number

    #     Returns
    #     -------
    #     xCR_cand : `list`
    #         Candidates for the heuristic factor xCR based on the
    #         current xCR and its bounds. The candidates are sorted
    #         in ascending order and current xCR is not included
    #     """

    #     # Setting odd number of points for regression
    #     n_pts = max(3, n_pts + 1 if n_pts % 2 == 0 else n_pts)

    #     # Limits for the heuristic factor
    #     xCR_inf, xCR_sup = self.xCR_lim

    #     # Estimated intial value
    #     xCR = self.xCR

    #     # Determining the xCR candidates for regression
    #     if xCR in self.xCR_lim:
    #         xCR_cand = list(np.linspace(xCR_inf, xCR_sup, n_pts))
    #         xCR_cand.remove(xCR)
    #     else:
    #         xCR_cand = list(np.linspace(xCR_inf, xCR_sup, n_pts-1))

    #     format_xCR = ', '.join(['{:.3f}'.format(x) for x in xCR_cand])
    #     pprint(f"Candidates for Heuristic Factor xCR: [{format_xCR}]", comm=self.comm)

    #     return xCR_cand

    # def get_xCR_optimal(self, dat_reg_xCR, crit_opt="err_sum"):
    #     """
    #     Get the optimal heuristic factor for the quadratic damping.

    #     Parameters
    #     ----------
    #     dat_reg_xCR : `list`
    #         Data for the regression of the parameter xCR.
    #         Structure: [xCR, max_errIt, max_errPK]
    #     crit_opt : `string`, optional
    #         Criterion for the optimal heuristic factor
    #         Default is 'err_difference'.
    #         - 'err_difference' : Difference between integral and peak errors
    #         - 'err_integral' : Minimum integral error
    #         - 'err_sum' : Sum of integral and peak errors

    #     Returns
    #     -------
    #     xCR_opt : `float`, optional
    #         Optimal heuristic factor for the quadratic damping
    #     """

    #     # Data for regression
    #     xCR = dat_reg_xCR[0]
    #     max_errIt = dat_reg_xCR[1]
    #     max_errPK = dat_reg_xCR[2]

    #     validate_parameter("crit_opt", crit_opt,
    #                           ["err_difference", "err_integral", "err_sum"])

    #     if crit_opt == "err_difference":
    #         y_err = [eI - eP for eI, eP in zip(max_errIt, max_errPK)]

    #     elif crit_opt == "err_integral":
    #         y_err = max_errIt

    #     elif crit_opt == "err_sum":
    #         y_err = [eI + eP for eI, eP in zip(max_errIt, max_errPK)]

    #     # Limits for the heuristic factor
    #     xCR_inf, xCR_sup = self.xCR_lim

    #     # Coefficients for the quadratic equation
    #     eq_xCR = np.polyfit(xCR, y_err, 2)

    #     if crit_opt == "err_difference":
    #         # Roots of the quadratic equation
    #         roots = np.roots(eq_xCR)
    #         valid_roots = [np.clip(rth, xCR_inf, xCR_sup)
    #                        for rth in roots if isinstance(rth, float)]

    #         if valid_roots:
    #             # Real root that provides the absolute minimum error
    #             min_err = [abs(np.polyval(eq_xCR, rth)) for rth in valid_roots]
    #             xCR_opt = valid_roots[np.argmin(min_err)]
    #         else:
    #             # Vertex when there are no real roots
    #             vtx = - eq_xCR[1] / (2 * eq_xCR[0])
    #             xCR_opt = np.clip(vtx, xCR_inf, xCR_sup)

    #     elif crit_opt == "err_integral" or crit_opt == "err_sum":

    #         # Vertex of the quadratic equation
    #         vtx = - eq_xCR[1] / (2 * eq_xCR[0])
    #         xCR_opt = np.clip(vtx, xCR_inf, xCR_sup)

    #     pprint(f"Optimal Heuristic Factor xCR: {xCR_opt:.3f}", comm=self.comm)

    #     return xCR_opt
