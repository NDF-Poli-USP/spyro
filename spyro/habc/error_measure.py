import numpy as np
from os import getcwd
from firedrake import assemble
from scipy.signal import find_peaks
from spyro.plots.plots_habc import plot_hist_receivers, \
    plot_rfft_receivers, plot_xCR_opt
from spyro.utils.freq_tools import freq_response
from spyro.utils.error_management import value_parameter_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Error():
    '''
    Class for the error calculation for the HABC scheme

    Attributes
    ----------
    dt : `float`
        Time step used in the simulation
    err_habc : `list`
        Error measures at the receivers for the HABC scheme.
        Structure: [errIt, errPk, pkMax, final_energy]
        - errIt : Integral error
        - errPk : Peak error
        - pkMax : Maximum reference peak
        - final_energy : Dissipated energy in the HABC scheme
    f_Nyq : `float`
        Nyquist frequency according to the time step. f_Nyq = 1 / (2 * dt)
    max_errIt : `float`
        Maximum integral error at the receivers for the HABC scheme
    max_errPK : `float`
        Maximum peak error at the receivers for the HABC scheme
    number_of_receivers: `int`
        Number of receivers used in the simulation
    path_save_error : `string`
        Path to save data
    path_save_err_case : `string`
        Path to save data for the current case study
    receiver_locations: `list`
        List of receiver locations
    receivers_output : `array`
        Receiver waveform data in the HABC scheme
    receivers_out_fft : `array`
        Frequency response at the receivers in the HABC scheme
    receivers_reference : `array`
        Receiver waveform data in the reference model
    receivers_ref_fft : `array`
        Frequency response at the receivers in the reference model

    Methods
    -------
    comparison_plots()
        Plot the comparison between the HABC scheme and the reference model
    error_measures_habc()
        Compute the error measures at the receivers for the HABC scheme
    get_reference_signal()
        Acquire the reference signal to compare with the HABC scheme
    get_xCR_candidates()
        Get the heuristic factor candidates for the quadratic regression
    get_xCR_optimal()
        Get the optimal heuristic factor for the quadratic damping
    save_reference_signal()
        Save the reference signal for the HABC scheme
    '''

    def __init__(self, dt, f_Nyq, receiver_locations, receivers_output=None,
                 output_folder=None, output_case=None):
        '''
        Initialize the HABC_Error class.

        Parameters
        ----------
        dt : `float`
            Time step used in the simulation
        f_Nyq : `float`
            Nyquist frequency according to the time step. f_Nyq = 1 / (2 * dt)
        receiver_locations: `list`
            List of receiver locations
        receivers_output : `array`, optional
            Receiver waveform data in the HABC scheme. Default is None
        output_folder : str, optional
            The folder where output data will be saved. Default is None
        output_case : str, optional
            The folder for the current case study. Default is None

        Returns
        -------
        None
        '''

        # Time step
        self.dt = dt

        # Nyquist frequency
        self.f_Nyq = f_Nyq

        # Receivers data and initialization
        self.receiver_locations = receiver_locations
        self.number_of_receivers = len(self.receiver_locations)
        self.receivers_output = receivers_output

        # Path to save data
        if output_folder is None:
            self.path_save_error = getcwd() + "/output/"
        else:
            self.path_save_error = output_folder

        # Path to save data
        if output_case is None:
            self.path_save_err_case = self.path_save_error
        else:
            self.path_save_err_case = output_case

    def save_reference_signal(self):
        '''
        Save the reference signal for the HABC scheme

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nSaving Reference Output", flush=True)

        # Path to save the reference signal
        pth_str = self.path_save_error + "preamble/"

        # Saving reference signal
        self.receivers_reference = self.receivers_output.copy()
        np.save(pth_str + "habc_ref.npy", self.receivers_reference)

        # Computing and saving FFT of the reference signal at receivers
        self.receivers_ref_fft = []
        for rec in range(self.number_of_receivers):
            signal = self.receivers_reference[:, rec]
            yf = freq_response(signal, self.f_Nyq)
            self.receivers_ref_fft.append(yf)
        np.save(pth_str + "habc_fft.npy", self.receivers_ref_fft)

    def get_reference_signal(self, foldername="preamble/"):
        '''
        Acquire the reference signal to compare with the HABC scheme

        Parameters
        ----------
        foldername : `string`, optional
            Name of the folder where the reference signal is stored.
            Default is "preamble/"

        Returns
        -------
        None
        '''

        print("\nLoading Reference Signal from Infinite Model", flush=True)

        # Path to the reference data folder
        pth_str = self.path_save_error + foldername

        # Time domain signal
        self.receivers_reference = np.load(pth_str + "habc_ref.npy")

        # Frequency domain signal
        self.receivers_ref_fft = np.load(pth_str + "habc_fft.npy").T

    def error_measures_habc(self):
        '''
        Compute the error measures at the receivers for the HABC scheme.
        Error measures as in Salas et al. (2022) Sec. 2.5.
        Obs: If you get an error during running in find_peaks means that
        the transient time of the simulation must be increased.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nComputing Error Measures", flush=True)

        # Initializing error measures
        pkMax = []
        errPk = []
        errIt = []

        for i in range(self.number_of_receivers):

            # Transient response in receiver
            u_abc = self.receivers_output[:, i]
            u_ref = self.receivers_reference[:, i]

            # Finding peaks in transient response
            u_pks = find_peaks(u_abc)
            if u_pks[0].size == 0:
                wrn_str0 = "No peak observed in the transient response. "
                wrn_str1 = "Increase the transient time of the simulation."
                UserWarning(wrn_str0 + wrn_str1)

            # Maximum peak value
            p_abc = max(abs(u_abc))
            p_ref = max(abs(u_ref))
            pkMax.append(p_ref)

            # Completing with zeros if the length of arrays is different
            delta_len = abs(len(u_abc) - len(u_ref))
            if len(u_ref) < len(u_abc):
                u_ref = np.concatenate([u_ref, np.zeros(delta_len)])
            elif len(u_ref) > len(u_abc):
                u_abc = np.concatenate([u_abc, np.zeros(delta_len)])

            # Integral error
            errIt.append(np.trapezoid((u_abc - u_ref)**2, dx=self.dt)
                         / np.trapezoid(u_ref**2, dx=self.dt))

            # Peak error
            errPk.append(abs(p_abc / p_ref - 1))

        # Final value of the dissipated energy in the HABC scheme
        final_energy = assemble(self.acoustic_energy)
        self.err_habc = [errIt, errPk, pkMax, final_energy]
        self.max_errIt = max(errIt)
        self.max_errPK = max(errPk)
        print("Maximum Integral Error: {:.2%}".format(
            self.max_errIt), flush=True)
        print("Maximum Peak Error: {:.2%}".format(self.max_errPK), flush=True)
        print("Acoustic Energy: {:.2e}".format(final_energy), flush=True)

        # Save error measures
        err_str = self.path_save_err_case + "habc_errs.txt"
        np.savetxt(err_str, (errIt, errPk, pkMax), delimiter='\t')

        # Append the energy value at the end
        with open(err_str, 'a') as f:
            np.savetxt(f, np.array([final_energy]), delimiter='\t')

    def comparison_plots(self, regression_xCR=False, data_regr_xCR=None):
        '''
        Plot the comparison between the HABC scheme and the reference model.

        Parameters
        ----------
        regression_xCR : `bool`, optional
            If True, Plot the regression for the error measure vs xCR
            Default is False.
        data_regr_xCR: `list`
            Data for the regression of the parameter xCR.
            Structure: [xCR, max_errIt, max_errPK, crit_opt]
            - xCR: Values of xCR used in the regression.
              The last value IS the optimal xCR
            - max_errIt: Values of the maximum integral error.
              The last value corresponds to the optimal xCR
            - max_errPK: Values of the maximum peak error.
              The last value corresponds to the optimal xCR
            - crit_opt : Criterion for the optimal heuristic factor.
              * 'err_difference' : Difference between integral and peak errors
              * 'err_integral' : Minimum integral error

        Returns
        -------
        None
        '''

        # Time domain comparison
        plot_hist_receivers(self)

        # Compute FFT for output signal at receivers
        self.receivers_out_fft = []
        for rec in range(self.number_of_receivers):
            signal = self.receivers_output[:, rec]
            yf = freq_response(signal, self.f_Nyq)
            self.receivers_out_fft.append(yf)
        self.receivers_out_fft = np.asarray(self.receivers_out_fft).T

        # Frequency domain comparison
        plot_rfft_receivers(self)

        # Plot the error measures
        if regression_xCR:
            plot_xCR_opt(self, data_regr_xCR)

    def get_xCR_candidates(self, n_pts=3):
        '''
        Get the heuristic factor candidates for the quadratic regression.

        Parameters
        ----------
        n_pts : `int`, optional
            Number of candidates for the heuristic factor xCR.
            Default is 3. Must be an odd number

        Returns
        -------
        xCR_cand : `list`
            Candidates for the heuristic factor xCR based on the
            current xCR and its bounds. The candidates are sorted
            in ascending order and current xCR is not included
        '''

        # Setting odd number of points for regression
        n_pts = max(3, n_pts + 1 if n_pts % 2 == 0 else n_pts)

        # Limits for the heuristic factor
        xCR_inf, xCR_sup = self.xCR_lim

        # Estimated intial value
        xCR = self.xCR

        # Determining the xCR candidates for regression
        if xCR in self.xCR_lim:
            xCR_cand = list(np.linspace(xCR_inf, xCR_sup, n_pts))
            xCR_cand.remove(xCR)
        else:
            xCR_cand = list(np.linspace(xCR_inf, xCR_sup, n_pts-1))

        format_xCR = ', '.join(['{:.3f}'.format(x) for x in xCR_cand])
        print("Candidates for Heuristic Factor xCR: [{}]".format(
            format_xCR), flush=True)

        return xCR_cand

    def get_xCR_optimal(self, dat_reg_xCR, crit_opt='err_sum'):
        '''
        Get the optimal heuristic factor for the quadratic damping.

        Parameters
        ----------
        dat_reg_xCR : `list`
            Data for the regression of the parameter xCR.
            Structure: [xCR, max_errIt, max_errPK]
        crit_opt : `string`, optional
            Criterion for the optimal heuristic factor
            Default is 'err_difference'.
            - 'err_difference' : Difference between integral and peak errors
            - 'err_integral' : Minimum integral error
            - 'err_sum' : Sum of integral and peak errors

        Returns
        -------
        xCR_opt : `float`, optional
            Optimal heuristic factor for the quadratic damping
        '''

        # Data for regression
        xCR = dat_reg_xCR[0]
        max_errIt = dat_reg_xCR[1]
        max_errPK = dat_reg_xCR[2]

        if crit_opt == 'err_difference':
            y_err = [eI - eP for eI, eP in zip(max_errIt, max_errPK)]

        elif crit_opt == 'err_integral':
            y_err = max_errIt

        elif crit_opt == 'err_sum':
            y_err = [eI + eP for eI, eP in zip(max_errIt, max_errPK)]

        else:
            value_parameter_error('crit_opt', crit_opt,
                                  ['err_difference', 'err_integral', 'err_sum'])

        # Limits for the heuristic factor
        xCR_inf, xCR_sup = self.xCR_lim

        # Coefficients for the quadratic equation
        eq_xCR = np.polyfit(xCR, y_err, 2)

        if crit_opt == 'err_difference':
            # Roots of the quadratic equation
            roots = np.roots(eq_xCR)
            valid_roots = [np.clip(rth, xCR_inf, xCR_sup)
                           for rth in roots if isinstance(rth, float)]

            if valid_roots:
                # Real root that provides the absolute minimum error
                min_err = [abs(np.polyval(eq_xCR, rth)) for rth in valid_roots]
                xCR_opt = valid_roots[np.argmin(min_err)]
            else:
                # Vertex when there are no real roots
                vtx = - eq_xCR[1] / (2 * eq_xCR[0])
                xCR_opt = np.clip(vtx, xCR_inf, xCR_sup)

        elif crit_opt == 'err_integral' or crit_opt == 'err_sum':

            # Vertex of the quadratic equation
            vtx = - eq_xCR[1] / (2 * eq_xCR[0])
            xCR_opt = np.clip(vtx, xCR_inf, xCR_sup)

        print("Optimal Heuristic Factor xCR: {:.3f}".format(
            xCR_opt), flush=True)

        return xCR_opt
