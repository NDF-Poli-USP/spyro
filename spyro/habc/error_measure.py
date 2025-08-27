import numpy as np
from os import getcwd

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
    path_save_error : `string`
        Path to save data
    receivers_reference : `array`
        Receiver waveform data in the reference model
    receivers_ref_fft : `array`
        Frequency response at the receivers in the reference model.

    Methods
    -------
    get_reference_signal()
        Acquire the reference signal to compare with the HABC scheme
    save_reference_signal()
        Save the reference signal for the HABC scheme
    '''

    def __init__(self, output_folder=None):
        '''
        Initialize the HABC_Error class.

        Parameters
        ----------
        output_folder : str, optional
            The folder where output data will be saved. Default is None

        Returns
        -------
        None
        '''

        # Path to save data
        if output_folder is None:
            self.path_save_error = getcwd() + "/output/"
        else:
            self.path_save_error = output_folder

    def save_reference_signal(self, receivers_output, number_of_receivers):
        '''
        Save the reference signal for the HABC scheme

        Parameters
        ----------
        receivers_output : `array`
            Receiver waveform data in the HABC scheme
        number_of_receivers: `int`
            Number of receivers used in the simulation

        Returns
        -------
        None
        '''

        print("\nSaving Reference Output")

        # Path to save the reference signal
        pth_str = self.path_save_error + "preamble/"

        # Saving reference signal
        self.receivers_reference = receivers_output.copy()
        np.save(pth_str + "habc_ref.npy", self.receivers_reference)

        # Computing and saving FFT of the reference signal at receivers
        self.receivers_ref_fft = []
        for rec in range(number_of_receivers):
            signal = self.receivers_reference[:, rec]
            yf = self.freq_response(signal)
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

        print("\nLoading Reference Signal from Infinite Model")

        # Path to the reference data folder
        pth_str = self.path_save_error + foldername

        # Time domain signal
        self.receivers_reference = np.load(pth_str + "habc_ref.npy")

        # Frequency domain signal
        self.receivers_ref_fft = np.load(pth_str + "habc_fft.npy").T
