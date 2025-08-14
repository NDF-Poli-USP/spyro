import firedrake as fire
import numpy as np

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
    path_save : `string`
        Path to save data
    receivers_reference : `array`
        Receiver waveform data in the reference model
    receivers_ref_fft : `array`
        Frequency response at the receivers in the reference model.

    Methods
    -------
    get_reference_signal()
        Acquire the reference signal to compare with the HABC scheme
    '''

    def __init__(self, path_save):
        '''
        Initialize the HABC_Error class.

        Parameters
        ----------
        path_save : `string`
            Path to save data

        Returns
        -------
        None
        '''

        self.path_save = path_save

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
        pth_str = self.path_save + foldername

        # Time domain signal
        self.receivers_reference = np.load(pth_str + "habc_ref.npy")

        # Frequency domain signal
        self.receivers_ref_fft = np.load(pth_str + "habc_fft.npy").T
