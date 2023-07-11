import math
import numpy as np
from firedrake import *
from scipy.signal import butter, filtfilt
import spyro


class Sources(spyro.receivers.Receivers.Receivers):
    """Methods that inject a wavelet into a mesh"""

    def __init__(self, wave_object):
        """Initializes class and gets all receiver parameters from
        input file.

        Parameters
        ----------
        model: `dictionary`
            Contains simulation parameters and options.
        mesh: a Firedrake.mesh
            2D/3D simplicial mesh read in by Firedrake.Mesh
        V: Firedrake.FunctionSpace object
            The space of the finite elements
        my_ensemble: Firedrake.ensemble_communicator
            An ensemble communicator

        Returns
        -------
        Receivers: :class: 'Receiver' object

        """
        my_ensemble = wave_object.comm

        self.mesh  = wave_object.mesh
        self.space = wave_object.function_space.sub(0)
        self.my_ensemble = my_ensemble
        self.dimension = wave_object.dimension
        self.degree = wave_object.degree

        self.receiver_locations = wave_object.source_locations
        self.num_receivers = wave_object.number_of_sources

        self.cellIDs = None
        self.cellVertices = None
        self.cell_tabulations = None
        self.cellNodeMaps = None
        self.nodes_per_cell = None
        self.is_local = [0]*self.num_receivers
        self.current_source = None
        if wave_object.cell_type == 'quadrilateral':
            self.quadrilateral = True
        else:
            self.quadrilateral = False

        self.build_maps()
    
    def build_maps(self):
        super().build_maps()


    def apply_source(self, rhs_forcing, value):
        """Applies source in a assembled right hand side.
        
        Parameters
        ----------
        rhs_forcing: Firedrake.Function
            The right hand side of the wave equation
        value: float
            The value of the source
            
        Returns
        -------
        rhs_forcing: Firedrake.Function
            The right hand side of the wave equation with the source applied
                """
        for source_id in range(self.num_receivers):
            if self.is_local[source_id] and source_id==self.current_source:
                for i in range(len(self.cellNodeMaps[source_id])):
                    rhs_forcing.dat.data_with_halos[int(self.cellNodeMaps[source_id][i])] = (
                        value * self.cell_tabulations[source_id][i]
                    )
            else: 
                for i in range(len(self.cellNodeMaps[source_id])):
                    tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing


def timedependentSource(model, t, freq=None, amp=1, delay=1.5):
    if model["acquisition"]["source_type"] == "Ricker":
        return ricker_wavelet(t, freq, amp, delay=delay)
    # elif model["acquisition"]["source_type"] == "MMS":
    #     return MMS_time(t)
    else:
        raise ValueError("source not implemented")


def ricker_wavelet(t, freq, amp=1.0, delay=1.5, delay_type="multiples_of_minimun"):
    """Creates a Ricker source function with a
    delay in term of multiples of the distance
    between the minimums.

    Parameters
    ----------
    t: float
        Time
    freq: float
        Frequency of the wavelet
    amp: float
        Amplitude of the wavelet
    delay: float
        Delay in term of multiples of the distance
        between the minimums.
    delay_type: string
        Type of delay. Options are:
        - multiples_of_minimun
        - time

    Returns
    -------
    float
        Value of the wavelet at time t
    """
    if delay_type == "multiples_of_minimun":
        time_delay = delay * math.sqrt(6.0) / (math.pi * freq)
    elif delay_type == "time":
        time_delay = delay
    t = t - time_delay
    # t = t - delay / freq
    return (
        amp
        * (1.0 - (2.0) * (math.pi * freq) * (math.pi * freq) * t * t)
        * math.exp(
            (-1.0 ) * (math.pi * freq) * (math.pi * freq) * t * t
        )
    )


def full_ricker_wavelet(dt, final_time, frequency, amplitude=1.0, cutoff=None, delay = 1.5, delay_type="multiples_of_minimun"):
    """Compute the Ricker wavelet optionally applying low-pass filtering
    using cutoff frequency in Hertz.

    Parameters
    ----------
    dt: float
        Time step
    final_time: float
        Final time
    frequency: float
        Frequency of the wavelet
    amplitude: float
        Amplitude of the wavelet
    cutoff: float
        Cutoff frequency in Hertz
    delay: float
        Delay in term of multiples of the distance
        between the minimums.
    delay_type: string
        Type of delay. Options are:
        - multiples_of_minimun
        - time

    Returns
    -------
    list of float
        list of ricker values at each time step
    """
    nt = int(final_time / dt) + 1 # number of timesteps
    time = 0.0
    full_wavelet = np.zeros((nt,))
    for t in range(nt):
        full_wavelet[t] = ricker_wavelet(time, frequency, amplitude, delay=delay, delay_type=delay_type)
        time += dt
    if cutoff is not None:
        fs = 1.0 / dt
        order = 2
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        full_wavelet = filtfilt(b, a, full_wavelet)
    return full_wavelet
