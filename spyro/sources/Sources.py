import math
import numpy as np
from scipy.signal import butter, filtfilt
from spyro.receivers.dirac_delta_projector import Delta_projector


class Sources(Delta_projector):
    """Methods that inject a wavelet into a mesh

    ...

    Attributes
    ----------
    mesh : Firedrake.mesh
        mesh where receivers are located
    V: Firedrake.FunctionSpace object
        The space of the finite elements
    my_ensemble: Firedrake.ensemble_communicator
        An ensemble communicator
    dimension: int
        The dimension of the space
    degree: int
        Degree of the function space
    source_locations: list
        List of tuples containing all source locations
    num_sources: int
        Number of sources
    quadrilateral: boolean
        Boolean that specifies if cells are quadrilateral
    is_local: list of booleans
        List that checks if sources are present in cores
        spatial paralelism
    wavelet: list of floats
        Values at timesteps of wavelet used in the simulation

    Methods
    -------
    build_maps()
        Calculates and stores tabulations for interpolation
    interpolate(field)
        Interpolates field value at receiver locations
    apply_source(rhs_forcing, value)
        Applies value at source locations in rhs_forcing operator
    """

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
        Sources: :class: 'Source' object

        """
        super().__init__(wave_object)

        self.point_locations = wave_object.source_locations
        self.number_of_points = wave_object.number_of_sources
        self.amplitude = wave_object.amplitude
        self.is_local = [0] * self.number_of_points
        self.current_sources = None
        self.update_wavelet(wave_object)
        if np.isscalar(self.amplitude) or (self.amplitude.size <= 3):
            self.integral = False
            self.build_maps(order=0)
        else:
            self.integral = True
            self.build_maps(order=1)
        self.update_wavelet(wave_object)

    def update_wavelet(self, wave_object):
        self.wavelet = full_ricker_wavelet(
            dt=wave_object.dt,
            final_time=wave_object.final_time,
            frequency=wave_object.frequency,
            delay=wave_object.delay,
            delay_type=wave_object.delay_type,
            integral=self.integral
        )

    def apply_source(self, rhs_forcing, step):
        """Applies source in a assembled right hand side.

        Parameters
        ----------
        rhs_forcing: Firedrake.Function
            The right hand side of the wave equation
        step: int
            Time step (index of the wavelet array)

        Returns
        -------
        rhs_forcing: Firedrake.Function
            The right hand side of the wave equation with the source applied
        """
        for source_id in range(self.number_of_points):
            if self.is_local[source_id] and source_id in self.current_sources:
                for i in range(len(self.cellNodeMaps[source_id])):
                    rhs_forcing.dat.data_with_halos[
                        int(self.cellNodeMaps[source_id][i])
                    ] = (self.wavelet[step] * np.dot(self.amplitude, self.cell_tabulations[source_id][i]))
            else:
                for i in range(len(self.cellNodeMaps[source_id])):
                    tmp = rhs_forcing.dat.data_with_halos[0]  # noqa: F841

        return rhs_forcing


def timedependentSource(model, t, freq=None, delay=1.5):
    if model["acquisition"]["source_type"] == "Ricker":
        return ricker_wavelet(t, freq, delay=delay)
    else:
        raise ValueError("source not implemented")


def ricker_wavelet(
    t, freq, delay=1.5, delay_type="multiples_of_minimun",
    integral=False
):
    """Creates a Ricker source function with a
    delay in term of multiples of the distance
    between the minimums.

    Parameters
    ----------
    t: float
        Time
    freq: float
        Frequency of the wavelet
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
    tt = (math.pi * freq * t) ** 2
    if integral:
        return t*math.exp((-1.0) * tt)
    else:
        return (1.0 - (2.0) * tt) * math.exp((-1.0) * tt)


def full_ricker_wavelet(
    dt,
    final_time,
    frequency,
    cutoff=None,
    delay=1.5,
    delay_type="multiples_of_minimun",
    integral=False
):
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
    nt = int(final_time / dt) + 1  # number of timesteps
    time = 0.0
    full_wavelet = np.zeros((nt,))
    for t in range(nt):
        full_wavelet[t] = ricker_wavelet(
            time, frequency, delay=delay, delay_type=delay_type, integral=integral
        )
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
