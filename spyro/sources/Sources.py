import math
import numpy as np
from scipy.signal import butter, filtfilt
from spyro.receivers.dirac_delta_projector import Delta_projector
from ..utils.typing import WaveType
import firedrake as fire


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
            self.build_maps(order=0)
        else:
            self.build_maps(order=1)

    def update_wavelet(self, wave_object):
        self.wavelet = full_ricker_wavelet(
            dt=wave_object.dt,
            final_time=wave_object.final_time,
            frequency=wave_object.frequency,
            delay=wave_object.delay,
            delay_type=wave_object.delay_type,
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

    def source_cofunction(self):
        """Return a cofunction with the source applied into the domain.

        Returns
        -------
        source_cofunction: Firedrake.Cofunction
            A cofunction with the source applied into the domain.
        """
        print("Wave type:", self.wave_type)
        source_mesh = fire.VertexOnlyMesh(
            self.mesh, [self.point_locations[self.current_sources[0]]])
        if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
            V_s = fire.VectorFunctionSpace(source_mesh, "DG", 0)
        elif self.wave_type == WaveType.ISOTROPIC_ACOUSTIC:
            V_s = fire.FunctionSpace(source_mesh, "DG", 0)
        else:
            raise ValueError("Invalid wave type")

        d_s = fire.Function(V_s)
        d_s.assign(1.0)
        source_cofunction = fire.assemble(fire.inner(d_s, fire.TestFunction(V_s)) * fire.dx)
        return fire.Cofunction(
            self.function_space.dual()).interpolate(source_cofunction)


def timedependentSource(model, t, freq=None, amp=1, delay=1.5):
    if model["acquisition"]["source_type"] == "Ricker":
        return ricker_wavelet(t, freq, amp, delay=delay)
    # elif model["acquisition"]["source_type"] == "MMS":
    #     return MMS_time(t)
    else:
        raise ValueError("source not implemented")


def ricker_wavelet(
    t, freq, amp=1.0, delay=1.5, delay_type="multiples_of_minimum"
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
    amp: float
        Amplitude of the wavelet
    delay: float
        Delay in term of multiples of the distance
        between the minimums.
    delay_type: string
        Type of delay. Options are:
        - multiples_of_minimum
        - time

    Returns
    -------
    float
        Value of the wavelet at time t
    """
    if delay_type == "multiples_of_minimum":
        time_delay = delay * math.sqrt(6.0) / (math.pi * freq)
    elif delay_type == "time":
        time_delay = delay
    t = t - time_delay
    # t = t - delay / freq
    tt = (math.pi * freq * t) ** 2
    return amp * (1.0 - (2.0) * tt) * math.exp((-1.0) * tt)


def full_ricker_wavelet(
    dt,
    final_time,
    frequency,
    cutoff=None,
    delay=1.5,
    delay_type="multiples_of_minimum",
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
        - multiples_of_minimum
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
            time, frequency, 1, delay=delay, delay_type=delay_type
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
