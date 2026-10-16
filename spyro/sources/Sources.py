"""Source utilities for injecting wavelets into simulation meshes."""

from functools import cached_property

import numpy as np
from scipy.signal import butter, filtfilt
from spyro.receivers.dirac_delta_projector import Delta_projector
from ..domains.space import create_function_space
from ..utils.typing import WaveType
import firedrake as fire


class Sources(Delta_projector):
    """Inject a wavelet into a mesh.

    ...

    Attributes
    ----------
    mesh : Firedrake.mesh
        mesh where receivers are located
    V : Firedrake.FunctionSpace object
        The space of the finite elements
    my_ensemble : Firedrake.ensemble_communicator
        An ensemble communicator
    dimension : int
        The dimension of the space
    degree : int
        Degree of the function space
    source_locations : list
        List of tuples containing all source locations
    num_sources : int
        Number of sources
    quadrilateral : boolean
        Boolean that specifies if cells are quadrilateral
    is_local : list of booleans
        List that checks if sources are present in cores
        spatial paralelism
    wavelet : list of floats
        Values at timesteps of wavelet used in the simulation
    time_function : Firedrake.Function
        Value of the wavelet at the current time step, in the real space
        ``R``. Built on first use.
    spatial_function : Firedrake.Function
        Spatial part :math:`q_h` of the active point sources, such that
        :math:`\\int q_h v \\, dx` reproduces :meth:`source_cofunction`.
        Built on first use.
    source_measure : ufl.Measure or None
        Measure of the time-step form the source term was put in by
        :meth:`source_term`, also used by :meth:`update_spatial_function`.

    Methods
    -------
    build_maps()
        Calculates and stores tabulations for interpolation
    interpolate(field)
        Interpolates field value at receiver locations
    apply_source(rhs_forcing, value)
        Applies value at source locations in rhs_forcing operator
    source_term(test_function, measure)
        Returns the point-source term of a time-step form
    update_time_function(step)
        Sets time_function to the wavelet at a time step
    update_spatial_function(source_cofunction, solver_parameters)
        Sets spatial_function from the point-source cofunction
    """

    def __init__(self, wave):
        """Initialize the class and load source parameters.

        Parameters
        ----------
        model : `dictionary`
            Contains simulation parameters and options.
        mesh : a Firedrake.mesh
            2D/3D simplicial mesh read in by Firedrake.Mesh
        V : Firedrake.FunctionSpace object
            The space of the finite elements
        my_ensemble : Firedrake.ensemble_communicator
            An ensemble communicator

        Returns
        -------
        Sources : :class: 'Source' object
        """
        super().__init__(wave)

        self.point_locations = wave.source_locations
        self.number_of_points = wave.number_of_sources
        self.amplitude = wave.amplitude
        self.is_local = [0] * self.number_of_points
        self.current_sources = None
        self.source_measure = None
        if wave.analysis == "transient":
            self.update_wavelet(wave)
        if np.isscalar(self.amplitude) or (self.amplitude.size <= 3):
            self.build_maps(order=0)
        else:
            self.build_maps(order=1)

    def update_wavelet(self, wave):
        """Update the cached wavelet from the current wave settings."""
        self.wavelet = full_ricker_wavelet(
            dt=wave.dt,
            final_time=wave.final_time,
            frequency=wave.frequency,
            delay=wave.delay,
            delay_type=wave.delay_type,
        )

    @cached_property
    def time_function(self) -> fire.Function:
        """Value of the wavelet at the current time step.

        Returns
        -------
        firedrake.Function
            Function in the real space ``R``, set by
            :meth:`update_time_function`.
        """
        return fire.Function(
            fire.FunctionSpace(self.mesh, "R", 0), name="source time function"
        )

    @cached_property
    def spatial_function(self) -> fire.Function:
        """Spatial part of the active point sources.

        Returns
        -------
        firedrake.Function
            Function in :attr:`function_space`, set by
            :meth:`update_spatial_function`.
        """
        return fire.Function(
            self.function_space, name="source spatial function"
        )

    def source_term(self, test_function: object, measure: object) -> object:
        """Return the point-source term of a time-step form.

        The term is :math:`s(t) \\int q_h v`, with :math:`s` the
        :attr:`time_function` and :math:`q_h` the :attr:`spatial_function`.
        ``measure`` is kept, so that :meth:`update_spatial_function` solves
        with the mass matrix of the same quadrature, which is what makes the
        term reproduce :meth:`source_cofunction`.

        Parameters
        ----------
        test_function : ufl.Argument
            Test function of the pressure in the form.
        measure : ufl.Measure
            Cell measure of the form, with its quadrature.

        Returns
        -------
        ufl.Form
            The source term, to be added to the right-hand side.
        """
        self.source_measure = measure
        return self.time_function * self.spatial_function * test_function \
            * measure

    def update_time_function(self, step: int) -> None:
        """Set :attr:`time_function` to the wavelet at ``step``.

        Parameters
        ----------
        step : int
            Time step (index of the wavelet array).

        Returns
        -------
        None
            :attr:`time_function` is updated in place.
        """
        self.time_function.assign(self.wavelet[step])

    def update_spatial_function(self, source_cofunction: fire.Cofunction,
                                solver_parameters: dict | None) -> None:
        r"""Set :attr:`spatial_function` from the point-source cofunction.

        The term built by :meth:`source_term` is
        :math:`s(t) \int q_h v \, dx`, with :math:`s` the
        :attr:`time_function` and :math:`q_h` the :attr:`spatial_function`.
        This solves :math:`M q_h = q` with the mass matrix of the measure of
        that term, so the term assembles back to :math:`s(t)\, q`, and a
        pyadjoint tape records one scalar per time step for the source
        instead of a cofunction.

        Parameters
        ----------
        source_cofunction : firedrake.Cofunction
            The point source :math:`q`, from :meth:`source_cofunction`.
        solver_parameters : dict or None
            Solver parameters for the mass solve.

        Returns
        -------
        None
            :attr:`spatial_function` is updated in place.

        Raises
        ------
        ValueError
            If :meth:`source_term` has not been called yet.
        """
        if self.source_measure is None:
            raise ValueError(
                "No form holds the point-source term: call source_term first."
            )
        V = self.function_space
        mass = fire.TrialFunction(V) * fire.TestFunction(V) \
            * self.source_measure
        fire.solve(mass == source_cofunction, self.spatial_function,
                   solver_parameters=solver_parameters)

    def apply_source(self, rhs_forcing, step):
        """Apply the source to an assembled right-hand side.

        Parameters
        ----------
        rhs_forcing : Firedrake.Function
            The right hand side of the wave equation
        step : int
            Time step (index of the wavelet array)

        Returns
        -------
        rhs_forcing : Firedrake.Function
            The right hand side of the wave equation with the source applied
        """
        for source_id in range(self.number_of_points):
            if self.is_local[source_id] and source_id in self.current_sources:
                for i in range(len(self.cellNodeMaps[source_id])):
                    rhs_forcing.dat.data_with_halos[
                        int(self.cellNodeMaps[source_id][i])
                    ] = self.wavelet[step] * np.dot(
                        self.amplitude, self.cell_tabulations[source_id][i]
                    )
            else:
                for i in range(len(self.cellNodeMaps[source_id])):
                    tmp = rhs_forcing.dat.data_with_halos[0]  # noqa: F841

        return rhs_forcing

    def source_cofunction(self):
        """Return a cofunction with the source applied into the domain.

        Returns
        -------
        source_cofunction : Firedrake.Cofunction
            A cofunction with the source applied into the domain.
        """
        if self.current_sources is None or len(self.current_sources) == 0:
            raise ValueError(
                "VertexOnlyMesh source assembly requires at least one active source."
            )

        source_locations = [
            self.point_locations[source_id] for source_id in self.current_sources
        ]
        source_mesh = fire.VertexOnlyMesh(self.mesh, source_locations)
        if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
            V_s = create_function_space(source_mesh, "DG0", 0, dim=self.dimension)
            source_value = fire.Function(V_s)
            if source_value.dat.data.shape[0] > 0:
                source_value.dat.data[:] = self.amplitude
            source_form = fire.inner(source_value, fire.TestFunction(V_s)) * fire.dx
        elif self.wave_type == WaveType.ISOTROPIC_ACOUSTIC:
            V_s = create_function_space(source_mesh, "DG0", 0)
            source_value = fire.Function(V_s)
            source_value.assign(float(self.amplitude))
            source_form = source_value * fire.TestFunction(V_s) * fire.dx
        else:
            raise ValueError("Invalid wave type")

        return fire.Cofunction(self.function_space.dual()).interpolate(
            fire.assemble(source_form)
        )


def timedependentSource(model, t, freq=None, amp=1, delay=1.5):
    """Return the configured time-dependent source value.

    Parameters
    ----------
    model : dict
        Simulation configuration dictionary.
    t : float
        Current time.
    freq : float, optional
        Source frequency.
    amp : float, default=1
        Source amplitude.
    delay : float, default=1.5
        Delay multiplier for the source wavelet.

    Returns
    -------
    float
        Source amplitude evaluated at time ``t``.
    """
    if model["acquisition"]["source_type"] == "Ricker":
        return ricker_wavelet(t, freq, amp, delay=delay)
    # elif model["acquisition"]["source_type"] == "MMS":
    #     return MMS_time(t)
    else:
        raise ValueError("source not implemented")


def ricker_wavelet(
    t: float,
    frequency: float,
    amplitude: float = 1.0,
    delay: float | int = 1.5,
    delay_type: str = "multiples_of_minimum",
):
    """Create a delayed Ricker source function.

    The delay is expressed in either multiples of the distance between minima or
    in time.

    Parameters
    ----------
    t : float
        Time
    frequency : float
        Frequency of the wavelet
    amplitude : float, optional
        Amplitude of the wavelet. Default value of 1.0.
    delay : float or int
        Delay in term of multiples of the distance
        between the minimums or in seconds.
    delay_type : string
        Type of delay. Options are:
        - multiples_of_minimum
        - time

    Returns
    -------
    float
        Value of the wavelet at time t
    """
    if delay_type == "multiples_of_minimum":
        time_delay = delay * np.sqrt(6.0) / (np.pi * frequency)
    elif delay_type == "time":
        time_delay = delay
    t = t - time_delay
    tt = (np.pi * frequency * t) ** 2
    return amplitude * (1.0 - (2.0) * tt) * np.exp((-1.0) * tt)


def full_ricker_wavelet(
    dt,
    final_time,
    frequency,
    cutoff=None,
    delay=1.5,
    delay_type="multiples_of_minimum",
):
    """Compute the Ricker wavelet, optionally applying low-pass filtering.

    Cutoff frequency in Hertz.

    Parameters
    ----------
    dt : float
        Time step
    final_time : float
        Final time
    frequency : float
        Frequency of the wavelet
    cutoff : float
        Cutoff frequency in Hertz
    delay : float
        Delay in term of multiples of the distance
        between the minimums.
    delay_type : string
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


def ricker_integral(
    frequency: float,
    t: float,
    time_delay: float,
):
    """Get source time function (integral of Ricker wavelet)."""
    a = np.pi * frequency * (t - time_delay)
    return (t - time_delay) * np.exp(-(a**2))
