"""Source utilities for injecting wavelets into simulation meshes."""

import math

import numpy as np
from scipy.signal import butter, filtfilt
from spyro.receivers.dirac_delta_projector import Delta_projector
from ..domains.space import create_function_space
from ..utils.typing import WaveType
import firedrake as fire
import ufl


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

    Methods
    -------
    build_maps()
        Calculates and stores tabulations for interpolation
    interpolate(field)
        Interpolates field value at receiver locations
    apply_source(rhs_forcing, value)
        Applies value at source locations in rhs_forcing operator
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
        self.frequency = wave.frequency
        self.delay = wave.delay
        self.delay_type = wave.delay_type
        self.is_local = [0] * self.number_of_points
        self.current_sources = None
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

    def wavelet_expression(self, t: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Return the source wavelet as a UFL expression of the time ``t``.

        This is the continuous counterpart of :attr:`wavelet`, whose entries
        sample the same function at the time steps. Time integrators that
        evaluate the source between time levels, such as the Runge-Kutta
        stages of :mod:`spyro.solvers.time_integration_irksome`, need the
        expression rather than the samples.

        Parameters
        ----------
        t : ufl.core.expr.Expr
            The time, usually a ``firedrake.Constant`` the time integrator
            updates.

        Returns
        -------
        ufl.core.expr.Expr
            The unit-amplitude Ricker wavelet at time ``t``. The amplitude
            enters through the point source, see
            :meth:`point_source_cofunction`.
        """
        return ricker_wavelet_ufl(
            t,
            self.frequency,
            amplitude=1.0,
            delay=self.delay,
            delay_type=self.delay_type,
        )

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

    def point_source_cofunction(self) -> fire.Cofunction:
        """Return the cofunction of the active point sources at unit wavelet.

        The right-hand side contribution of the sources at any time is this
        cofunction scaled by the wavelet value at that time. It is built from
        the vertex-only mesh when the solver uses one and from the tabulated
        basis functions of :class:`Delta_projector` otherwise, so both
        source paths give the same source term.

        Returns
        -------
        firedrake.Cofunction
            The point sources listed in :attr:`current_sources`, weighted
            by :attr:`amplitude`, as a cofunction of the wave function space.

        Raises
        ------
        ValueError
            If no source is active.
        """
        if self.current_sources is None or len(self.current_sources) == 0:
            raise ValueError(
                "Point source assembly requires at least one active source."
            )
        if self.use_vertex_only_mesh:
            return self.source_cofunction()

        source_cofunction = fire.Cofunction(self.function_space.dual())
        for source_id in self.current_sources:
            # ``is_local`` holds the cell containing the source, or ``None``
            # when it lies on another rank, so it is compared to ``None``
            # rather than tested for truth: the first cell has id ``0``.
            if self.is_local[source_id] is None:
                continue
            for node, tabulation in zip(
                self.cellNodeMaps[source_id], self.cell_tabulations[source_id]
            ):
                source_cofunction.dat.data_with_halos[int(node)] = np.dot(
                    self.amplitude, tabulation
                )
        return source_cofunction

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


def ricker_wavelet_ufl(
    t: ufl.core.expr.Expr,
    frequency: float,
    amplitude: float = 1.0,
    delay: float | int = 1.5,
    delay_type: str = "multiples_of_minimum",
) -> ufl.core.expr.Expr:
    """Create a delayed Ricker wavelet as a UFL expression of the time.

    Symbolic counterpart of :func:`ricker_wavelet`: the same function of
    time, but built with UFL operators so that ``t`` can be a
    ``firedrake.Constant`` inside a variational form and be evaluated
    wherever the time integrator needs it.

    Parameters
    ----------
    t : ufl.core.expr.Expr
        Time, typically a ``firedrake.Constant``.
    frequency : float
        Peak frequency of the wavelet.
    amplitude : float, optional
        Amplitude of the wavelet. Default is 1.0.
    delay : float or int, optional
        Delay, in multiples of the distance between the minima of the
        wavelet or in seconds, according to ``delay_type``. Default is 1.5.
    delay_type : str, optional
        ``"multiples_of_minimum"`` (default) or ``"time"``.

    Returns
    -------
    ufl.core.expr.Expr
        Value of the wavelet at time ``t``.

    Raises
    ------
    ValueError
        If ``delay_type`` is not one of the two options.
    """
    if delay_type == "multiples_of_minimum":
        time_delay = delay * math.sqrt(6.0) / (math.pi * frequency)
    elif delay_type == "time":
        time_delay = delay
    else:
        raise ValueError(
            "delay_type must be 'multiples_of_minimum' or 'time', "
            f"got {delay_type!r}."
        )
    tt = (math.pi * frequency * (t - time_delay)) ** 2
    return amplitude * (1.0 - 2.0 * tt) * ufl.exp(-tt)


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
