"""Methods that deal with time related io operations."""

import numpy as np
import numpy.typing as npt

EPS = np.finfo(np.float32).eps


class Read_time_axis:
    """Class that reads time axis related dictionary."""

    def __init__(self):
        # some default parameters we might use in the future
        self.input_dictionary["time_axis"].setdefault(
            "time_integration_scheme", "central_difference"
        )
        self.time_integrator = self.input_dictionary["time_axis"][
            "time_integration_scheme"
        ]

        self.input_dictionary["time_axis"].setdefault("initial_time", 0.0)
        self.initial_time = self.input_dictionary["time_axis"]["initial_time"]
        self._dt = None
        self.final_time = self.input_dictionary["time_axis"]["final_time"]
        self.input_dictionary["time_axis"].setdefault("dt", None)
        self.dt = self.input_dictionary["time_axis"]["dt"]
        self.input_dictionary["time_axis"].setdefault(
            "gradient_sampling_frequency", 99999
        )
        self.input_dictionary["time_axis"].setdefault("save_forward_solution", True)

    @property
    def initial_time(self):
        """Initial simulation time."""
        return self._initial_time

    @initial_time.setter
    def initial_time(self, value):
        if value is None:
            value = 0.0
        self._initial_time = value

    @property
    def final_time(self) -> float:
        """Final simulation time, snapped onto the last time discretization.

        The value last given to the setter is kept as the proposed final
        time, and every change of `dt` snaps it again. Repeated `dt` changes
        therefore do not accumulate snapping error.
        """
        return self._final_time

    @final_time.setter
    def final_time(self, value: float) -> None:
        if value < self.initial_time:
            raise ValueError(
                (
                    f"Final time of {value} lower than initial"
                    f"time of {self.initial_time} not allowed."
                )
            )
        self._proposed_final_time = value
        self._update_time_discretization()

    @property
    def time_integrator(self):
        """Time integration scheme."""
        return self._time_integrator

    @time_integrator.setter
    def time_integrator(self, value):
        if value != "central_difference":
            raise ValueError(f"The time integrator of {value} is not implemented yet")
        self._time_integrator = value

    @property
    def dt(self) -> float | None:
        """Timestep value."""
        return self._dt

    @dt.setter
    def dt(self, value: float | None) -> None:
        self._dt = value
        self._update_time_discretization()

    def _update_time_discretization(self) -> None:
        """Derive the time discretization from the proposed final time and `dt`.

        Sets `number_of_timesteps`, `time_vector` and `final_time`, which is
        the proposed final time snapped onto the last time level when it is
        not a multiple of `dt`. Both setters call this, so the time levels
        always match the current pair.

        Returns
        -------
        None
            `number_of_timesteps`, `time_vector` and `final_time` are updated
            in place. The first two are `None` while `dt` is not set.
        """
        self._final_time = self._proposed_final_time
        if self._dt is None:
            self.number_of_timesteps = None
            self.time_vector = None
        else:
            self.number_of_timesteps = round(self._proposed_final_time / self._dt) + 1
            if not np.isclose(
                self._proposed_final_time / self._dt + 1,
                self.number_of_timesteps,
                rtol=0.0,
                atol=EPS,
            ):
                self._final_time = (self.number_of_timesteps - 1) * self._dt

            self.time_vector = np.linspace(
                0.0, self._final_time, self.number_of_timesteps
            )


def interpolate_time_series(
    values: npt.ArrayLike,
    target_dt: float,
    initial_time: float | None = None,
    final_time: float | None = None,
    time_shift: float = 0.0,
    source_times: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Resample receiver data from one time grid onto another.

    Parameters
    ----------
    values : array_like
        Time series data with time on the first axis, e.g. `(time,)`,
        `(time, receiver)` or `(time, receiver, direction)`.
    target_dt : float
        Desired timestep.
    initial_time : float, optional
        Starting time of the simulation. Default is 0.0.
    final_time : float
        Final time of the simulation.
    time_shift : float, optional
        Target sample `i` is taken at `initial_time + time_shift +
        i * target_dt`. Default is 0.0. Samples past either end of the
        data take its first or last value.
    source_times : array_like, optional
        Times of the samples in `values`, such as the `time_vector` of the
        wave object that computed them. Default is `None`, which assumes the
        samples are evenly spaced from `initial_time` to `final_time`.

    Returns
    -------
    numpy.ndarray
        Data interpolated onto the target time grid, with the same trailing
        shape as `values`.

    Raises
    ------
    ValueError
        If `target_dt` is not positive, `final_time` is missing or lower than
        `initial_time`, `values` has fewer than two samples, the target grid
        has fewer than two samples, or `source_times` is not a 1D array with
        one entry per sample of `values`.
    """
    if target_dt <= 0.0:
        raise ValueError("target_dt must be positive.")

    if initial_time is None:
        initial_time = 0.0
    if final_time is None:
        raise ValueError("final_time must be provided.")
    if final_time < initial_time:
        raise ValueError("final_time must be greater than or equal to initial_time.")

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("Time series interpolation expects at least a 1D array.")
    trailing_shape = array.shape[1:]
    array = array.reshape(array.shape[0], -1)

    num_source_steps = array.shape[0]
    if num_source_steps <= 1:
        raise ValueError("values must contain at least two time samples.")

    if source_times is None:
        source_dt = (final_time - initial_time) / (num_source_steps - 1)
        source_times = initial_time + np.arange(num_source_steps) * source_dt
    else:
        source_times = np.asarray(source_times, dtype=float)
        if source_times.shape != (num_source_steps,):
            raise ValueError(
                f"source_times has shape {source_times.shape}, expected "
                f"({num_source_steps},) to match the samples of values."
            )

    target_num_steps = int(np.round((final_time - initial_time) / target_dt)) + 1
    if target_num_steps <= 1:
        raise ValueError("target_dt and time interval produce too few target samples.")

    target_times = initial_time + time_shift + np.arange(target_num_steps) * target_dt

    interpolated = np.empty((target_num_steps, array.shape[1]), dtype=float)
    for receiver_id in range(array.shape[1]):
        interpolated[:, receiver_id] = np.interp(
            target_times,
            source_times,
            array[:, receiver_id],
            left=array[0, receiver_id],
            right=array[-1, receiver_id],
        )

    return interpolated.reshape((target_num_steps, *trailing_shape))
