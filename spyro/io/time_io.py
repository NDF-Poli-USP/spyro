import numpy as np


class Read_time_axis:
    def __init__(self):
        # some default parameters we might use in the future
        self.input_dictionary["time_axis"].setdefault("time_integration_scheme", "central_difference")
        self.time_integrator = self.input_dictionary["time_axis"]["time_integration_scheme"]

        self.input_dictionary["time_axis"].setdefault("initial_time", 0.0)
        self.initial_time = self.input_dictionary["time_axis"]["initial_time"]
        self.final_time = self.input_dictionary["time_axis"]["final_time"]
        self.dt = self.input_dictionary["time_axis"]["dt"]
        self.input_dictionary["time_axis"].setdefault("gradient_sampling_frequency", 99999)
        self.input_dictionary["time_axis"].setdefault("save_forward_solution", True)

    @property
    def initial_time(self):
        return self._initial_time

    @initial_time.setter
    def initial_time(self, value):
        if value is None:
            value = 0.0
        self._initial_time = value

    @property
    def final_time(self):
        return self._final_time

    @final_time.setter
    def final_time(self, value):
        if value < self.initial_time:
            raise ValueError(f"Final time of {value} lower than initial time of {self.initial_time} not allowed.")

        self._final_time = value

    @property
    def time_integrator(self):
        return self._time_integrator

    @time_integrator.setter
    def time_integrator(self, value):
        if value != "central_difference":
            raise ValueError(f"The time integrator of {value} is not implemented yet")
        self._time_integrator = value


def interpolate_time_series(
    values,
    target_dt,
    initial_time=None,
    final_time=None,
):
    """Resample receiver data from one time grid onto another.

    Parameters
    ----------
    values : array_like
        Time series data stored as ``(time, receiver)`` or ``(time,)``.
    target_dt : float
        Desired timestep.
    initial_time : float
        Starting time of the simulation.
    final_time : float
        Final time of the simulation

    Returns
    -------
    numpy.ndarray
        Data interpolated onto the target time grid.
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
    input_was_1d = array.ndim == 1
    if array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim != 2:
        raise ValueError("Time series interpolation expects a 1D or 2D array.")

    num_source_steps = array.shape[0]
    if num_source_steps <= 1:
        raise ValueError("values must contain at least two time samples.")

    source_dt = (final_time - initial_time) / (num_source_steps - 1)

    target_num_steps = int(np.round((final_time - initial_time) / target_dt)) + 1
    if target_num_steps <= 1:
        raise ValueError("target_dt and time interval produce too few target samples.")

    source_times = initial_time + np.arange(num_source_steps) * source_dt
    target_times = initial_time + np.arange(target_num_steps) * target_dt

    interpolated = np.empty((target_num_steps, array.shape[1]), dtype=float)
    for receiver_id in range(array.shape[1]):
        interpolated[:, receiver_id] = np.interp(
            target_times,
            source_times,
            array[:, receiver_id],
            left=array[0, receiver_id],
            right=array[-1, receiver_id],
        )

    if input_was_1d:
        return interpolated[:, 0]
    return interpolated
