"""Class and methods for reading time related inputs."""


class Read_time_axis:
    """Read and validate time-axis settings from ``self.input_dictionary``.

    This helper expects a ``"time_axis"`` section in ``self.input_dictionary``
    and populates simulation-time attributes used by the solver.

    Attributes
    ----------
    time_integrator : str
        Time integration scheme. Currently only ``"central_difference"`` is
        supported.
    initial_time : float
        Initial simulation time. Defaults to ``0.0`` when omitted or when
        explicitly provided as ``None``.
    final_time : float
        Final simulation time. Must be greater than or equal to
        ``initial_time``.
    dt : float
        Time step size.

    Raises
    ------
    KeyError
        If required keys such as ``"time_axis"``, ``"final_time"``, or
        ``"dt"`` are missing.
    ValueError
        If ``final_time < initial_time`` or if an unsupported time integrator
        is provided.
    """

    def __init__(self):
        """Initialize time-axis attributes from the input dictionary.

        Notes
        -----
        This initializer sets defaults for optional entries and validates
        consistency through property setters.
        """
        # some default parameters we might use in the future
        self.input_dictionary["time_axis"].setdefault(
            "time_integration_scheme", "central_difference"
        )
        self.time_integrator = self.input_dictionary["time_axis"][
            "time_integration_scheme"
        ]

        self.input_dictionary["time_axis"].setdefault("initial_time", 0.0)
        self.initial_time = self.input_dictionary["time_axis"]["initial_time"]
        self.final_time = self.input_dictionary["time_axis"]["final_time"]
        self.dt = self.input_dictionary["time_axis"]["dt"]
        self.input_dictionary["time_axis"].setdefault(
            "gradient_sampling_frequency", 99999
        )

    @property
    def initial_time(self):
        """Float: Initial simulation time."""
        return self._initial_time

    @initial_time.setter
    def initial_time(self, value):
        if value is None:
            value = 0.0
        self._initial_time = value

    @property
    def final_time(self):
        """Float: Final simulation time."""
        return self._final_time

    @final_time.setter
    def final_time(self, value):
        if value < self.initial_time:
            raise ValueError(
                f"Final time of {value} lower than initial time of {self.initial_time} "
                "not allowed."
            )

        self._final_time = value

    @property
    def time_integrator(self):
        """str: Time integration scheme."""
        return self._time_integrator

    @time_integrator.setter
    def time_integrator(self, value):
        if value != "central_difference":
            raise ValueError(f"The time integrator of {value} is not implemented yet")
        self._time_integrator = value
