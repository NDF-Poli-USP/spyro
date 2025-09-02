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
