from dataclasses import field, dataclass
import warnings


@dataclass
class TimeAxis:
    initial_time: float
    final_time: float
    dt: float
    current_time: float = field(init=False)
    index: int = field(init=False)
    max_index: int = field(init=False)

    def __post_init__(self):
        if self.final_time < 0.0:
            raise ValueError(f"Negative time of {self.final_time} not valid.")
        if self.dt > 1.0:
            warnings.warn(f"Time step of {self.dt} too big.")
        if self.dt is None:
            warnings.warn(
                "Timestep not given. Will calculate internally when user \
                    attemps to propagate wave."
            )

        self.current_time = self.initial_time
        self.index = 0
        self.max_index = (self.final_time - self.initial_time)/self.dt

    def update(self):
        self.current_time += self.dt
        self.index += 1

    def get_reverse_index(self):
        return self.max_index - self.index

    def can_update(self):
        return self.current_time < self.final_time

    def copy(self):
        return TimeAxis(initial_time=self.initial_time, final_time=self.final_time, dt=self.dt)
