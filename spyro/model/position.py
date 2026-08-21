
from dataclasses import dataclass


@dataclass
class Position:
    x: float
    y: float
    z: float | None = None

    def get_position(self) -> tuple:
        if self.z is None:
            return self.x, self.y

        return self.x, self.y, self.z
