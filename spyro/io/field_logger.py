"""Deprecated alias of `spyro.io.simulation_logger`.

`FieldLogger` was renamed to `SimulationLogger`, since it also logs functionals and
the time, not only fields. Import it from `spyro.io.simulation_logger` instead.
"""

import warnings

from .simulation_logger import Field, Functional, SimulationLogger

warnings.warn(
    "`spyro.io.field_logger` is deprecated, use `spyro.io.simulation_logger` and "
    "`SimulationLogger` instead.",
    DeprecationWarning,
    stacklevel=2,
)

FieldLogger = SimulationLogger

__all__ = ["Field", "FieldLogger", "Functional"]
