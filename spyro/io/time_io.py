"""Methods that deal with time related io operations."""

from dataclasses import dataclass, field

import numpy as np

from ..utils.error_management import validate_enum
from ..utils.typing import TimeIntegrationScheme


@dataclass(frozen=True)
class IrksomeOptions:
    """Options of the Irksome time integration, read from ``time_axis["irksome"]``.

    Parameters
    ----------
    tableau : str or object
        Runge-Kutta method. Either a name from the catalogue of
        :mod:`spyro.solvers.time_integration_irksome` (``"rk4"``,
        ``"classic_nystrom4"``, ``"gauss_legendre"``, ``"radau_iia"``,
        ``"lobatto_iiia"``, ``"lobatto_iiic"``, ``"backward_euler"``,
        ``"alexander"``, ``"qin_zhang"``) or an Irksome tableau object
        (a ``ButcherTableau`` or a ``NystromTableau``).
    stages : int, optional
        Number of stages of the collocation families (``gauss_legendre``,
        ``radau_iia``, ``lobatto_iiia``, ``lobatto_iiic``). ``None`` takes
        the smallest number of stages the family admits. Not accepted for
        methods with a fixed number of stages.
    bc_type : str, optional
        How Irksome imposes strong boundary conditions on the stages:
        ``"DAE"``, ``"dDAE"`` or ``"ODE"``. ``None`` picks ``"dDAE"`` for
        explicit tableaux and ``"DAE"`` otherwise.
    solver_parameters : dict, optional
        PETSc options of the stage system. ``None`` picks defaults suited to
        the tableau: a block forward substitution with the mass-matrix solver
        of the spatial method for explicit tableaux, a matrix-free conjugate
        gradient solve for diagonally implicit ones and a direct solve of the
        coupled stages for fully implicit ones. A dictionary replaces the
        defaults entirely.
    """

    tableau: object = "rk4"
    stages: int | None = None
    bc_type: str | None = None
    solver_parameters: dict | None = field(default=None)

    #: Boundary condition types Irksome's Nystrom steppers know.
    BC_TYPES = ("DAE", "dDAE", "ODE")

    @classmethod
    def from_dictionary(cls, dictionary: dict) -> "IrksomeOptions":
        """Build the options from the ``time_axis["irksome"]`` entry.

        Parameters
        ----------
        dictionary : dict
            The user's ``time_axis["irksome"]`` dictionary. Missing keys take
            their defaults.

        Returns
        -------
        IrksomeOptions
            The validated options.

        Raises
        ------
        ValueError
            If the dictionary holds an unknown key or an invalid value.
        """
        known = {"tableau", "stages", "bc_type", "solver_parameters"}
        unknown = set(dictionary) - known
        if unknown:
            raise ValueError(
                "Unknown time_axis['irksome'] options "
                f"{sorted(unknown)}; expected a subset of {sorted(known)}."
            )
        options = cls(**dictionary)
        if isinstance(options.tableau, str) and not options.tableau:
            raise ValueError("time_axis['irksome']['tableau'] cannot be empty.")
        if options.stages is not None and (
            not isinstance(options.stages, int) or options.stages < 1
        ):
            raise ValueError(
                "time_axis['irksome']['stages'] must be a positive integer, "
                f"got {options.stages!r}."
            )
        if options.bc_type is not None and options.bc_type not in cls.BC_TYPES:
            raise ValueError(
                f"time_axis['irksome']['bc_type'] must be one of {cls.BC_TYPES}, "
                f"got {options.bc_type!r}."
            )
        if options.solver_parameters is not None and not isinstance(
            options.solver_parameters, dict
        ):
            raise ValueError(
                "time_axis['irksome']['solver_parameters'] must be a "
                f"dictionary, got {type(options.solver_parameters).__name__}."
            )
        return options


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
        self.input_dictionary["time_axis"].setdefault("irksome", {})
        self.irksome_options = IrksomeOptions.from_dictionary(
            self.input_dictionary["time_axis"]["irksome"]
        )

        self.input_dictionary["time_axis"].setdefault("initial_time", 0.0)
        self.initial_time = self.input_dictionary["time_axis"]["initial_time"]
        self.final_time = self.input_dictionary["time_axis"]["final_time"]
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
    def final_time(self):
        """Final simulation time."""
        return self._final_time

    @final_time.setter
    def final_time(self, value):
        if value < self.initial_time:
            raise ValueError(
                (
                    f"Final time of {value} lower than initial"
                    f"time of {self.initial_time} not allowed."
                )
            )

        self._final_time = value

    @property
    def time_integrator(self):
        """Time integration scheme, a :class:`~spyro.utils.typing.TimeIntegrationScheme`."""
        return self._time_integrator

    @time_integrator.setter
    def time_integrator(self, value):
        self._time_integrator = validate_enum(
            "time_integration_scheme", value, TimeIntegrationScheme
        )


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
        Final time of the simulation.

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
