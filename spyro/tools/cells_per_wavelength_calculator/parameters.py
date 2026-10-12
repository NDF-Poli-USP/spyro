"""Typed parameters of the cells-per-wavelength calculator."""

import dataclasses
import warnings
from dataclasses import dataclass

from .heterogeneous_case import HeterogeneousCase
from .line_search import MAX_CPW
from .threshold_discretization import (
    ThresholdDiscretization,
    _parse_threshold_discretization,
)
from ...utils.error_management import validate_enum
from ...utils.typing import (
    CpwSearchDirection,
    TimeStepCalculationType,
    VelocityProfileType,
    WaveType,
)

#: Parameters dictionary key of each field whose name differs from it.
DICTIONARY_KEYS = {
    "fem_method": "FEM_method_to_evaluate",
    "minimum_velocity": "minimum_velocity_in_the_domain",
    "cpw_initial": "C_initial",
    "cpw_accuracy": "C_accuracy",
    "cpw_max": "C_max",
    "cpw_reference": "C_reference",
    "timestep_calculation": "time-step_calculation",
    "fixed_timestep": "time-step",
}


@dataclass(frozen=True, kw_only=True)
class CpwParameters:
    """Settings of a cells-per-wavelength calculation.

    Build it directly, or from the calculator parameters dictionary with
    :meth:`from_dict`. Enum fields accept their string values, and
    ``threshold_discretization`` accepts what
    :func:`~.threshold_discretization._parse_threshold_discretization`
    does; both are converted on creation.

    Attributes
    ----------
    source_frequency : float
        Ricker peak frequency.
    mesh_frequency_factor : float
        The wavelength of every cells-per-wavelength value is taken at this
        multiple of ``source_frequency``, e.g. 1.0 (the peak frequency), 2.0
        (about 90% of the Ricker energy) or 2.5 (close to its maximum
        frequency). It has no default, since the result depends on it.
    fem_method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    dimension : int
        Spatial dimension. Only 2 is supported.
    desired_degree : int
        Polynomial degree the cpw is calculated for.
    cpw_initial : float
        Starting cells-per-wavelength. Must meet the error threshold for a
        backward search and exceed it for a forward search.
    cpw_accuracy : float
        Resolution of the cells-per-wavelength result.
    wave_type : WaveType
        Wave physics. Default is ``WaveType.ISOTROPIC_ACOUSTIC``.
    velocity_profile_type : VelocityProfileType
        Default is heterogeneous when ``heterogeneous_case`` is given, else
        homogeneous.
    receiver_setup : str or None
        Receiver setup label, kept for older parameter dictionaries. The
        acquisition is set by the problem.
    minimum_velocity : float or None
        Velocity, homogeneous acoustic only.
    p_wave_velocity, s_wave_velocity, density : float or None
        Material, homogeneous elastic only.
    heterogeneous_case : HeterogeneousCase or None
        Domain, material and any acquisition or time axis overrides,
        heterogeneous only.
    reference_degree : int or None
        Polynomial degree of the numerical reference, heterogeneous only.
    cpw_reference : float or None
        Cells-per-wavelength of the numerical reference, heterogeneous only.
    accepted_error_threshold : float or None
        Largest accepted receiver error. Exactly one of this and
        ``threshold_discretization`` must be given.
    threshold_discretization : ThresholdDiscretization or None
        Discretization whose error is used as the threshold.
    cpw_max : float
        Largest cells-per-wavelength evaluated. Default is :data:`MAX_CPW`.
    search_direction : CpwSearchDirection
        Default is ``CpwSearchDirection.BACKWARD``.
    timestep_calculation : TimeStepCalculationType
        How the time step of each forward solve is chosen. Default is
        ``TimeStepCalculationType.EXACT``.
    fixed_timestep : float or None
        Time step of every solve, given only with
        ``TimeStepCalculationType.FIXED``.
    testing : bool
        Use a reduced acquisition, for faster tests. Default is False.
    save_reference : bool
        Save the reference solution to ``reference_solution_file``.
    load_reference : bool
        Load the reference solution from ``reference_solution_file``.
    reference_solution_file : str or None
        File the reference is saved to or loaded from. Default is None, for
        :meth:`CpwProblem.default_reference_file`.
    save_shot_records : bool
        Save the shot record of every search solve to
        ``test_shot_record<cpw>``. Default is False.
    """

    source_frequency: float
    mesh_frequency_factor: float
    fem_method: str
    dimension: int
    desired_degree: int
    cpw_initial: float
    cpw_accuracy: float
    wave_type: WaveType | str = WaveType.ISOTROPIC_ACOUSTIC
    velocity_profile_type: VelocityProfileType | str | None = None
    receiver_setup: str | None = None
    minimum_velocity: float | None = None
    p_wave_velocity: float | None = None
    s_wave_velocity: float | None = None
    density: float | None = None
    heterogeneous_case: HeterogeneousCase | None = None
    reference_degree: int | None = None
    cpw_reference: float | None = None
    accepted_error_threshold: float | None = None
    threshold_discretization: ThresholdDiscretization | bool | dict | None = None
    cpw_max: float = MAX_CPW
    search_direction: CpwSearchDirection | str = CpwSearchDirection.BACKWARD
    timestep_calculation: TimeStepCalculationType | str = (
        TimeStepCalculationType.EXACT
    )
    fixed_timestep: float | None = None
    testing: bool = False
    save_reference: bool = False
    load_reference: bool = False
    reference_solution_file: str | None = None
    save_shot_records: bool = False

    def __post_init__(self) -> None:
        """Convert the enum and threshold fields, and check the settings.

        Raises
        ------
        ValueError
            If ``mesh_frequency_factor`` is not positive, both or neither of
            ``accepted_error_threshold`` and ``threshold_discretization`` are
            given, or ``fixed_timestep`` is given without
            ``TimeStepCalculationType.FIXED`` or missing with it.
        """
        if self.velocity_profile_type is None:
            profile = (
                VelocityProfileType.HOMOGENEOUS
                if self.heterogeneous_case is None
                else VelocityProfileType.HETEROGENEOUS
            )
        else:
            profile = self.velocity_profile_type
        # The dataclass is frozen, so conversions bypass its __setattr__
        for name, value, enum in (
            ("wave_type", self.wave_type, WaveType),
            ("velocity_profile_type", profile, VelocityProfileType),
            ("search_direction", self.search_direction, CpwSearchDirection),
            (
                "timestep_calculation",
                self.timestep_calculation,
                TimeStepCalculationType,
            ),
        ):
            object.__setattr__(self, name, validate_enum(name, value, enum))
        object.__setattr__(
            self,
            "threshold_discretization",
            _parse_threshold_discretization(self.threshold_discretization),
        )

        if self.mesh_frequency_factor <= 0.0:
            raise ValueError(
                "mesh_frequency_factor must be positive, got "
                f"{self.mesh_frequency_factor}."
            )
        if (self.accepted_error_threshold is None) == (
            self.threshold_discretization is None
        ):
            raise ValueError(
                "Give exactly one of accepted_error_threshold and "
                "threshold_discretization."
            )
        is_fixed = self.timestep_calculation is TimeStepCalculationType.FIXED
        if is_fixed != (self.fixed_timestep is not None):
            raise ValueError(
                "Give a time-step exactly when time-step_calculation is "
                f"{TimeStepCalculationType.FIXED.value!r}."
            )

    @classmethod
    def from_dict(cls, dictionary: dict) -> "CpwParameters":
        """Build the parameters from a calculator parameters dictionary.

        Keys are the field names, except those in :data:`DICTIONARY_KEYS`,
        e.g. ``"C_initial"`` for ``cpw_initial``. Unknown keys are ignored
        with a warning.

        Parameters
        ----------
        dictionary : dict
            Calculator parameters dictionary.

        Returns
        -------
        CpwParameters
            The parameters.

        Raises
        ------
        KeyError
            If a required key is missing.
        """
        values = {}
        known_keys = set()
        missing = []
        for field in dataclasses.fields(cls):
            key = DICTIONARY_KEYS.get(field.name, field.name)
            known_keys.add(key)
            if key in dictionary:
                values[field.name] = dictionary[key]
            elif field.default is dataclasses.MISSING:
                missing.append(key)
        if missing:
            raise KeyError(f"Missing cells-per-wavelength parameters {missing}.")
        unknown = sorted(dictionary.keys() - known_keys)
        if unknown:
            warnings.warn(f"Ignoring unknown cells-per-wavelength parameters {unknown}.")
        return cls(**values)

    def require(self, name: str) -> object:
        """Return a field that the chosen problem needs.

        Parameters
        ----------
        name : str
            Field name, e.g. ``"s_wave_velocity"``.

        Returns
        -------
        object
            The field value.

        Raises
        ------
        ValueError
            If the field is None.
        """
        value = getattr(self, name)
        if value is None:
            raise ValueError(
                f"The {self.wave_type.name} {self.velocity_profile_type.value} "
                f"problem needs {DICTIONARY_KEYS.get(name, name)!r}."
            )
        return value
