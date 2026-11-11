"""Typed parameters of the cells-per-wavelength calculator."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .line_search import MAX_CPW
from .threshold_discretization import (
    ThresholdDiscretization,
    _parse_threshold_discretization,
)
from ...utils.error_management import validate_numeric
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
    "fixed_timestep": "timestep",
}


class CpwParameters(BaseModel):
    """Settings of a cells-per-wavelength calculation.

    Build it directly, or from the calculator parameters dictionary with
    `CpwParameters(**dictionary)`, whose keys are the field names except
    those in :data:`DICTIONARY_KEYS`. Unknown keys are rejected. Enum fields
    accept their string values, and
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
        Default is `VelocityProfileType.HOMOGENEOUS`.
    minimum_velocity : float or None
        Velocity, homogeneous acoustic only.
    accepted_error_threshold : float or None
        Largest accepted receiver error. Exactly one of this and
        ``threshold_discretization`` must be given.
    threshold_discretization : ThresholdDiscretization or None
        Discretization whose error is used as the threshold.
    cpw_max : float
        Largest cells-per-wavelength evaluated. Default is :data:`MAX_CPW`.
    search_direction : CpwSearchDirection
        Default is ``CpwSearchDirection.FORWARD``.
    timestep_calculation : TimeStepCalculationType
        How the time step of each forward solve is chosen. Default is
        `TimeStepCalculationType.FIXED` when `fixed_timestep` is given, else
        `TimeStepCalculationType.ESTIMATE`.
    fixed_timestep : float or None
        Time step of every solve, given only with
        ``TimeStepCalculationType.FIXED``. Default is None.
    testing : bool
        Use a reduced acquisition, for faster tests. Default is False.
    save_reference : bool
        Save the reference solution to ``reference_solution_file``.
    load_reference : bool
        Load the reference solution from ``reference_solution_file``.
    reference_solution_file : str or None
        `.npz` file the reference is saved to or loaded from, with its
        metadata. Default is None, for
        :meth:`CpwProblem.default_reference_file`.
    save_shot_records : bool
        Save the shot record of every search solve to
        ``test_shot_record<cpw>``. Default is False.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=lambda name: DICTIONARY_KEYS.get(name, name),
        validate_by_name=True,
    )

    source_frequency: float
    mesh_frequency_factor: float
    fem_method: str
    dimension: int
    desired_degree: int
    cpw_initial: float
    cpw_accuracy: float
    wave_type: WaveType = WaveType.ISOTROPIC_ACOUSTIC
    velocity_profile_type: VelocityProfileType = VelocityProfileType.HOMOGENEOUS
    minimum_velocity: float | None = None
    accepted_error_threshold: float | None = None
    threshold_discretization: ThresholdDiscretization | None = None
    cpw_max: float = MAX_CPW
    search_direction: CpwSearchDirection = CpwSearchDirection.FORWARD
    # Declared before timestep_calculation, whose default depends on it
    fixed_timestep: float | None = None
    timestep_calculation: TimeStepCalculationType = Field(
        default_factory=lambda data: (
            TimeStepCalculationType.ESTIMATE
            if data["fixed_timestep"] is None
            else TimeStepCalculationType.FIXED
        )
    )
    testing: bool = False
    save_reference: bool = False
    load_reference: bool = False
    reference_solution_file: str | None = None
    save_shot_records: bool = False

    @field_validator("threshold_discretization", mode="before")
    @classmethod
    def _parse_threshold(
        cls, value: ThresholdDiscretization | bool | dict | None
    ) -> ThresholdDiscretization | None:
        """Convert the `threshold_discretization` input.

        Parameters
        ----------
        value : ThresholdDiscretization, bool, dict or None
            See :func:`~.threshold_discretization._parse_threshold_discretization`.

        Returns
        -------
        ThresholdDiscretization or None
            The threshold discretization.
        """
        return _parse_threshold_discretization(value)

    @model_validator(mode="after")
    def _check_settings(self) -> "CpwParameters":
        """Check the settings.

        Returns
        -------
        CpwParameters
            The checked parameters.

        Raises
        ------
        ValueError
            If `mesh_frequency_factor` is not positive, both or neither of
            `accepted_error_threshold` and `threshold_discretization` are
            given, `fixed_timestep` is given without
            `TimeStepCalculationType.FIXED` or missing with it, or
            `reference_solution_file` does not end in `.npz`.
        """
        validate_numeric(
            "mesh_frequency_factor", self.mesh_frequency_factor, lower_bound=0.0
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
                "Give a timestep exactly when timestep_calculation is "
                f"{TimeStepCalculationType.FIXED.value!r}."
            )
        # np.savez appends .npz to any other name, which a load would miss
        if self.reference_solution_file is not None and not (
            self.reference_solution_file.endswith(".npz")
        ):
            raise ValueError(
                "reference_solution_file must be a .npz file, got "
                f"{self.reference_solution_file!r}."
            )
        return self

    def require(self, name: str) -> object:
        """Return a field that the chosen problem needs.

        Parameters
        ----------
        name : str
            Field name, e.g. ``"minimum_velocity"``.

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
