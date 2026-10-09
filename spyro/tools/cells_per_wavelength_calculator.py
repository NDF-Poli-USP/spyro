"""Find the minimum cells-per-wavelength meshing parameter for a wave physics."""

import copy
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import spyro
from .error_measure import MeasureError
from .input_models import (
    create_elastic_model_for_meshing_parameter_2D_homogeneous,
    create_initial_model_for_meshing_parameter,
    set_mesh_type,
)
from ..io.time_io import interpolate_time_series
from ..mpi.spyro_mpi import SpyroEnsemble
from ..solvers.wave import Wave
from ..utils.error_management import validate_enum
from ..utils.typing import (
    CpwSearchDirection,
    TimeStepCalculationType,
    VelocityProfileType,
    WaveType,
)

#: Fraction of the current cpw added or removed per coarse search step.
COARSE_STEP_FRACTION = 0.1

#: Default largest cells-per-wavelength evaluated by the line searches.
MAX_CPW = 5.0


@dataclass(frozen=True)
class CpwEvaluation:
    """One forward solve of the cells-per-wavelength search.

    Attributes
    ----------
    cpw : float
        Cells-per-wavelength used to build the mesh.
    dt : float
        Time step used in the forward solve.
    error : float
        Receiver error against the reference solution.
    runtime : float
        Wall-clock time of the forward solve, in seconds.
    """

    cpw: float
    dt: float
    error: float
    runtime: float


@dataclass(frozen=True)
class ThresholdCase:
    """Discretization whose receiver error sets the accepted error threshold.

    Attributes
    ----------
    method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    degree : int
        Spatial polynomial degree.
    cpw : float
        Cells-per-wavelength of the mesh.
    """

    method: str = "spectral_quadrilateral"
    degree: int = 4
    cpw: float = 2.0


def _parse_threshold_case(
    value: "bool | dict | ThresholdCase | None",
) -> ThresholdCase | None:
    """Convert the ``"threshold_case"`` parameter to a threshold case.

    Parameters
    ----------
    value : bool, dict, ThresholdCase or None
        None or False for no threshold case, True for the default
        :class:`ThresholdCase`, or a dict of :class:`ThresholdCase` fields
        overriding the defaults.

    Returns
    -------
    ThresholdCase or None
        The threshold case, or None if the threshold is given directly.

    Raises
    ------
    TypeError
        If ``value`` is of an unsupported type, or a dict has unknown keys.
    """
    if value is None or value is False:
        return None
    if value is True:
        return ThresholdCase()
    if isinstance(value, ThresholdCase):
        return value
    if isinstance(value, dict):
        return ThresholdCase(**value)
    raise TypeError(
        "threshold_case must be a bool, a dict or a ThresholdCase, "
        f"got {type(value).__name__}."
    )


class CpwPhysics(ABC):
    """Wave-physics-specific part of the cells-per-wavelength calculator.

    A new physics is supported by subclassing this and registering it in
    :data:`CPW_PHYSICS` under its :class:`~spyro.utils.typing.WaveType`.

    Attributes
    ----------
    parameters_dictionary : dict
        The calculator parameters dictionary.
    wave_class : type[Wave]
        Wave solver class built for every forward solve.
    has_displacement : bool
        Whether receiver data has a trailing direction axis.
    """

    wave_class: type[Wave]
    has_displacement: bool

    def __init__(self, parameters_dictionary: dict) -> None:
        """Store the parameters dictionary.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        """
        self.parameters_dictionary = parameters_dictionary

    @property
    @abstractmethod
    def wavelength_velocity(self) -> float:
        """Velocity whose wavelength the cells-per-wavelength refers to.

        Returns
        -------
        float
            The minimum propagation velocity of the physics.
        """

    @abstractmethod
    def build_dictionary(
        self, calculator: "MeshingParameterCalculator", method: str, degree: int
    ) -> dict:
        """Build the model dictionary of a wave object.

        Parameters
        ----------
        calculator : MeshingParameterCalculator
            The calculator, providing the shared search settings.
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :attr:`wave_class`.
        """

    @abstractmethod
    def setup_homogeneous_wave(self, wave: Wave, edge_length: float) -> None:
        """Build the mesh and the constant material of a homogeneous model.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        edge_length : float
            Mesh edge length.
        """

    @abstractmethod
    def analytical_solution(self, wave: Wave) -> np.ndarray:
        """Compute the analytical receiver data of the homogeneous model.

        Parameters
        ----------
        wave : Wave
            Wave object holding the acquisition and time axis.

        Returns
        -------
        np.ndarray
            Receiver data sampled at ``wave.dt``, shaped like
            ``wave.forward_solution_receivers``.
        """


class AcousticCpwPhysics(CpwPhysics):
    """Isotropic acoustic physics for the cells-per-wavelength calculator."""

    wave_class = spyro.AcousticWave
    has_displacement = False

    @property
    def wavelength_velocity(self) -> float:
        """Minimum acoustic velocity in the domain.

        Returns
        -------
        float
            The ``"minimum_velocity_in_the_domain"`` parameter.
        """
        return self.parameters_dictionary["minimum_velocity_in_the_domain"]

    def build_dictionary(
        self, calculator: "MeshingParameterCalculator", method: str, degree: int
    ) -> dict:
        """Build the acoustic model dictionary.

        Parameters
        ----------
        calculator : MeshingParameterCalculator
            The calculator, providing the shared search settings.
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :class:`spyro.AcousticWave`.
        """
        dictionary = create_initial_model_for_meshing_parameter(calculator)
        # The input model builders use the calculator's method and degree
        dictionary["options"]["method"] = method
        dictionary["options"]["degree"] = degree
        dictionary["mesh"]["mesh_type"] = set_mesh_type(method)
        return dictionary

    def setup_homogeneous_wave(self, wave: Wave, edge_length: float) -> None:
        """Build the mesh and set a constant velocity.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        edge_length : float
            Mesh edge length.
        """
        wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
        wave.set_initial_velocity_model(constant=self.wavelength_velocity)

    def analytical_solution(self, wave: Wave) -> np.ndarray:
        """Compute the acoustic analytical receiver data.

        Parameters
        ----------
        wave : Wave
            Wave object holding the acquisition and time axis.

        Returns
        -------
        np.ndarray
            Pressure at the receivers, shaped ``(n_time, n_receivers)``.
        """
        velocity = self.wavelength_velocity
        num_t = round(wave.final_time / wave.dt) + 1
        analytical = np.zeros((num_t, wave.number_of_receivers))
        source = np.asarray(wave.source_locations[0])
        for i, receiver in enumerate(wave.receiver_locations):
            offset = np.linalg.norm(np.asarray(receiver) - source)
            analytical[:, i] = spyro.utils.nodal_homogeneous_analytical(
                wave, offset, velocity
            )
        return analytical / velocity**2


class IsotropicElasticCpwPhysics(CpwPhysics):
    """Isotropic elastic physics for the cells-per-wavelength calculator.

    The parameters dictionary must hold ``"p_wave_velocity"``,
    ``"s_wave_velocity"`` and ``"density"``. The wavelength is the
    S-wavelength.
    """

    wave_class = spyro.IsotropicWave
    has_displacement = True

    @property
    def wavelength_velocity(self) -> float:
        """S-wave velocity, the slowest elastic body wave.

        Returns
        -------
        float
            The ``"s_wave_velocity"`` parameter.
        """
        return self.parameters_dictionary["s_wave_velocity"]

    def build_dictionary(
        self, calculator: "MeshingParameterCalculator", method: str, degree: int
    ) -> dict:
        """Build the isotropic elastic model dictionary.

        Parameters
        ----------
        calculator : MeshingParameterCalculator
            The calculator, providing the shared search settings.
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :class:`spyro.IsotropicWave`.

        Raises
        ------
        NotImplementedError
            For heterogeneous or 3D models.
        """
        if calculator.velocity_profile_type is not VelocityProfileType.HOMOGENEOUS:
            raise NotImplementedError(
                "Elastic cpw calculation only supports homogeneous models."
            )
        if calculator.dimension != 2:
            raise NotImplementedError("Elastic cpw calculation only supports 2D.")

        frequency = calculator.source_frequency
        dt = calculator.fixed_timestep
        if dt is None:
            # Fine enough that interpolating onto the solver dt is negligible
            dt = 1.0 / (500.0 * frequency)

        return create_elastic_model_for_meshing_parameter_2D_homogeneous(
            method=method,
            degree=degree,
            frequency=frequency,
            p_wave_velocity=self.parameters_dictionary["p_wave_velocity"],
            s_wave_velocity=self.wavelength_velocity,
            density=self.parameters_dictionary["density"],
            dt=dt,
            reduced=calculator.reduced_obj_for_testing,
        )

    def setup_homogeneous_wave(self, wave: Wave, edge_length: float) -> None:
        """Build a periodic mesh; the material comes from the dictionary.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        edge_length : float
            Mesh edge length.
        """
        wave.set_mesh(
            input_mesh_parameters={"edge_length": edge_length, "periodic": True}
        )

    def analytical_solution(self, wave: Wave) -> np.ndarray:
        """Compute the 2D point-force analytical receiver displacement.

        Parameters
        ----------
        wave : Wave
            Wave object holding the acquisition and time axis.

        Returns
        -------
        np.ndarray
            Displacement at the receivers, shaped
            ``(n_time, n_receivers, 2)``.

        Raises
        ------
        ValueError
            If the source amplitude is not aligned with a single axis.
        """
        amplitude = np.asarray(wave.amplitude, dtype=float)
        force_directions = np.flatnonzero(amplitude)
        if len(force_directions) != 1:
            raise ValueError(
                "The analytical elastic solution needs a source amplitude "
                f"along a single axis, got {amplitude}."
            )
        force_direction = int(force_directions[0])

        num_t = round(wave.final_time / wave.dt) + 1
        analytical = np.zeros((num_t, wave.number_of_receivers, wave.dimension))
        source = np.asarray(wave.source_locations[0])
        for i, receiver in enumerate(wave.receiver_locations):
            components = spyro.utils.analytical_solution_elastic(
                source_type="force_source",
                offsets=source - np.asarray(receiver),
                p_wave_velocity=self.parameters_dictionary["p_wave_velocity"],
                s_wave_velocity=self.wavelength_velocity,
                density=self.parameters_dictionary["density"],
                amplitude=amplitude[force_direction],
                frequency=wave.frequency,
                time_delay=wave.delay,
                final_time=wave.final_time,
                dt=wave.dt,
                force_direction=force_direction,
                dimension=wave.dimension,
            )
            analytical[:, i, :] = np.column_stack(components)
        return analytical


#: Cells-per-wavelength physics, selected by the ``"wave_type"`` parameter.
CPW_PHYSICS: dict[WaveType, type[CpwPhysics]] = {
    WaveType.ISOTROPIC_ACOUSTIC: AcousticCpwPhysics,
    WaveType.ISOTROPIC_ELASTIC: IsotropicElasticCpwPhysics,
}


class MeshingParameterCalculator:
    """Calculate the minimum meshing parameter C (cells-per-wavelength).

    The physics is chosen by the optional ``"wave_type"`` parameter, a
    :class:`~spyro.utils.typing.WaveType` defaulting to
    ``WaveType.ISOTROPIC_ACOUSTIC``. See :data:`CPW_PHYSICS`.

    Attributes
    ----------
    parameters_dictionary : dict
        All the parameters needed for the calculation.
    wave_type : WaveType
        The wave physics being evaluated.
    physics : CpwPhysics
        The physics-specific part of the calculation.
    source_frequency : float
        Source (Ricker peak) frequency in Hz.
    mesh_frequency_factor : float
        Ratio of the meshing frequency to ``source_frequency``.
    mesh_frequency : float
        Frequency in Hz whose wavelength every cells-per-wavelength value
        refers to, ``mesh_frequency_factor * source_frequency``.
    minimum_velocity : float
        Velocity whose wavelength the cpw refers to, in km/s. The minimum
        velocity for acoustic, the S-wave velocity for elastic.
    velocity_profile_type : VelocityProfileType
        Homogeneous (analytical reference) or heterogeneous (numerical
        reference).
    velocity_model_file_name : str or None
        Velocity model file of a heterogeneous model.
    FEM_method_to_evaluate : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    dimension : int
        Spatial dimension, 2 or 3.
    receiver_setup : str
        Receiver setup, ``"near"``, ``"line"`` or ``"far"``.
    accepted_error_threshold : float
        Largest receiver error accepted. Given directly, or the error of
        ``threshold_case``.
    threshold_case : ThresholdCase or None
        Discretization whose error sets ``accepted_error_threshold``, or
        None if the threshold is given directly.
    threshold_case_evaluation : CpwEvaluation or None
        The forward solve of ``threshold_case``.
    desired_degree : int
        Polynomial degree the cpw is calculated for.
    reference_degree : int or None
        Polynomial degree of a heterogeneous numerical reference.
    cpw_reference : float or None
        Cells-per-wavelength of a heterogeneous numerical reference.
    cpw_initial : float
        Starting cells-per-wavelength. Must meet the error threshold for a
        backward search and exceed it for a forward search.
    cpw_accuracy : float
        Resolution of the cells-per-wavelength result.
    cpw_max : float
        Largest cells-per-wavelength evaluated.
    search_direction : CpwSearchDirection
        Direction of the line search run by :meth:`find_minimum`.
    reduced_obj_for_testing : bool
        Use a smaller receiver set, for testing.
    save_reference : bool
        Save the reference solution to ``reference_solution.npy``.
    load_reference : bool
        Load the reference solution from ``"reference_solution_file"``.
    timestep_calculation : TimeStepCalculationType
        How the time step of each forward solve is chosen.
    fixed_timestep : float or None
        Time step when ``timestep_calculation`` is ``FIXED``.
    search_history : list of CpwEvaluation
        Every forward solve of the last :meth:`find_minimum` call.
    initial_dictionary : dict
        Model dictionary of the initial guess object.
    initial_guess_object : Wave
        Wave object built from ``initial_dictionary``.
    comm : SpyroEnsemble
        The MPI communicator.
    reference_solution : np.ndarray
        Receiver data of the reference, sampled at the initial object's dt.
    """

    def __init__(self, parameters_dictionary: dict) -> None:
        """Initialize the calculator and compute its reference solution.

        Parameters
        ----------
        parameters_dictionary : dict
            All the parameters needed for the calculation:

            - ``"wave_type"``: WaveType, optional. Default acoustic.
            - ``"source_frequency"``: float.
            - ``"mesh_frequency_factor"``: float, optional. The wavelength
              of every cells-per-wavelength value is taken at this multiple
              of ``"source_frequency"``, e.g. 2.0 (about 90% of the Ricker
              energy) or 2.5 (close to its maximum frequency). Default is
              1.0, the peak frequency.
            - ``"minimum_velocity_in_the_domain"``: float, acoustic only.
            - ``"p_wave_velocity"``, ``"s_wave_velocity"``, ``"density"``:
              float, elastic only.
            - ``"velocity_profile_type"``: ``"homogeneous"`` or
              ``"heterogeneous"``.
            - ``"velocity_model_file_name"``: str or None.
            - ``"FEM_method_to_evaluate"``: str.
            - ``"dimension"``: int.
            - ``"receiver_setup"``: str.
            - ``"accepted_error_threshold"``: float. Exactly one of this
              and ``"threshold_case"`` must be given.
            - ``"threshold_case"``: True, dict or ThresholdCase, optional.
              Use the error of this discretization as the threshold. True
              is a degree 4 ``"spectral_quadrilateral"`` at cpw 2.0, and a
              dict overrides fields of :class:`ThresholdCase`.
            - ``"desired_degree"``: int.
            - ``"reference_degree"``, ``"C_reference"``: heterogeneous
              reference settings.
            - ``"C_initial"``, ``"C_accuracy"``: float, search settings.
            - ``"C_max"``: float, optional. Largest cells-per-wavelength
              evaluated. Default is :data:`MAX_CPW`.
            - ``"search_direction"``: ``"backward"`` (default) or
              ``"forward"``.
            - ``"time-step_calculation"``: ``"exact"`` (default),
              ``"estimate"`` or ``"float"``, with ``"time-step"`` for the
              latter.
            - ``"testing"``, ``"save_reference"``, ``"load_reference"``,
              ``"reference_solution_file"``: optional.

        Raises
        ------
        ValueError
            If ``"mesh_frequency_factor"`` is not positive, or both or
            neither of ``"accepted_error_threshold"`` and
            ``"threshold_case"`` are given.
        """
        self.parameters_dictionary = parameters_dictionary
        self.wave_type = validate_enum(
            "wave_type",
            parameters_dictionary.get("wave_type", WaveType.ISOTROPIC_ACOUSTIC),
            WaveType,
        )
        if self.wave_type not in CPW_PHYSICS:
            raise NotImplementedError(
                f"No cells-per-wavelength physics for {self.wave_type.name}."
            )
        self.physics = CPW_PHYSICS[self.wave_type](parameters_dictionary)

        self.source_frequency = parameters_dictionary["source_frequency"]
        self.mesh_frequency_factor = parameters_dictionary.get(
            "mesh_frequency_factor", 1.0
        )
        if self.mesh_frequency_factor <= 0.0:
            raise ValueError(
                "mesh_frequency_factor must be positive, got "
                f"{self.mesh_frequency_factor}."
            )
        self.mesh_frequency = self.mesh_frequency_factor * self.source_frequency
        self.minimum_velocity = self.physics.wavelength_velocity
        self.velocity_profile_type = validate_enum(
            "velocity_profile_type",
            parameters_dictionary["velocity_profile_type"],
            VelocityProfileType,
        )
        self.velocity_model_file_name = parameters_dictionary[
            "velocity_model_file_name"
        ]
        self._check_velocity_profile_type()
        self.FEM_method_to_evaluate = parameters_dictionary["FEM_method_to_evaluate"]
        self.dimension = parameters_dictionary["dimension"]
        self.receiver_setup = parameters_dictionary["receiver_setup"]
        self.accepted_error_threshold = parameters_dictionary.get(
            "accepted_error_threshold"
        )
        self.threshold_case = _parse_threshold_case(
            parameters_dictionary.get("threshold_case")
        )
        if (self.accepted_error_threshold is None) == (self.threshold_case is None):
            raise ValueError(
                "Give exactly one of accepted_error_threshold and threshold_case."
            )
        self.threshold_case_evaluation: CpwEvaluation | None = None
        self.desired_degree = parameters_dictionary["desired_degree"]

        # Only for use in heterogeneous models
        self.reference_degree = parameters_dictionary["reference_degree"]
        self.cpw_reference = parameters_dictionary["C_reference"]

        # Search parameters
        self.cpw_initial = parameters_dictionary["C_initial"]
        self.cpw_accuracy = parameters_dictionary["C_accuracy"]
        self.cpw_max = parameters_dictionary.get("C_max", MAX_CPW)
        self.search_direction = validate_enum(
            "search_direction",
            parameters_dictionary.get(
                "search_direction", CpwSearchDirection.BACKWARD
            ),
            CpwSearchDirection,
        )
        self.search_history: list[CpwEvaluation] = []

        # Debugging and testing parameters
        self.reduced_obj_for_testing = parameters_dictionary.get("testing", False)
        self.save_reference = parameters_dictionary.get("save_reference", False)
        self.load_reference = parameters_dictionary.get("load_reference", False)

        self.timestep_calculation = validate_enum(
            "time-step_calculation",
            parameters_dictionary.get(
                "time-step_calculation", TimeStepCalculationType.EXACT
            ),
            TimeStepCalculationType,
        )
        self.fixed_timestep = None
        if self.timestep_calculation is TimeStepCalculationType.FIXED:
            self.fixed_timestep = parameters_dictionary["time-step"]

        self.initial_dictionary = self.physics.build_dictionary(
            self, self.FEM_method_to_evaluate, self.desired_degree
        )
        self.initial_guess_object = self.physics.wave_class(
            copy.deepcopy(self.initial_dictionary)
        )
        self.comm = self.initial_guess_object.comm
        self.reference_solution = self.get_reference_solution()
        if self.threshold_case is not None:
            self.accepted_error_threshold = self._calculate_threshold()

    def _check_velocity_profile_type(self) -> None:
        """Check the velocity model inputs against the profile type.

        Raises
        ------
        ValueError
            If a homogeneous model names a velocity file, or a heterogeneous
            model lacks one or its domain lengths.
        """
        if self.velocity_profile_type is VelocityProfileType.HOMOGENEOUS:
            if self.velocity_model_file_name is not None:
                raise ValueError(
                    "Velocity model file name should be None for homogeneous models"
                )
        else:
            self._check_heterogeneous_mesh_lengths()
            if self.velocity_model_file_name is None:
                raise ValueError(
                    "Velocity model file name should be defined for heterogeneous models"
                )

    def _check_heterogeneous_mesh_lengths(self) -> None:
        """Check the domain lengths of a heterogeneous model.

        Raises
        ------
        ValueError
            If a length is missing, or ``length_x`` is negative.
        """
        parameters = self.parameters_dictionary
        if parameters.get("length_z") is None:
            raise ValueError("Length in z direction not defined")
        if parameters.get("length_x") is None:
            raise ValueError("Length in x direction not defined")
        if parameters["length_z"] < 0.0:
            parameters["length_z"] = abs(parameters["length_z"])
        if parameters["length_x"] < 0.0:
            raise ValueError("Length in x direction must be positive")

    def get_reference_solution(self) -> np.ndarray:
        """Load, compute numerically or compute analytically the reference.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """
        if self.load_reference:
            filename = self.parameters_dictionary.get(
                "reference_solution_file", "reference_solution.npy"
            )
            return np.load(filename)

        if self.velocity_profile_type is VelocityProfileType.HETEROGENEOUS:
            reference = self.calculate_reference_solution()
        else:
            reference = self.calculate_analytical_solution()

        if self.save_reference:
            np.save("reference_solution.npy", reference)
        return reference

    def calculate_reference_solution(self) -> np.ndarray:
        """Compute the numerical reference of a heterogeneous model.

        Uses the ``C_reference`` and ``reference_degree`` parameters.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """
        wave = self.build_current_object(self.cpw_reference, degree=self.reference_degree)
        wave.forward_solve()
        return wave.forward_solution_receivers

    def calculate_analytical_solution(self) -> np.ndarray:
        """Compute the analytical reference of a homogeneous model.

        Returns
        -------
        np.ndarray
            The reference receiver data, at the initial object's dt.
        """
        return self.physics.analytical_solution(self.initial_guess_object)

    def build_current_object(
        self, cpw: float, degree: int | None = None, method: str | None = None
    ) -> Wave:
        """Build the wave object for a cells-per-wavelength value.

        Parameters
        ----------
        cpw : float
            Cells-per-wavelength of the mesh.
        degree : int, optional
            Polynomial degree. Default is ``desired_degree``.
        method : str, optional
            Finite element method. Default is ``FEM_method_to_evaluate``.

        Returns
        -------
        Wave
            Wave object with its mesh built.
        """
        if degree is None and method is None:
            dictionary = copy.deepcopy(self.initial_dictionary)
        else:
            dictionary = self.physics.build_dictionary(
                self,
                method if method is not None else self.FEM_method_to_evaluate,
                degree if degree is not None else self.desired_degree,
            )
        wave = self.physics.wave_class(dictionary)
        if self.velocity_profile_type is VelocityProfileType.HOMOGENEOUS:
            wavelength = self.minimum_velocity / self.mesh_frequency
            self.physics.setup_homogeneous_wave(wave, wavelength / cpw)
        else:
            # The mesher sizes cells from the source frequency, and the edge
            # length depends only on frequency * cpw
            wave.set_mesh(
                input_mesh_parameters={
                    "cells_per_wavelength": cpw * self.mesh_frequency_factor
                }
            )
        return wave

    def calculate_error(self, cpw: float) -> float:
        """Solve at a cells-per-wavelength and measure the receiver error.

        The shot record is saved to ``test_shot_record<cpw>`` and the
        result is appended to :attr:`search_history`.

        Parameters
        ----------
        cpw : float
            Cells-per-wavelength of the mesh.

        Returns
        -------
        float
            Receiver error from :meth:`MeasureError.calculate_receiver_error`.
        """
        SpyroEnsemble.print(f"Trying cells-per-wavelength = {cpw}")
        wave = self.build_current_object(cpw)
        evaluation = self._solve_and_measure(wave, cpw)
        spyro.io.save_shots(wave, file_name=f"test_shot_record{cpw}")
        self.search_history.append(evaluation)
        return evaluation.error

    def _calculate_threshold(self) -> float:
        """Measure the receiver error of :attr:`threshold_case`.

        The solve is stored in :attr:`threshold_case_evaluation`.

        Returns
        -------
        float
            The error of the threshold case, used as the threshold.
        """
        case = self.threshold_case
        SpyroEnsemble.print(
            f"Computing the error threshold from {case.method} degree "
            f"{case.degree} at cells-per-wavelength = {case.cpw}"
        )
        wave = self.build_current_object(case.cpw, degree=case.degree, method=case.method)
        self.threshold_case_evaluation = self._solve_and_measure(wave, case.cpw)
        SpyroEnsemble.print(
            f"Accepted error threshold is {self.threshold_case_evaluation.error}"
        )
        return self.threshold_case_evaluation.error

    def _solve_and_measure(self, wave: Wave, cpw: float) -> CpwEvaluation:
        """Run a forward solve and measure its error against the reference.

        The time step is set per :attr:`timestep_calculation` and the
        reference is interpolated onto it.

        Parameters
        ----------
        wave : Wave
            Wave object with its mesh built.
        cpw : float
            Cells-per-wavelength the mesh was built with.

        Returns
        -------
        CpwEvaluation
            The cpw, time step, receiver error and runtime of the solve.
        """
        wave.initialize_physical_parameters()
        self._set_time_step(wave)
        SpyroEnsemble.print(f"Maximum dt (seconds) is {wave.dt}")

        start = time.perf_counter()
        wave.forward_solve()
        runtime = time.perf_counter() - start

        reference = interpolate_time_series(
            self.reference_solution,
            wave.dt,
            initial_time=wave.initial_time,
            final_time=wave.final_time,
        )
        error = MeasureError.calculate_receiver_error(
            wave.forward_solution_receivers,
            reference,
            wave.dt,
            has_displacement=self.physics.has_displacement,
        )
        SpyroEnsemble.print(f"Error is {error}")
        return CpwEvaluation(cpw, wave.dt, error, runtime)

    def _set_time_step(self, wave: Wave) -> None:
        """Set the forward solve time step.

        Parameters
        ----------
        wave : Wave
            Wave object with its mesh and material set.
        """
        if self.timestep_calculation is TimeStepCalculationType.FIXED:
            wave.dt = self.fixed_timestep
        else:
            wave.get_and_set_maximum_dt(
                fraction=0.5,
                estimate_max_eigenvalue=(
                    self.timestep_calculation is TimeStepCalculationType.ESTIMATE
                ),
            )

    def find_minimum(
        self,
        starting_cpw: float | None = None,
        TOL: float | None = None,
        accuracy: float | None = None,
        savetxt: bool = False,
        search_direction: CpwSearchDirection | str | None = None,
        max_cpw: float | None = None,
    ) -> float:
        """Find the minimum cells-per-wavelength below the error threshold.

        Runs :func:`backward_line_search` or :func:`forward_line_search`
        from ``starting_cpw``.

        Parameters
        ----------
        starting_cpw : float, optional
            Starting cells-per-wavelength, which must meet the threshold
            for a backward search and exceed it for a forward search.
            Default is ``cpw_initial``.
        TOL : float, optional
            Largest accepted receiver error. Default is
            ``accepted_error_threshold``.
        accuracy : float, optional
            Resolution of the result. Default is ``cpw_accuracy``.
        savetxt : bool, optional
            Save the search history to ``p<degree>_cpw_results.txt``.
            Default is False.
        search_direction : CpwSearchDirection or str, optional
            Direction of the line search. Default is ``search_direction``.
        max_cpw : float, optional
            Largest cells-per-wavelength evaluated. Default is ``cpw_max``.

        Returns
        -------
        float
            The smallest evaluated cells-per-wavelength meeting ``TOL``.
        """
        if starting_cpw is None:
            starting_cpw = self.cpw_initial
        if TOL is None:
            TOL = self.accepted_error_threshold
        if accuracy is None:
            accuracy = self.cpw_accuracy
        if search_direction is None:
            search_direction = self.search_direction
        search_direction = validate_enum(
            "search_direction", search_direction, CpwSearchDirection
        )
        if max_cpw is None:
            max_cpw = self.cpw_max
        line_search = LINE_SEARCHES[search_direction]

        SpyroEnsemble.print(f"Starting {search_direction} line search")
        self.search_history = []
        cpw = line_search(
            self.calculate_error, starting_cpw, TOL, accuracy, max_cpw=max_cpw
        )

        if savetxt:
            np.savetxt(
                f"p{self.initial_guess_object.degree}_cpw_results.txt",
                [
                    (e.cpw, e.dt, e.error, e.runtime)
                    for e in self.search_history
                ],
                header="cpw dt error runtime",
            )
        return cpw


def _floor_to_accuracy(value: float, accuracy: float) -> float:
    """Floor a value onto the grid of multiples of ``accuracy``.

    Parameters
    ----------
    value : float
        Value to floor.
    accuracy : float
        Grid spacing.

    Returns
    -------
    float
        Largest multiple of ``accuracy`` not above ``value``.
    """
    # The small tolerance keeps values already on the grid in place
    return round(math.floor(value / accuracy + 1e-9) * accuracy, 10)


def _ceil_to_accuracy(value: float, accuracy: float) -> float:
    """Ceil a value onto the grid of multiples of ``accuracy``.

    Parameters
    ----------
    value : float
        Value to ceil.
    accuracy : float
        Grid spacing.

    Returns
    -------
    float
        Smallest multiple of ``accuracy`` not below ``value``.
    """
    # The small tolerance keeps values already on the grid in place
    return round(math.ceil(value / accuracy - 1e-9) * accuracy, 10)


class _CachedThresholdCheck:
    """Memoized check of whether a cells-per-wavelength meets a threshold.

    Each forward solve is expensive, so every value is evaluated at most
    once.

    Attributes
    ----------
    calculate_error : Callable[[float], float]
        Returns the error of a cells-per-wavelength value.
    threshold : float
        Largest accepted error.
    errors : dict[float, float]
        Error of every value evaluated so far.
    """

    def __init__(
        self, calculate_error: Callable[[float], float], threshold: float
    ) -> None:
        """Store the error function and the threshold.

        Parameters
        ----------
        calculate_error : Callable[[float], float]
            Returns the error of a cells-per-wavelength value.
        threshold : float
            Largest accepted error.
        """
        self.calculate_error = calculate_error
        self.threshold = threshold
        self.errors: dict[float, float] = {}

    def __call__(self, cpw: float) -> bool:
        """Check a value, evaluating its error only the first time.

        Parameters
        ----------
        cpw : float
            Cells-per-wavelength to check.

        Returns
        -------
        bool
            Whether the error of ``cpw`` meets the threshold.
        """
        if cpw not in self.errors:
            self.errors[cpw] = self.calculate_error(cpw)
        return self.errors[cpw] <= self.threshold


def _check_search_settings(
    starting_cpw: float, accuracy: float, coarse_step_fraction: float
) -> None:
    """Check the settings shared by the line searches.

    Parameters
    ----------
    starting_cpw : float
        First value evaluated.
    accuracy : float
        Resolution of the result.
    coarse_step_fraction : float
        Fraction of the current value moved per coarse step.

    Raises
    ------
    ValueError
        If any setting is not positive.
    """
    for name, value in (
        ("starting_cpw", starting_cpw),
        ("accuracy", accuracy),
        ("coarse_step_fraction", coarse_step_fraction),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")


def _bisect_grid(
    passes: _CachedThresholdCheck,
    failing: float,
    passing: float,
    accuracy: float,
) -> float:
    """Bisect the grid for the smallest value meeting the threshold.

    The error is assumed to decrease as the cells-per-wavelength grows, so
    every grid value up to ``failing`` fails and every value from
    ``passing`` on passes. Only grid values strictly between them are
    evaluated.

    Parameters
    ----------
    passes : _CachedThresholdCheck
        Threshold check of a cells-per-wavelength value.
    failing : float
        A value known to fail, or 0.0 when none is known.
    passing : float
        A value known to pass. It may be off the grid.
    accuracy : float
        Grid spacing.

    Returns
    -------
    float
        The smallest grid value in ``(failing, passing)`` that passes, or
        ``passing`` if none does.
    """
    # Grid indices: values at or below low fail, values at or above high pass
    low = math.floor(failing / accuracy + 1e-9)
    high = math.ceil(passing / accuracy - 1e-9)
    smallest_passing = passing
    while high - low > 1:
        middle = (low + high) // 2
        candidate = round(middle * accuracy, 10)
        if passes(candidate):
            high = middle
            smallest_passing = candidate
        else:
            low = middle
    return smallest_passing


def backward_line_search(
    calculate_error: Callable[[float], float],
    starting_cpw: float,
    threshold: float,
    accuracy: float,
    coarse_step_fraction: float = COARSE_STEP_FRACTION,
    max_cpw: float = MAX_CPW,
) -> float:
    """Search downward for the smallest cells-per-wavelength meeting a threshold.

    The error is assumed to decrease as the cells-per-wavelength grows.
    Starting from a passing value, a coarse phase removes
    ``coarse_step_fraction`` of the current value per step, but at least
    ``accuracy``, until the error first exceeds the threshold. A fine phase
    then bisects the grid between the last passing and the first failing
    value. Each value is evaluated at most once.

    Parameters
    ----------
    calculate_error : Callable[[float], float]
        Returns the error of a cells-per-wavelength value.
    starting_cpw : float
        First value evaluated. Its error must meet ``threshold``.
    threshold : float
        Largest accepted error.
    accuracy : float
        Resolution of the result. Candidates after the start are multiples
        of it.
    coarse_step_fraction : float, optional
        Fraction of the current value removed per coarse step. Default is
        :data:`COARSE_STEP_FRACTION`.
    max_cpw : float, optional
        Largest value evaluated. Default is :data:`MAX_CPW`.

    Returns
    -------
    float
        The smallest evaluated value whose error meets ``threshold``.

    Raises
    ------
    ValueError
        If a setting is not positive, ``starting_cpw`` is above
        ``max_cpw``, or ``starting_cpw`` does not meet ``threshold``.
    """
    _check_search_settings(starting_cpw, accuracy, coarse_step_fraction)
    if starting_cpw > max_cpw:
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} is above "
            f"the maximum {max_cpw}."
        )
    passes = _CachedThresholdCheck(calculate_error, threshold)

    if not passes(starting_cpw):
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} has error "
            f"{passes.errors[starting_cpw]} above the threshold {threshold}. "
            "Start from a larger value."
        )
    last_passing = starting_cpw
    first_failing = 0.0

    # Coarse phase: large relative steps until the first failure
    while True:
        step = max(coarse_step_fraction * last_passing, accuracy)
        candidate = _floor_to_accuracy(last_passing - step, accuracy)
        if candidate <= 0.0:
            break
        if not passes(candidate):
            first_failing = candidate
            break
        last_passing = candidate

    return _bisect_grid(passes, first_failing, last_passing, accuracy)


def forward_line_search(
    calculate_error: Callable[[float], float],
    starting_cpw: float,
    threshold: float,
    accuracy: float,
    coarse_step_fraction: float = COARSE_STEP_FRACTION,
    max_cpw: float = MAX_CPW,
) -> float:
    """Search upward for the smallest cells-per-wavelength meeting a threshold.

    The error is assumed to decrease as the cells-per-wavelength grows.
    Starting from a failing value, a coarse phase adds
    ``coarse_step_fraction`` of the current value per step, but at least
    ``accuracy``, until the error first meets the threshold. A fine phase
    then bisects the grid between the last failing and the first passing
    value. No value above ``max_cpw`` is evaluated, and each value is
    evaluated at most once.

    Parameters
    ----------
    calculate_error : Callable[[float], float]
        Returns the error of a cells-per-wavelength value.
    starting_cpw : float
        First value evaluated. Its error must exceed ``threshold``.
    threshold : float
        Largest accepted error.
    accuracy : float
        Resolution of the result. Candidates after the start are multiples
        of it.
    coarse_step_fraction : float, optional
        Fraction of the current value added per coarse step. Default is
        :data:`COARSE_STEP_FRACTION`.
    max_cpw : float, optional
        Largest value evaluated. Default is :data:`MAX_CPW`.

    Returns
    -------
    float
        The smallest evaluated value whose error meets ``threshold``.

    Raises
    ------
    ValueError
        If a setting is not positive, ``starting_cpw`` already meets
        ``threshold`` or is not below ``max_cpw``, or ``max_cpw`` does not
        meet ``threshold``.
    """
    _check_search_settings(starting_cpw, accuracy, coarse_step_fraction)
    if starting_cpw >= max_cpw:
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} is not below "
            f"the maximum {max_cpw}."
        )
    passes = _CachedThresholdCheck(calculate_error, threshold)

    if passes(starting_cpw):
        raise ValueError(
            f"Starting cells-per-wavelength {starting_cpw} has error "
            f"{passes.errors[starting_cpw]} within the threshold {threshold}. "
            "Start from a smaller value."
        )
    last_failing = starting_cpw

    # Coarse phase: large relative steps until the first success
    while True:
        step = max(coarse_step_fraction * last_failing, accuracy)
        candidate = min(_ceil_to_accuracy(last_failing + step, accuracy), max_cpw)
        if passes(candidate):
            first_passing = candidate
            break
        if candidate >= max_cpw:
            raise ValueError(
                f"Maximum cells-per-wavelength {max_cpw} has error "
                f"{passes.errors[candidate]} above the threshold {threshold}."
            )
        last_failing = candidate

    return _bisect_grid(passes, last_failing, first_passing, accuracy)


#: Line search run by :meth:`MeshingParameterCalculator.find_minimum`,
#: selected by the ``"search_direction"`` parameter.
LINE_SEARCHES: dict[CpwSearchDirection, Callable[..., float]] = {
    CpwSearchDirection.BACKWARD: backward_line_search,
    CpwSearchDirection.FORWARD: forward_line_search,
}
