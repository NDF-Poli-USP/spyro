"""Find the minimum cells-per-wavelength meshing parameter of a wave problem."""

import copy
import time
from dataclasses import dataclass

import numpy as np

import spyro
from .line_search import LINE_SEARCHES
from .parameters import CpwParameters
from .problems import CPW_PROBLEMS
from ..error_measure import MeasureError
from ...io.time_io import interpolate_time_series
from ...mpi.spyro_mpi import SpyroEnsemble
from ...solvers.wave import Wave
from ...utils.error_management import validate_enum
from ...utils.typing import CpwSearchDirection, TimeStepCalculationType

#: Fraction of the maximum stable time step used by every forward solve
#: whose time step is not fixed.
MAXIMUM_DT_FRACTION = 0.5


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


class MeshingParameterCalculator:
    """Calculate the minimum meshing parameter C (cells-per-wavelength).

    The wave problem is chosen by the ``wave_type`` and
    ``velocity_profile_type`` parameters. See :data:`~.problems.CPW_PROBLEMS`.

    Attributes
    ----------
    parameters : CpwParameters
        All the parameters of the calculation.
    problem : CpwProblem
        The wave problem solved at every cells-per-wavelength.
    accepted_error_threshold : float
        Largest receiver error accepted. Given directly, or the error of
        ``parameters.threshold_discretization``.
    threshold_evaluation : CpwEvaluation or None
        The forward solve of ``parameters.threshold_discretization``, or
        None if the threshold is given directly.
    search_history : list of CpwEvaluation
        Every forward solve of the last :meth:`find_minimum` call.
    reference_solution_file : str
        File the reference is saved to or loaded from. Defaults to
        :meth:`CpwProblem.default_reference_file`, e.g.
        ``"acoustic_homogeneous_cpw_reference.npy"``.
    initial_dictionary : dict
        Model dictionary of the discretization under evaluation.
    initial_wave : Wave
        Wave object built from ``initial_dictionary``.
    reference_solution : np.ndarray
        Receiver data of the reference, sampled at the initial wave's dt.
    """

    def __init__(self, parameters: CpwParameters | dict) -> None:
        """Initialize the calculator and compute its reference solution.

        Parameters
        ----------
        parameters : CpwParameters or dict
            All the parameters of the calculation. A dict is converted with
            :meth:`CpwParameters.from_dict`, whose keys are the
            :class:`~.parameters.CpwParameters` field names except those in
            :data:`~.parameters.DICTIONARY_KEYS`, e.g. ``"C_initial"``.

        Raises
        ------
        NotImplementedError
            If no problem is registered for the wave and profile types.
        ValueError
            If the parameters are inconsistent, see
            :class:`~.parameters.CpwParameters` and the problem classes.
        """
        if isinstance(parameters, dict):
            parameters = CpwParameters.from_dict(parameters)
        self.parameters = parameters

        problem_class = CPW_PROBLEMS.get(
            (parameters.wave_type, parameters.velocity_profile_type)
        )
        if problem_class is None:
            raise NotImplementedError(
                f"No cells-per-wavelength problem for {parameters.wave_type.name} "
                f"with a {parameters.velocity_profile_type.value} velocity profile."
            )
        self.problem = problem_class(parameters)

        self.accepted_error_threshold = parameters.accepted_error_threshold
        self.threshold_evaluation: CpwEvaluation | None = None
        self.search_history: list[CpwEvaluation] = []
        self.reference_solution_file = (
            parameters.reference_solution_file
            or self.problem.default_reference_file()
        )

        self.initial_dictionary = self.problem.model_dictionary(
            parameters.fem_method, parameters.desired_degree
        )
        self.initial_wave = self.problem.physics.wave_class(
            copy.deepcopy(self.initial_dictionary)
        )
        self.reference_solution = self.get_reference_solution()
        if parameters.threshold_discretization is not None:
            self.accepted_error_threshold = self._calculate_threshold()

    def get_reference_solution(self) -> np.ndarray:
        """Load or compute the reference receiver data.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """
        if self.parameters.load_reference:
            return np.load(self.reference_solution_file)

        reference = self.problem.reference_solution(
            self.initial_wave, self.parameters.fem_method
        )

        if self.parameters.save_reference:
            SpyroEnsemble.run_in_one_core_and_broadcast(np.save)(
                self.reference_solution_file, reference
            )
        return reference

    def build_wave(
        self, cpw: float, degree: int | None = None, method: str | None = None
    ) -> Wave:
        """Build the wave object for a cells-per-wavelength value.

        Parameters
        ----------
        cpw : float
            Cells-per-wavelength of the mesh.
        degree : int, optional
            Polynomial degree. Default is ``parameters.desired_degree``.
        method : str, optional
            Finite element method. Default is ``parameters.fem_method``.

        Returns
        -------
        Wave
            Wave object with its mesh built.
        """
        if degree is None and method is None:
            dictionary = copy.deepcopy(self.initial_dictionary)
        else:
            dictionary = self.problem.model_dictionary(
                method if method is not None else self.parameters.fem_method,
                degree if degree is not None else self.parameters.desired_degree,
            )
        return self.problem.build_wave(dictionary, cpw)

    def calculate_error(self, cpw: float) -> float:
        """Solve at a cells-per-wavelength and measure the receiver error.

        The result is appended to :attr:`search_history`, and the shot
        record is saved to ``test_shot_record<cpw>`` if
        ``parameters.save_shot_records`` is set.

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
        wave = self.build_wave(cpw)
        evaluation = self._solve_and_measure(wave, cpw)
        if self.parameters.save_shot_records:
            spyro.io.save_shots(wave, file_name=f"test_shot_record{cpw}")
        self.search_history.append(evaluation)
        return evaluation.error

    def _calculate_threshold(self) -> float:
        """Measure the receiver error of the threshold discretization.

        The solve is stored in :attr:`threshold_evaluation`.

        Returns
        -------
        float
            The error of the threshold discretization, used as the threshold.
        """
        discretization = self.parameters.threshold_discretization
        SpyroEnsemble.print(
            f"Computing the error threshold from {discretization.method} degree "
            f"{discretization.degree} at cells-per-wavelength = "
            f"{discretization.cpw}"
        )
        wave = self.build_wave(
            discretization.cpw,
            degree=discretization.degree,
            method=discretization.method,
        )
        self.threshold_evaluation = self._solve_and_measure(wave, discretization.cpw)
        SpyroEnsemble.print(
            f"Accepted error threshold is {self.threshold_evaluation.error}"
        )
        return self.threshold_evaluation.error

    def _solve_and_measure(self, wave: Wave, cpw: float) -> CpwEvaluation:
        """Run a forward solve and measure its error against the reference.

        The time step is set per ``parameters.timestep_calculation`` and the
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
            time_shift=self.problem.reference_time_shift(wave),
        )
        error = MeasureError.calculate_receiver_error(
            wave.forward_solution_receivers,
            reference,
            wave.dt,
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
        timestep_calculation = self.parameters.timestep_calculation
        if timestep_calculation is TimeStepCalculationType.FIXED:
            wave.dt = self.parameters.fixed_timestep
        else:
            wave.get_and_set_maximum_dt(
                fraction=MAXIMUM_DT_FRACTION,
                estimate_max_eigenvalue=(
                    timestep_calculation is TimeStepCalculationType.ESTIMATE
                ),
            )

    def find_minimum(
        self,
        starting_cpw: float | None = None,
        tol: float | None = None,
        accuracy: float | None = None,
        savetxt: bool = False,
        search_direction: CpwSearchDirection | str | None = None,
        max_cpw: float | None = None,
    ) -> float:
        """Find the minimum cells-per-wavelength below the error threshold.

        Runs :func:`~.line_search.backward_line_search` or
        :func:`~.line_search.forward_line_search`
        from ``starting_cpw``.

        Parameters
        ----------
        starting_cpw : float, optional
            Starting cells-per-wavelength, which must meet the threshold
            for a backward search and exceed it for a forward search.
            Default is ``parameters.cpw_initial``.
        tol : float, optional
            Largest accepted receiver error. Default is
            ``accepted_error_threshold``.
        accuracy : float, optional
            Resolution of the result. Default is ``parameters.cpw_accuracy``.
        savetxt : bool, optional
            Save the search history to ``p<degree>_cpw_results.txt``.
            Default is False.
        search_direction : CpwSearchDirection or str, optional
            Direction of the line search. Default is
            ``parameters.search_direction``.
        max_cpw : float, optional
            Largest cells-per-wavelength evaluated. Default is
            ``parameters.cpw_max``.

        Returns
        -------
        float
            The smallest evaluated cells-per-wavelength meeting ``tol``.
        """
        if starting_cpw is None:
            starting_cpw = self.parameters.cpw_initial
        if tol is None:
            tol = self.accepted_error_threshold
        if accuracy is None:
            accuracy = self.parameters.cpw_accuracy
        if search_direction is None:
            search_direction = self.parameters.search_direction
        search_direction = validate_enum(
            "search_direction", search_direction, CpwSearchDirection
        )
        if max_cpw is None:
            max_cpw = self.parameters.cpw_max
        line_search = LINE_SEARCHES[search_direction]

        SpyroEnsemble.print(f"Starting {search_direction} line search")
        self.search_history = []
        cpw = line_search(
            self.calculate_error, starting_cpw, tol, accuracy, max_cpw=max_cpw
        )

        if savetxt:
            SpyroEnsemble.run_in_one_core_and_broadcast(np.savetxt)(
                f"p{self.initial_wave.degree}_cpw_results.txt",
                [(e.cpw, e.dt, e.error, e.runtime) for e in self.search_history],
                header="cpw dt error runtime",
            )
        return cpw
