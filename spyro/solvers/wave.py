from abc import abstractmethod, ABCMeta
import sys
import warnings
import firedrake as fire
from .automatic_differentiation_solver import AutomatedAdjoint

from .time_integration_central_difference import (
    central_difference as time_integrator)
from ..domains.quadrature import quadrature_rules
from ..io import Model_parameters
from ..io.basicio import ensemble_propagator
from ..io.field_logger import FieldLogger
from .. import utils
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from ..utils.typing import WaveType, AdjointType
from .solver_parameters import get_default_parameters_for_method

fire.set_log_level(fire.ERROR)


class Wave(Model_parameters, metaclass=ABCMeta):
    """
    Base class for wave equation solvers.

    Attributes
    ----------
    comm : MPI communicator
        Communicator used by the solver.
    initial_velocity_model : firedrake.Function or None
        Initial velocity model used to initialize ``c``.
    function_space : firedrake.FunctionSpace or None
        Function space used by the wave equation.
    current_time : float
        Current simulation time.
    solver_parameters : dict
        Solver options for linear/nonlinear solves.
    real_shot_record : list or numpy.ndarray or None
        Observed receiver data used in inversion workflows.
    mesh : firedrake.Mesh or None
        Mesh used in the simulation (2D or 3D).
    sources : Sources or None
        Source manager.
    receivers : Receivers or None
        Receiver manager.
    adjoint_type : AdjointType
        Active adjoint mode (none, Spyro adjoint, or automated adjoint).

    Methods
    -------
    set_mesh(...)
        Set or rebuild the mesh and dependent objects.
    set_solver_parameters(...)
        Set custom solver parameters or defaults.
    set_initial_velocity_model(...)
        Set the initial velocity model.
    forward_solve()
        Build operators and propagate the forward wave.
    enable_automated_adjoint()
        Enable Firedrake automated adjoint mode.
    enable_spyro_adjoint()
        Enable Spyro native adjoint mode.
    get_and_set_maximum_dt(...)
        Estimate and set a stable time step.
    set_last_solve_as_real_shot_record()
        Store the last simulated data as observed data.
    """

    def __init__(self, dictionary=None, comm=None, real_shot_record=None):
        """Initialize a wave solver with forward and adjoint capabilities.

        Parameters
        ----------
        dictionary : dict, optional
            Input model dictionary with options, mesh, acquisition,
            parallelism, time-axis, and visualization entries.
        comm : MPI communicator, optional
            Communicator used to build distributed solver objects.
        real_shot_record : list or numpy.ndarray, optional
            Receiver data to be used in inversion workflows. This data can
            come from a previous forward simulation or from field measurements.
            If not provided, it can be set later using the ``real_shot_record``
            property setter.

        Notes
        -----
        Initializes core state (mesh, function space, sources/receivers,
        logging hooks, and adjoint mode flags).
        """
        super().__init__(dictionary=dictionary, comm=comm)
        self.initial_velocity_model = None
        self.wave_type = WaveType.NONE

        self.function_space = None
        self.lhs = None
        self.rhs = None
        self.forward_solution = None
        self.receivers_data = None
        self.current_time = 0.0
        self.set_solver_parameters()

        self.mesh = self.get_mesh()
        self.c = None
        self.sources = None
        self._compute_functional = False
        self._functional_value = None

        self.current_sources = None
        self.automated_adjoint = False
        self._store_misfit = False
        self._real_shot_record = real_shot_record
        self.store_forward_time_steps = False
        self.adjoint_type = AdjointType.NONE
        self.misfit = None
        if self.mesh is not None:
            self._build_function_space()
            self._map_sources_and_receivers()
        elif self.mesh_parameters.mesh_type == "firedrake_mesh":
            warnings.warn(
                "No mesh file, Firedrake mesh will be automatically generated."
            )
        else:
            warnings.warn("No mesh found. Please define a mesh.")
        # Expression to define sources through UFL (less efficient)
        self.source_expression = None
        # Object for efficient application of sources

        self.field_logger = FieldLogger(self.comm,
                                        self.input_dictionary["visualization"])
        self.field_logger.add_field("forward", self.get_function_name(),
                                    lambda: self.get_function())

    def forward_solve(self):
        """Solve one forward wave simulation.

        Ensures a valid function space and model parameters are available,
        builds the solver operators, and advances the wavefield in time.
        """

        if "-s" in sys.argv or bool(sys.flags.no_user_site):
            print("\nSolving Forward Problem")

        if self.function_space is None:
            self.force_rebuild_function_space()

        if self.abc_boundary_layer_type != "hybrid":
            self._initialize_model_parameters()
        self.matrix_building()
        self.wave_propagator()

    def force_rebuild_function_space(self):
        if self.mesh is None:
            self.mesh = self.get_mesh()
        self._build_function_space()
        self._map_sources_and_receivers()

    @abstractmethod
    def matrix_building(self):
        """Builds the matrix for the forward problem."""
        pass

    def set_mesh(
            self,
            user_mesh=None,
            input_mesh_parameters={},
    ):
        """
        Set the mesh for the solver.

        Args:
            user_mesh (optional): User-defined mesh. Defaults to None.
            mesh_parameters (optional): Parameters for generating a mesh. Defaults to None.
        """
        super().set_mesh(
            user_mesh=user_mesh,
            input_mesh_parameters=input_mesh_parameters,
        )

        self.mesh = self.get_mesh()
        self._build_function_space()
        self._map_sources_and_receivers()

    def set_solver_parameters(self, parameters=None):
        """
        Set the solver parameters.

        Args:
            parameters (dict): A dictionary containing the solver parameters.

        Returns:
            None
        """
        if parameters is not None:
            self.solver_parameters = parameters
        elif parameters is None:
            self.solver_parameters = get_default_parameters_for_method(
                self.method
            )

    def get_spatial_coordinates(self):
        if self.dimension == 2:
            return self.mesh_z, self.mesh_x
        elif self.dimension == 3:
            return self.mesh_z, self.mesh_x, self.mesh_y

    def set_initial_velocity_model(
        self,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
        dg_velocity_model=True,
    ):
        """Method to define new user velocity model or file. It is optional.

        Parameters:
        -----------
        conditional:  (optional)
            Firedrake conditional object.
        velocity_model_function: Firedrake function (optional)
            Firedrake function to be used as the velocity model. Has to be in the same function space as the object.
        expression:  str (optional)
            If you use an expression, you can use the following variables:
            x, y, z, pi, tanh, sqrt. Example: "2.0 + 0.5*tanh((x-2.0)/0.1)".
            It will be interpoalte into either the same function space as the object or a DG0 function space
            in the same mesh.
        new_file:  str (optional)
            Name of the file containing the velocity model.
        output:  bool (optional)
            If True, outputs the velocity model to a pvd file for visualization.
        """
        if new_file is not None:
            self.initial_velocity_model_file = new_file
        # If no mesh is set, we have to do it beforehand
        if self.mesh is None:
            self.set_mesh()
        # Resseting old velocity model
        self.initial_velocity_model = None
        self.initial_velocity_model_file = None

        if self.debug_output:
            output = True

        if conditional is not None:
            if dg_velocity_model:
                V = fire.FunctionSpace(self.mesh, "DG", 0)
            else:
                V = self.function_space
            vp = fire.Function(V, name="velocity")
            vp.interpolate(conditional)
            self.initial_velocity_model = vp
        elif expression is not None:
            z = self.mesh_z  # noqa: F841
            x = self.mesh_x  # noqa: F841
            if self.dimension == 3:
                y = self.mesh_y  # noqa: F841
            expression = eval(expression)
            V = self.function_space
            vp = fire.Function(V, name="velocity")
            vp.interpolate(expression)
            self.initial_velocity_model = vp
        elif velocity_model_function is not None:
            self.initial_velocity_model = velocity_model_function
        elif new_file is not None:
            self.initial_velocity_model_file = new_file
            self._get_initial_velocity_model()
        elif constant is not None:
            V = self.function_space
            vp = fire.Function(V, name="velocity")
            vp.interpolate(fire.Constant(constant))
            self.initial_velocity_model = vp
        else:
            raise ValueError(
                "Please specify either a conditional, expression, firedrake "
                "function or new file name (segy or hdf5)."
            )
        if output:
            fire.VTKFile("initial_velocity_model.pvd").write(
                self.initial_velocity_model, name="velocity"
            )

    def _map_sources_and_receivers(self):
        if self.source_type == "ricker":
            self.sources = Sources(self)
        self.receivers = Receivers(self)

    @abstractmethod
    def _initialize_model_parameters(self):
        pass

    @abstractmethod
    def _create_function_space(self):
        pass

    def _build_function_space(self):
        self.function_space = self._create_function_space()

        quad_rule, k_rule, s_rule = quadrature_rules(self.function_space)
        self.quadrature_rule = quad_rule
        self.stiffness_quadrature_rule = k_rule
        self.surface_quadrature_rule = s_rule

        # TO REVIEW: why are the mesh coordinates assigned here? I believe they
        # should be copied when the mesh is assigned
        if self.dimension == 2:
            z, x = fire.SpatialCoordinate(self.mesh)
            self.mesh_z = z
            self.mesh_x = x
        elif self.dimension == 3:
            z, x, y = fire.SpatialCoordinate(self.mesh)
            self.mesh_z = z
            self.mesh_x = x
            self.mesh_y = y

    def get_and_set_maximum_dt(self, fraction=0.7, estimate_max_eigenvalue=False):
        """
        Calculates and sets the maximum stable time step (dt) for the wave solver.

        Args:
            fraction (float, optional):
                Fraction of the estimated time step to use. Defaults to 0.7.
            estimate_max_eigenvalue (bool, optional):
                Whether to estimate the maximum eigenvalue. Defaults to False.

        Returns:
            float: The calculated maximum time step (dt).
        """
        # if self.method == "mass_lumped_triangle":
        #     estimate_max_eigenvalue = True
        # elif self.method == "spectral_quadrilateral":
        #     estimate_max_eigenvalue = True
        # else:

        if self.c is None:
            c = self.initial_velocity_model
        else:
            c = self.c

        dt = utils.estimate_timestep.estimate_timestep(
            self.mesh,
            self.function_space,
            c,
            estimate_max_eigenvalue=estimate_max_eigenvalue,
        )
        dt *= fraction
        nt = int(self.final_time / dt) + 1
        dt = self.final_time / (nt - 1)

        self.dt = dt

        return dt

    def get_mass_matrix_diagonal(self):
        """Builds a section of the mass matrix for debugging purposes."""
        A = fire.assemble(self.lhs, mat_type="aij")
        petsc_matrix = A.petscmat
        diagonal = petsc_matrix.getDiagonal()
        return diagonal.array

    def set_last_solve_as_real_shot_record(self):
        if self.current_time == 0.0:
            raise ValueError("No previous solve to set as real shot record.")
        self._real_shot_record = self.receivers_data

    @abstractmethod
    def _set_vstate(self, vstate):
        pass

    @abstractmethod
    def _get_vstate(self):
        pass

    @abstractmethod
    def _set_prev_vstate(self, vstate):
        pass

    @abstractmethod
    def _get_prev_vstate(self):
        pass

    @abstractmethod
    def _set_next_vstate(self, vstate):
        pass

    @abstractmethod
    def _get_next_vstate(self):
        pass

    # Managed attributes to access state variables in current, previous and next iteration
    vstate = property(fget=lambda self: self._get_vstate(),
                      fset=lambda self, value: self._set_vstate(value))
    prev_vstate = property(fget=lambda self: self._get_prev_vstate(),
                           fset=lambda self, value: self._set_prev_vstate(value))
    next_vstate = property(fget=lambda self: self._get_next_vstate(),
                           fset=lambda self, value: self._set_next_vstate(value))

    @abstractmethod
    def get_receivers_output(self):
        pass

    @abstractmethod
    def get_function(self):
        '''Returns the function (e.g., pressure or displacement) associated with
        the wave object without additional variables (e.g., PML variables)'''
        pass

    @abstractmethod
    def get_function_name(self):
        '''Returns the string representing the function of the wave object
        (e.g., "pressure" or "displacement")'''
        pass

    def update_source_expression(self, t):
        '''Update the source expression during wave propagation. This method must be
        implemented only by subclasses that make use of the source term'''
        pass

    @ensemble_propagator
    def wave_propagator(self, dt=None, final_time=None, source_nums=None):
        """Propagate the wavefield forward in time.

        Parameters
        ----------
        dt : float, optional
            Time step to use for this propagation.
            If omitted, uses ``self.dt``.
        final_time : float, optional
            Final simulation time.
            If omitted, uses ``self.final_time``.
        source_nums : list[int], optional
            Source indices to activate during this propagation. When ``None``,
            defaults to ``[0]``.

        Notes
        -----
        This method is wrapped by ``ensemble_propagator`` to support ensemble
        and spatial source parallelism.
        """
        if source_nums is None:
            source_nums = [0]
        if final_time is not None:
            self.final_time = final_time
        if dt is not None:
            self.dt = dt

        self.current_sources = source_nums
        time_integrator(self, source_nums)

    def get_dt(self):
        return self._dt

    def set_dt(self, dt):
        self._dt = dt
        if self.sources is not None:
            self.sources.update_wavelet(self)

    dt = property(fget=get_dt, fset=set_dt)

    @abstractmethod
    def rhs_no_pml(self):
        '''
        Returns the right-hand side Cofunction without PML DOFs (i.e., only
        the DOFs associated with the subspace of the original problem).
        '''
        pass

    def enable_automated_adjoint(self):
        """Enable the automated adjoint from the Firedrake adjoint module."""
        control = self.c if self.c is not None else self.initial_velocity_model
        if control is None:
            raise ValueError(
                "Set an initial velocity model before enabling the adjoint."
            )
        self.automated_adjoint = AutomatedAdjoint(control)
        self.use_vertex_only_mesh = True
        self._compute_functional = True
        self.store_forward_time_steps = False
        self.adjoint_type = AdjointType.AUTOMATED_ADJOINT

    def enable_spyro_adjoint(self):
        """Enable the Spyro implemented adjoint."""
        self.automated_adjoint = False
        self._compute_functional = True
        self.enable_store_misfit()
        self.store_forward_time_steps = True
        self.adjoint_type = AdjointType.SPYRO_ADJOINT

    @property
    def real_shot_record(self):
        """Returns the real shot record."""
        return self._real_shot_record

    @real_shot_record.setter
    def real_shot_record(self, real_data):
        """Set the real shot record.

        Parameters
        ----------
        real_data : list of firedrake functions or numpy arrays
            The real shot record data to be set.
        """
        self._real_shot_record = real_data

    @property
    def compute_functional(self):
        """Return whether the computation of the cost functional is enabled."""
        return self._compute_functional

    def enable_compute_functional(self):
        """Enable the computation of the cost functional during wave propagation."""
        self._compute_functional = True

    @property
    def store_misfit(self):
        """Return whether the storage of the misfit is enabled."""
        return self._store_misfit

    def enable_store_misfit(self):
        """Enable the storage of the misfit at each time step during wave propagation."""
        self._store_misfit = True

    def disable_store_misfit(self):
        """Disable the storage of the misfit at each time step during wave propagation."""
        self._store_misfit = False

    @property
    def functional_value(self):
        """Returns the current value of the cost functional."""
        return self._functional_value

    @functional_value.setter
    def functional_value(self, value):
        """Set the value of the cost functional."""
        self._functional_value = value
