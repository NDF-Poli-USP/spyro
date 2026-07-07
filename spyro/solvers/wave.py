from abc import abstractmethod, ABCMeta
import warnings
import firedrake as fire
import spyro.meshing.meshing_operations as mshops

from .time_integration_central_difference import \
    _propagate_forward_central_difference as _forward_time_integrator
from ..domains.quadrature import quadrature_rules
from ..domains.space import check_function_space_type
from ..io import Model_parameters
from ..io import material_properties_io
from ..io.basicio import ensemble_propagator
from ..io import parallel_print
from ..io.field_logger import FieldLogger
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from ..utils.typing import AdjointType, WaveType, FunctionalEvaluationMode
from .solver_parameters import get_default_parameters_for_method
from ..utils import eval_functions_to_ufl
from .modal.modal_sol import Modal_Solver
from .automatic_differentiation_solver import AutomatedAdjoint

fire.set_log_level(fire.ERROR)


class Wave(Model_parameters, metaclass=ABCMeta):
    """
    Base class for wave equation solvers.

    Attributes:
    -----------
    comm : `object`
        An object representing the communication interface
    boundary_idx_map: dict
        Mapping of boundary IDs for applying absorbing boundary conditions
    initial_velocity_model: firedrake function
        Initial velocity model
    function_space: firedrake function space
        Function space for the wave equation
    current_time: float
        Current time of the simulation
    solver_parameters: Python object
        Contains solver parameters
    real_shot_record: firedrake function
        Real shot record
    mesh: firedrake mesh
        Mesh used in the simulation (2D or 3D)
    mesh_x: `ufl.geometry.SpatialCoordinate`
        Symbolic coordinate x of the mesh object
    mesh_y: `ufl.geometry.SpatialCoordinate`
        Symbolic coordinate y of the mesh object
    mesh_z : `ufl.geometry.SpatialCoordinate`
        Symbolic coordinate z of the mesh object
    sources: Sources object
        Contains information about sources
    receivers: Receivers object
        Contains information about receivers

    Methods:
    --------
    get_and_set_maximum_dt()
        Calculates and/or sets maximum dt
    get_mass_matrix_diagonal()
        Returns diagonal of mass matrix
    get_spatial_coordinates()
        Get the coordinates of the mesh.
    set_mesh()
        Sets or calculates new mesh
    set_initial_velocity_model()
        Sets initial velocity model
    set_last_solve_as_real_shot_record()
        Sets last solve as real shot record
    set_solver_parameters()
        Sets new or default solver parameters
    """

    def __init__(self, dictionary=None, comm=None):
        """Wave object solver. Contains both the forward solver
        and gradient calculator methods.

        Parameters:
        -----------
        comm: MPI communicator

        model_parameters: Python object
            Contains model parameters
        """
        super().__init__(dictionary=dictionary, comm=comm)
        self.initial_velocity_model = None
        self.gradient_mask_available = False
        self.wave_type = WaveType.NONE

        self.function_space = None
        self.dg0_scalar_function_space = None
        self.dg0_vector_function_space = None
        self.scalar_function_space = None
        self.vector_function_space = None
        self.tensor_function_space0 = None
        self.tensor_function_space1 = None
        self._forward_solution_receivers = None
        self._store_forward_time_steps = False
        self.forward_solution = None
        self.adjoint_solution = None
        self.adjoint_type = AdjointType.NONE
        self.automated_adjoint = None
        self.functional_value = None
        self.misfit = None
        self.forward_residual_form = None
        self.forward_residual_states = None
        self.current_time = 0.0
        # Expression to define sources through UFL (less efficient)
        self.source_expression = None
        self.adjoint_source_function = None
        self.set_solver_parameters()

        # Create or get the mesh
        self.mesh = self.get_mesh()
        self.c = None
        self.sources = None
        self.real_shot_record = None

        self.set_solver_parameters()

        # Creating mesh operations manager
        self.mesh_ops = mshops.MeshOps(
            self.domain_dimensions(), dimension=self.dimension,
            quadrilateral=self.mesh_parameters.quadrilateral,
            comm=self.mesh_parameters.comm)

        # Getting parameters from the mesh
        if self.mesh is not None:
            self.building_mesh_derived_paramenters()
        elif self.mesh_parameters.mesh_type == "firedrake_mesh":
            warnings.warn(
                "No mesh file, Firedrake mesh will be automatically generated."
            )
        else:
            warnings.warn("No mesh found. Please define a mesh.")

        # Logger
        self.field_logger = FieldLogger(self.comm,
                                        self.input_dictionary["visualization"])
        self.field_logger.add_field("forward", self.get_function_name(),
                                    lambda: self.get_function())

    def forward_solve(self):
        """Solves the forward problem."""

        parallel_print("\nSolving Forward Problem", comm=self.comm)

        if self.function_space is None:
            self.force_rebuild_function_space()

        if self.abc_boundary_layer_type != "hybrid":
            self._initialize_model_parameters()
        self.matrix_building()
        self.wave_propagator()

    def force_rebuild_function_space(self):
        if self.mesh is None:
            self.mesh = self.get_mesh()
        self.building_mesh_derived_paramenters()

    @abstractmethod
    def matrix_building(self):
        """Builds the matrix for the forward problem."""
        pass

    def get_absorbing_boundaries(self):
        """Get the absorbing boundaries for the problem.

        Parameters:
        -----------
        None

        Returns:
        --------
        boundaries : `tuple`
            Tuple containing the boundary boolean labels for applying absorbing BCs.
            - (absorb_top, absorb_bottom, absorb_right, absorb_left) for 2D
            - (absorb_top, absorb_bottom, absorb_right,
                absorb_left, absorb_front, absorb_back) for 3D
        """
        boundaries = (self.absorb_top, self.absorb_bottom,
                      self.absorb_right, self.absorb_left)

        if self.dimension == 3:
            boundaries += (self.absorb_front, self.absorb_back,)

        return boundaries

    def building_mesh_derived_paramenters(self):
        """Build parameters that are derived from the mesh."""
        coordinates = self.mesh_ops._set_spatial_coordinates(self.mesh)
        self.mesh_z, self.mesh_x = coordinates[0], coordinates[1]
        if self.dimension == 3:
            self.mesh_y = coordinates[2]
        self._build_function_space()
        self._map_sources_and_receivers()

        # TODO: Create a flag for other domains that are not of type box
        if self.mesh_ops.func_space_type is None:
            self.mesh_ops.func_space_type = 'scalar' \
                if len(self.function_space.value_shape) == 0 else 'vector'

        # Build the boundary ID mapping
        # TODO: Include the logic for hypershape layer from HABC
        boundaries = self.get_absorbing_boundaries()
        if not (hasattr(self, 'abc_boundary_layer_shape')
                and hasattr(self.mesh_parameters, 'boundary_ids_map')
                and self.abc_boundary_layer_shape == 'hypershape'):
            self.mesh_parameters.boundary_ids_map, \
                self.mesh_parameters.boundary_nodes_ids = \
                self.mesh_ops.mapping_boundary_ids(self.mesh, self.function_space,
                                                   boundaries, box_domain=True,
                                                   get_boundary_node_ids=True)

    def set_mesh(
            self,
            user_mesh=None,
            input_mesh_parameters=None,
    ):
        """
        Set the mesh for the solver.

        Args:
            user_mesh (optional): User-defined mesh. Defaults to None.
            mesh_parameters (optional): Parameters for generating a mesh.
            Defaults to None.
        """

        if input_mesh_parameters is None:
            input_mesh_parameters = {}

        super().set_mesh(
            user_mesh=user_mesh,
            input_mesh_parameters=input_mesh_parameters,
        )

        self.mesh = self.get_mesh()
        self.building_mesh_derived_paramenters()

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
        """
        Get the coordinates of the mesh.

        Parameters
        ----------
        None

        Returns
        -------
        mesh_z : `ufl.geometry.SpatialCoordinate`
            Symbolic coordinate z of the mesh object
        mesh_x: `ufl.geometry.SpatialCoordinate`
            Symbolic coordinate x of the mesh object
        mesh_y: `ufl.geometry.SpatialCoordinate`
            Symbolic coordinate y of the mesh object
        """
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
        # Resseting old velocity model
        self.initial_velocity_model = None
        self.initial_velocity_model_file = None
        if new_file is not None:
            self.initial_velocity_model_file = new_file
        # If no mesh is set, we have to do it beforehand
        if self.mesh is None:
            self.set_mesh()

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
            V = self.function_space
            vp = eval_functions_to_ufl.generate_ufl_functions(
                self.mesh, expression, self.dimension)
            self.initial_velocity_model = fire.Function(
                V, name="velocity").interpolate(vp)

        elif velocity_model_function is not None:
            self.initial_velocity_model = velocity_model_function
        elif new_file is not None:
            self.initial_velocity_model_file = new_file
            self._initialize_model_parameters()  # TODO in PR206
        elif constant is not None:
            V = self.function_space
            vp = fire.Function(V, name="velocity")
            vp.interpolate(fire.Constant(constant))
            self.initial_velocity_model = vp
        else:
            raise ValueError(
                "Please specify either a conditional, expression, "
                "firedrake function or new file name (segy or hdf5)."
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
        function_space_type = check_function_space_type(self.function_space)
        if function_space_type == "scalar":
            self.scalar_function_space = self.function_space
        elif function_space_type == "mixed":
            scalar_function_space_type = check_function_space_type(self.function_space.sub(0))
            if scalar_function_space_type != "scalar":
                raise ValueError("Do not change mixed space order, use scalar first!!! (ノಠ益ಠ)ノ彡┻━┻")
            self.scalar_function_space = self.function_space.sub(0)
            self.vector_function_space = self.function_space.sub(1)
        elif function_space_type == "vector":
            self.vector_function_space = self.function_space

        quad_rule, k_rule, s_rule = quadrature_rules(self.function_space)
        self.quadrature_rule = quad_rule
        self.stiffness_quadrature_rule = k_rule
        self.surface_quadrature_rule = s_rule

    def get_and_set_maximum_dt(self, fraction=0.7,
                               estimate_max_eigenvalue=False):
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

        if self.c is None:
            c = self.initial_velocity_model
        else:
            c = self.c

        # Maximum timestep size
        method = 'ANALYTICAL' if estimate_max_eigenvalue else 'ARNOLDI'
        dt_solver = Modal_Solver(self.dimension, method=method,
                                 calc_max_dt=True)
        max_dt = dt_solver.estimate_timestep(c, self.function_space,
                                             self.final_time,
                                             quad_rule=self.quadrature_rule,
                                             fraction=fraction)
        self.dt = max_dt

        return max_dt

    def get_mass_matrix_diagonal(self):
        """Builds a section of the mass matrix for debugging purposes."""
        A = fire.assemble(self.lhs, mat_type="aij")
        petsc_matrix = A.petscmat
        diagonal = petsc_matrix.getDiagonal()
        return diagonal.array

    def set_last_solve_as_real_shot_record(self):
        if self.current_time == 0.0:
            raise ValueError("No previous solve to set as real shot record.")
        self.real_shot_record = self.forward_solution_receivers

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
    def get_forward_solution_receivers(self):
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
        """
        Propagate the wave forward in time.
        Currently uses central differences.

        Parameters:
        -----------
        dt: Python 'float' (optional)
            Time step to be used explicitly. If not mentioned uses the default,
            that was estabilished in the wave object.
        final_time: Python 'float' (optional)
            Time which simulation ends. If not mentioned uses the default,
            that was estabilished in the wave object.
        source_nums: list of int (optional)
            List of source numbers to be simulated. If not mentioned, simulates all sources.

        Returns:
        --------
        usol: Firedrake 'Function'
            Wavefield at the final time.
        u_rec: numpy array
            Wavefield at the receivers across the timesteps.
        """
        if final_time is not None:
            self.final_time = final_time
        if dt is not None:
            self.dt = dt
        if source_nums is None:
            source_nums = [0]
        self.current_sources = source_nums
        _forward_time_integrator(self, source_nums)

    def get_dt(self):
        return self._dt

    def set_dt(self, dt):
        self._dt = dt
        if self.sources is not None:
            self.sources.update_wavelet(self)

    dt = property(fget=get_dt, fset=set_dt)

    @abstractmethod
    def rhs_no_pml(self):
        """
        Return the right-hand side Cofunction without PML DOFs (i.e., only
        the DOFs associated with the subspace of the original problem).
        """
        pass

    def reset_adjoint_state(self):
        """Reset the time-stepping registers used by the adjoint solve."""
        self.prev_vstate.assign(0.0)
        self.vstate.assign(0.0)
        self.next_vstate.assign(0.0)

        try:
            older_state = self.u_nm2
        except AttributeError:
            return

        if older_state is not None:
            older_state.assign(0.0)

    def get_adjoint_receiver_source_space(self):
        """Return the state-space component where receiver misfit is injected.

        Acoustic solvers inject the misfit into their scalar (pressure) space.
        Solvers without a dedicated scalar space (e.g. elastic) fall back to
        the full solution space.
        """
        get_scalar_space = getattr(self, "get_scalar_function_space", None)
        if get_scalar_space is None:
            # Solver does not expose a scalar space (e.g. elastic).
            return self.function_space
        try:
            return get_scalar_space()
        except ValueError:
            # Acoustic solver whose scalar space has not been created yet.
            return self.function_space

    def set_forward_residual_form(
        self, residual_form, live_states, state_space=None,
        state_name="residual state",
    ):
        """Expose a forward residual for UFL adjoint differentiation.

        The residual is rewritten with formal state ``Function`` objects that
        are independent of the live time-stepping registers.  During adjoint
        replay these formal states are assigned from the stored forward
        solution before differentiating

            R(u^{n+1}, u^n, u^{n-1}; m)

        with respect to the state and control parameters.

        Parameters
        ----------
        residual_form : ufl.Form
            Forward time-step residual form.
        live_states : tuple
            Objects representing ``u^{n+1}``, ``u^n`` and ``u^{n-1}`` in
            ``residual_form``.
        state_space : firedrake.FunctionSpace, optional
            Space for the formal residual states.  If omitted, the space is
            inferred from the first live state.
        state_name : str, optional
            Base name used for the formal residual state functions.

        Notes
        -----
        This currently assumes a three-level central-difference residual,
        ``R(u^{n+1}, u^n, u^{n-1}; m)``, hence exactly three ``live_states``.
        Other integrators (Newmark, Runge-Kutta, staggered leapfrog, schemes
        with memory) would need a different number of formal states and a
        matching adjoint advance; see ``docs/GENERALIZED_UFL_ADJOINT_NOTES.md``
        (section 7).
        """
        if len(live_states) != 3:
            raise ValueError("Expected live states (np1, n, nm1).")

        if state_space is None:
            try:
                state_space = live_states[0].function_space()
            except AttributeError as exc:
                raise ValueError(
                    "state_space is required when it cannot be inferred from "
                    "the first live state."
                ) from exc

        residual_states = (
            fire.Function(state_space, name=f"{state_name} t+dt"),
            fire.Function(state_space, name=state_name),
            fire.Function(state_space, name=f"{state_name} t-dt"),
        )
        self.forward_residual_states = residual_states
        self.forward_residual_form = fire.replace(
            residual_form,
            dict(zip(live_states, residual_states)),
        )

    def get_adjoint_source(self):
        """Return the cofunction used as the adjoint equation source.

        ``source_function`` is reserved for the forward problem. This method
        returns a distinct cofunction used by the adjoint problem, with the same
        dual space by default.
        """
        if self.adjoint_source_function is None:
            self.adjoint_source_function = fire.Cofunction(
                self.source_function.function_space()
            )
        return self.adjoint_source_function

    def set_adjoint_source(self, misfit_form):
        """Assign the misfit form into the adjoint source space.

        Parameters
        ----------
        misfit_form : firedrake.Cofunction
            Cofunction representing the derivative of the misfit functional with
            respect to the wave state.
        """
        adjoint_source = self.get_adjoint_source()
        if adjoint_source.function_space() == misfit_form.function_space():
            adjoint_source.assign(misfit_form)
            return

        # ``sub(0)`` only makes sense on a mixed (e.g. PML) adjoint source.
        # On a non-mixed source it fails in a backend-dependent way, so the
        # broad catch is intentional: translate any such failure into a single
        # clear "incompatible spaces" error.
        try:
            adjoint_source_component = adjoint_source.sub(0)
        except (AttributeError, IndexError, ValueError) as exc:
            raise ValueError(
                "Misfit form space is incompatible with the adjoint source "
                "space."
            ) from exc

        if adjoint_source_component.function_space() != misfit_form.function_space():
            raise ValueError(
                "Misfit form space is incompatible with the first component of "
                "the adjoint source space."
            )

        adjoint_source.assign(0.0)
        adjoint_source_component.assign(misfit_form)

    def set_material_properties(self, *args, **kwargs):
        """Wrapper for material_properties_io.set_material_property."""
        return material_properties_io.set_material_property(
            self,
            *args,
            **kwargs
        )

    def set_material_property(self, *args, **kwargs):
        """Backward-compatible alias for set_material_properties."""
        return self.set_material_properties(*args, **kwargs)

    @property
    def store_forward_time_steps(self):
        return self._store_forward_time_steps

    @store_forward_time_steps.setter
    def store_forward_time_steps(self, value):
        self._store_forward_time_steps = value

    def enable_automated_adjoint(self):
        self.store_forward_time_steps = False
        self.enable_compute_functional(
            mode=FunctionalEvaluationMode.PER_TIMESTEP
        )
        self.adjoint_type = AdjointType.AUTOMATED_ADJOINT
        self.use_vertex_only_mesh = True
        self._initialize_model_parameters()
        if self.c is None:
            raise ValueError(
                "self.c must be set before enabling automated adjoint."
                "Please set the velocity model using set_initial_velocity_model()"
                "or set c directly."
            )
        controls = self.c
        # ``self.comm`` is the Firedrake ``Ensemble`` distributing the shots
        # across ensemble members. It is forwarded to ``AutomatedAdjoint`` so
        # that the reduced functional is built as an
        # ``EnsembleReducedFunctional``, summing the per-shot functionals and
        # gradients over the ensemble communicator.
        self.automated_adjoint = AutomatedAdjoint(self.comm, controls)
        self.functional_value = None
        self.misfit = None

    def enable_implemented_adjoint(self):
        """Switch the solver into implemented-adjoint (UFL-derived) mode.

        Side effects, required before a backward/gradient solve:

        - selects :attr:`AdjointType.IMPLEMENTED_ADJOINT`;
        - stores the forward field at every gradient-sampling step
          (``store_forward_time_steps``) so the adjoint replay can reassign it;
        - enables the vertex-only-mesh receiver path used to inject the misfit
          into the adjoint source;
        - forces functional evaluation to ``PER_TIMESTEP`` so the functional is
          accumulated during the forward solve.
        """
        self.adjoint_type = AdjointType.IMPLEMENTED_ADJOINT
        self.store_forward_time_steps = True
        self.use_vertex_only_mesh = True
        if self.functional_evaluation_mode is not FunctionalEvaluationMode.PER_TIMESTEP:
            self.enable_compute_functional(
                mode=FunctionalEvaluationMode.PER_TIMESTEP
            )

    def _prepare_implemented_adjoint(self, misfit=None, forward_solution=None):
        """Enable the implemented adjoint and ensure misfit + forward solution.

        Shared ``gradient_solve`` preamble for the acoustic and elastic
        solvers. It turns on the implemented-adjoint bookkeeping, stores the
        supplied ``misfit``, makes sure a stored forward solution is available,
        and, when no misfit was given, falls back to
        ``real_shot_record - forward_solution_receivers``.

        Parameters
        ----------
        misfit : optional
            Precomputed receiver misfit. If ``None`` it is derived from the
            stored real shot record.
        forward_solution : optional
            Stored forward solution to reuse. If ``None`` and none is stored,
            a fresh forward solve is run.
        """
        self.enable_implemented_adjoint()
        if misfit is not None:
            self.misfit = misfit

        if forward_solution is not None:
            self.forward_solution = forward_solution
        elif not self.forward_solution:
            # Only re-run when ``self.forward_solution`` is empty — either it
            # was never run, or it ran before ``enable_implemented_adjoint()``
            # (so ``store_forward_time_steps`` was False and nothing was
            # stored).
            #
            # IMPORTANT: for the multi-source FWI path, ``ensemble_gradient``
            # invokes ``gradient_solve`` once per source after
            # ``switch_serial_shot`` loads that source's stored forward
            # solution into ``self.forward_solution``. Calling
            # ``forward_solve()`` here would discard the per-source data and
            # run a fresh (multi-source) ensemble forward solve, leaving the
            # backward propagator with the wrong forward state and producing an
            # incorrect gradient.
            if misfit is None:
                self.forward_solve()
            else:
                functional_mode = self.functional_evaluation_mode
                self._functional_evaluation_mode = None
                try:
                    self.forward_solve()
                finally:
                    self._functional_evaluation_mode = functional_mode

        if self.misfit is None:
            if self.real_shot_record is None:
                raise ValueError(
                    "Please load or calculate a real shot record first"
                )
            self.misfit = (
                self.real_shot_record - self.forward_solution_receivers
            )

    @property
    def forward_solution_receivers(self):
        return self._forward_solution_receivers

    @forward_solution_receivers.setter
    def forward_solution_receivers(self, value):
        self._forward_solution_receivers = value

    def enable_compute_functional(
        self, mode=FunctionalEvaluationMode.AFTER_SOLVE
    ):
        """Enable functional evaluation during forward solves.

        Parameters:
        -----------
        mode: FunctionalEvaluationMode, optional
            The mode in which to evaluate the functional.
            Default is :attribute:`FunctionalEvaluationMode.AFTER_SOLVE`.
        """
        # Create the Wave attributes required to compute functional.
        self.functional_evaluation_mode = mode

    @property
    def functional_evaluation_mode(self):
        """Get the current functional evaluation mode."""
        try:
            return self._functional_evaluation_mode
        except AttributeError:
            return None

    @functional_evaluation_mode.setter
    def functional_evaluation_mode(self, mode: FunctionalEvaluationMode):
        if not isinstance(mode, FunctionalEvaluationMode):
            raise ValueError(
                f"Invalid functional evaluation mode: {mode}. "
                f"Expected an instance of FunctionalEvaluationMode enum."
            )
        self._functional_evaluation_mode = mode
        self.functional_value = None
        self.misfit = None

    @abstractmethod
    def get_control_parameters(self):
        """Return inversion controls exposed by a concrete wave solver.

        Subclasses override this method when they can participate in inversion
        workflows. The base class raises because a generic ``Wave`` does not
        know which physical parameters should be optimized.

        Returns
        -------
        object
            Solver-specific control structure.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.

        Examples
        --------
        ``AcousticWave.get_control_parameters()`` returns the velocity model;
        an elastic solver may return a dictionary of material parameters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose inversion control parameters.",
        )

    @abstractmethod
    def set_control_parameters(self, controls):
        """Assign inversion controls on a concrete wave solver.

        Parameters
        ----------
        controls : object
            Solver-specific control structure.

        Returns
        -------
        None
            Concrete subclasses assign the controls in-place.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.

        Examples
        --------
        ``AcousticWave.set_control_parameters(vp)`` assigns a velocity model;
        elastic solvers expect a dictionary keyed by material-parameter enums.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot assign inversion control parameters.",
        )

    @abstractmethod
    def gradient_solve(self, guess=None, misfit=None, forward_solution=None):
        """Compute an adjoint gradient for inversion.

        Concrete wave solvers override this method when they provide the
        adjoint-state machinery required by FWI. The base implementation raises
        because a generic ``Wave`` does not define the physical model-specific
        gradient equation.

        Parameters
        ----------
        guess : firedrake.Function, optional
            Control value used by solvers that accept an explicit guess.
        misfit : array_like, optional
            Difference between observed and simulated receiver data.
        forward_solution : firedrake.Function, optional
            Forward wavefield used by adjoint solvers that need it explicitly.

        Returns
        -------
        firedrake.Function
            Gradient of the objective functional with respect to the active
            control.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement gradient_solve().",
        )

    @abstractmethod
    def get_control_parameter_function_space(self):
        """Return the function space used by inversion controls.

        Subclasses override this method to tell the FWI driver where scalar
        controls should live when constants or expressions need to be converted
        to Firedrake ``Function`` objects.

        Returns
        -------
        firedrake.FunctionSpace
            Solver-specific control function space.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.

        Examples
        --------
        Acoustic controls use the acoustic pressure/velocity function space;
        elastic material controls use a scalar material-parameter space.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a control parameter function space.",
        )
