from abc import abstractmethod, ABCMeta
import warnings

import numpy as np
import firedrake as fire

from .time_integration_central_difference import \
    _propagate_forward_central_difference as _forward_time_integrator
from ..domains.quadrature import quadrature_rules
from ..domains.space import check_function_space_type
from ..io import Model_parameters
from ..io import material_properties_io
from ..io.parallelism_wrappers import ensemble_propagator
from ..io import parallel_print
from ..io.field_logger import FieldLogger
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from .solver_parameters import get_default_linear_solver_parameters
from ..utils.error_management import enum_parameter_error
from ..utils.typing import (AdjointType, FunctionalEvaluationMode, AbsorbingBCsType,
                            ElasticMaterialParameter,
                            ElasticMaterialParameterization, LayerShapeType,
                            WaveType)
from .modal.modal_sol import Modal_Solver
from .automatic_differentiation_solver import AutomatedAdjoint


fire.set_log_level(fire.ERROR)


_ELASTIC_PARAMETER_ALIASES = {
    ElasticMaterialParameter.LAMBDA: ("lmbda", "lame_first"),
    ElasticMaterialParameter.MU: ("lame_second",),
}


class Wave(Model_parameters, metaclass=ABCMeta):
    """
    Base class for wave equation solvers.

    Attributes:
    -----------
    comm : `object`
        An object representing the communication interface.
    boundary_idx_map: dict
        Mapping of boundary IDs for applying absorbing boundary conditions.
    initial_velocity_model: `Firedrake.Function`
        Snapshot of the first initialized acoustic velocity model.
    function_space: firedrake function space
        Function space for the wave equation.
    current_time: float
        Current time of the simulation.
    solver_parameters: `dict` or `None`
        PETSc/KSP options passed to Firedrake's linear solver.
    real_shot_record: `Firedrake.Function`
        Real shot record.
    mesh: `Firedrake.Mesh`
        Mesh used in the simulation (2D or 3D).
    mesh_parameters : `Python object`
        Contains mesh parameters.
    mesh_x: `ufl.geometry.SpatialCoordinate`
        Symbolic coordinate x of the mesh object.
    mesh_y: `ufl.geometry.SpatialCoordinate`
        Symbolic coordinate y of the mesh object.
    mesh_z : `ufl.geometry.SpatialCoordinate`
        Symbolic coordinate z of the mesh object.
    sources: Sources object
        Contains information about sources.
    receivers: Receivers object
        Contains information about receivers.
    path_case_abc : `string`
        Path to save data for the abc case study.
    path_save : `string`
        Path to save data
    mesh_ops : `meshing_operations.MeshOps` or `meshing_HABC.HABCMesh`.
        Mesh operation manager
    layer_ops : `habc.HABCLayer` or `pml_nsnc.PMLLayer`
        ABC layer operation manager.

    Methods:
    --------
    get_and_set_maximum_dt()
        Calculates and/or sets maximum dt.
    get_mass_matrix_diagonal()
        Returns diagonal of mass matrix.
    get_spatial_coordinates()
        Get the coordinates of the mesh.
    set_mesh()
        Sets or calculates new mesh.
    initialize_model_parameters()
        Sets or loads the material parameters required by the wave equation.
    set_last_solve_as_real_shot_record()
        Sets last solve as real shot record.
    Notes
    -----
    New attributes added to the wave object in mesh_parameters:
    mesh_parameters.alpha : `float`
        Ratio between the representative mesh dimensions.
    mesh_parameters.diam_mesh : `ufl.geometry.CellDiameter`
        Mesh cell diameters.
    mesh_parameters.lmin : `float`
        Minimum mesh size.
    mesh_parameters.lmax : `float`
        Maxmum mesh size.
    mesh_parameters.tol : `float`
        Tolerance for searching nodes in the mesh.
    """

    def __init__(self, dictionary=None, wave_type=WaveType.NONE, comm=None):
        """Wave object solver. Contains both the forward solver
        and gradient calculator methods.

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the Wave class.
            Default is None
        wave_type : `typing.WaveType`, optional
            The type of wave equation to solve. Default is `WaveType.NONE`
        comm : `object`, optional
            MPI communicator for parallel execution. Default is `None`.

        Returns
        -------
        None

        model_parameters : `Python object`
            Contains model parameters.
        """

        super().__init__(dictionary=dictionary, comm=comm)
        self.initial_velocity_model = None
        self._model_parameters_initialized = False
        self.gradient_mask_available = False

        # Setting wave type
        self.wave_type = enum_parameter_error("wave_type", wave_type, WaveType)

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
        self.current_time = 0.0
        self.source_expression = None  # Expression for sources using UFL (less efficient)
        self.solver_parameters = get_default_linear_solver_parameters(
            self.method
        )

        # Create or get the mesh
        self.mesh = self.get_mesh()
        self.c = None
        self.sources = None
        self.real_shot_record = None

        # Mesh manager
        self.mesh_manager()

        # Getting parameters from the mesh
        if self.mesh is not None:
            self.building_mesh_derived_paramenters()
        elif self.mesh_parameters.mesh_type == "firedrake_mesh":
            warnings.warn(
                "No mesh file, Firedrake mesh will be automatically generated."
            )
        else:
            warnings.warn("No mesh found. Please define a mesh.")

        # Creating absorbing layer manager if needed
        if self.abc_active:
            self.layer_manager()

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

        if self.abc_type != AbsorbingBCsType.HYBRID:
            self.initialize_model_parameters()
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

        # Function space type for the mesh operations
        self.mesh_ops.func_space_type = 'scalar' \
            if self.wave_type == WaveType.ISOTROPIC_ACOUSTIC else 'vector'

        # Get boundaries
        boundaries = self.get_absorbing_boundaries()

        # Build the boundary ID mapping
        # TODO: Include the logic for hypershape layer from HABC
        # TODO: Create a flag for other domains that are not of type box
        if not (
            self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE
            and self.mesh_parameters.boundary_ids_map is not None
        ):
            self.mesh_parameters.boundary_ids_map, \
                self.mesh_parameters.boundary_nodes_ids = \
                self.mesh_ops.mapping_boundary_ids(self.mesh, self.function_space,
                                                   boundaries, box_domain=True,
                                                   get_boundary_node_ids=True)

        # Get geometry parameters from mesh
        if (
            self.mesh_ops.func_space_type == 'scalar'
            and self.mesh_parameters.diam_mesh is None
        ):
            data_mesh = self.mesh_ops.representative_mesh_dimensions(self.mesh,
                                                                     self.function_space)
            self.mesh_parameters.diam_mesh = data_mesh[0]
            self.mesh_parameters.lmin = data_mesh[1]
            self.mesh_parameters.lmax = data_mesh[2]
            self.mesh_parameters.alpha = data_mesh[3]
            self.mesh_parameters.tol = data_mesh[4]

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

    def _map_sources_and_receivers(self):
        if self.source_type == "ricker":
            self.sources = Sources(self)
            self.sources.wave_type = self.wave_type
        self.receivers = Receivers(self)
        self.receivers.wave_type = self.wave_type

    def initialize_model_parameters(
        self,
        synthetic_data=None,
        *,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
        dg_velocity_model=True,
        fast_interpolate=False,
    ):
        """Initialize the material parameters required by this wave equation.

        This is the single initialization entry point for acoustic and
        isotropic-elastic waves. Existing, complete parameter sets take
        precedence over the input dictionary, making repeated calls safe after
        a model has been changed with :meth:`set_material_property` or the
        inversion-control API.

        The active wave type selects the initialization path:

        - ``ISOTROPIC_ACOUSTIC`` uses the explicit acoustic velocity arguments
          (``constant``, ``conditional``, ``expression``,
          ``velocity_model_function`` or ``new_file``); when none is given the
          velocity is loaded from the previously set file or the grid data.
        - ``ISOTROPIC_ELASTIC`` uses ``synthetic_data`` (or the
          ``synthetic_data`` section of ``input_dictionary``) and accepts
          exactly one parameterization: ``{density, lambda, mu}`` or
          ``{density, p_wave_velocity, s_wave_velocity}``. The missing derived
          fields are computed and the active control parameterization stored.

        Parameters
        ----------
        synthetic_data : dict, optional
            Elastic material declaration. When omitted, the ``synthetic_data``
            section of ``input_dictionary`` is used. Ignored for acoustic
            waves.
        constant : float, optional
            Constant acoustic velocity model.
        conditional : firedrake expression, optional
            Conditional acoustic velocity model.
        velocity_model_function : firedrake.Function, optional
            Acoustic velocity model already represented as a Function.
        expression : str, optional
            String expression defining the acoustic velocity model.
        new_file : str, optional
            SEGY, HDF5 or H5 acoustic velocity-model file.
        output : bool, optional
            Write the initialized acoustic velocity to PVD.
        dg_velocity_model : bool, optional
            Materialize a conditional acoustic model in DG0. Default is True.
        fast_interpolate : bool, optional
            Only affects file/grid velocity inputs. When ``False`` (default)
            the model read from a SEGY/HDF5 file or grid is L2-projected onto
            the finite-element space -- robust, and smooths differences between
            the model grid and the mesh. When ``True`` the model is instead
            sampled directly at the node coordinates of the space, skipping the
            projection solve: this is faster but relies on point sampling of
            the grid. Has no effect for constant/conditional/expression inputs.
            Default is False.

        Raises
        ------
        ValueError
            If acoustic sources are combined with ``synthetic_data``, if no
            acoustic velocity source is available, if elastic parameters are
            inconsistent, or if the elastic declaration is missing or has an
            unknown ``type``.
        NotImplementedError
            For wave types other than acoustic or isotropic elastic, or for a
            file-based elastic declaration.
        """
        acoustic_sources = {
            "constant": constant,
            "conditional": conditional,
            "expression": expression,
            "fire_function": velocity_model_function,
            "from_file": new_file,
        }
        has_acoustic_source = any(
            value is not None for value in acoustic_sources.values()
        )

        if self.wave_type == WaveType.ISOTROPIC_ACOUSTIC:
            self._initialize_acoustic_parameters(
                acoustic_sources,
                has_acoustic_source=has_acoustic_source,
                synthetic_data=synthetic_data,
                conditional=conditional,
                new_file=new_file,
                output=output,
                dg_velocity_model=dg_velocity_model,
                fast_interpolate=fast_interpolate,
            )
            return

        if has_acoustic_source:
            raise ValueError(
                "Explicit velocity sources are supported only for acoustic "
                "waves; use synthetic_data for elastic material parameters."
            )

        if self.wave_type != WaveType.ISOTROPIC_ELASTIC:
            raise NotImplementedError(
                "Model-parameter initialization is not implemented for "
                f"wave type {self.wave_type.name}."
            )

        self._initialize_elastic_isotropic_parameters(synthetic_data)

    def _initialize_acoustic_parameters(
        self,
        acoustic_sources,
        *,
        has_acoustic_source,
        synthetic_data,
        conditional,
        new_file,
        output,
        dg_velocity_model,
        fast_interpolate,
    ):
        """Initialize the acoustic P-wave velocity ``self.c``.

        Two situations are handled. If an explicit source was given
        (``has_acoustic_source``), the velocity is built from it through the
        shared I/O engine. Otherwise the velocity is loaded from a previously
        registered file or from the grid data attached to the mesh. In both
        cases the existing ``self.c`` Function receives the model values and
        ``initial_velocity_model`` keeps a snapshot of the first model.

        Parameters
        ----------
        acoustic_sources : dict
            Mapping of I/O engine keyword to value (``constant``,
            ``conditional``, ``expression``, ``fire_function``, ``from_file``).
            At most one entry is non-``None``.
        has_acoustic_source : bool
            Whether ``acoustic_sources`` holds a user-provided source. When
            ``False`` the velocity is loaded from file/grid data instead.
        synthetic_data : dict or None
            Elastic declaration; only checked here to reject the invalid
            combination of an acoustic source together with ``synthetic_data``.
        conditional : firedrake expression or None
            The conditional source, inspected to decide whether the model is
            materialized in DG0 (``dg_velocity_model``).
        new_file : str or None
            File path to record in ``initial_velocity_model_file`` when an
            explicit source is used.
        output : bool
            Write the initialized velocity to PVD. Forced ``True`` when
            ``self.debug_output`` is set.
        dg_velocity_model : bool
            Materialize a conditional model in DG0 rather than the wave's own
            function space.
        fast_interpolate : bool
            Forwarded to the I/O engine; see
            :meth:`initialize_model_parameters` for the meaning.

        Raises
        ------
        ValueError
            If an acoustic source is combined with ``synthetic_data`` or if no
            velocity model or file is available to load.
        """
        if self.mesh is None:
            self.set_mesh()

        velocity = None
        if has_acoustic_source:
            if synthetic_data is not None:
                raise ValueError(
                    "Use either synthetic_data or an explicit acoustic "
                    "velocity source, not both."
                )
            if self.debug_output:
                output = True

            velocity = self.set_material_property(
                "velocity",
                "scalar",
                dg_property=(
                    dg_velocity_model if conditional is not None else False
                ),
                fast_interpolate=fast_interpolate,
                **acoustic_sources,
            )
            self.initial_velocity_model_file = new_file
        elif not self._model_parameters_initialized:
            velocity_source = self.initial_velocity_model
            if velocity_source is None:
                velocity_source = self.initial_velocity_model_file
            if velocity_source is None:
                velocity_source = self.mesh_parameters.grid_velocity_data
            if velocity_source is None:
                raise ValueError(
                    "No velocity model or velocity file to load."
                )
            velocity = self._material_parameter_field(
                velocity_source,
                "velocity",
                fast_interpolate=fast_interpolate,
            )

        if velocity is None:
            return

        if velocity.function_space() == self.c.function_space():
            self.c.assign(velocity)
        else:
            self.c.interpolate(velocity)
        if self.initial_velocity_model is None:
            self.initial_velocity_model = fire.Function(
                self.c.function_space(),
                name="initial_velocity_model",
            )
            self.initial_velocity_model.assign(self.c)
        self._model_parameters_initialized = True

        if output or self.debug_output:
            fire.VTKFile("initial_velocity_model.pvd").write(
                self.c,
                name="velocity",
            )

    def _initialize_elastic_isotropic_parameters(self, synthetic_data=None):
        """Initialize the isotropic-elastic material parameters.

        The isotropic elastic model is described by exactly one of two
        parameterizations: Lame ``{density, lambda, mu}`` or velocity
        ``{density, p_wave_velocity, s_wave_velocity}``. Whichever set is
        supplied becomes the *control* parameterization (the one an inversion
        updates), and the complementary fields are derived from it.

        The five material attributes are scalar ``Function`` objects created
        with the wave function space. On the first call this method assigns the
        declared primary fields. Every call refreshes the complementary fields
        in place, so references held by variational forms remain valid.

        Parameters
        ----------
        synthetic_data : dict, optional
            Material declaration with a ``type`` key and the parameter fields.
            When ``None`` the ``synthetic_data`` section of
            ``input_dictionary`` is used. Ignored once a parameterization is
            already active.

        Raises
        ------
        ValueError
            If the declaration is missing, has an unknown ``type``, or yields an
            inconsistent mix of Lame and velocity parameters.
        NotImplementedError
            If the declaration requests a file-based initialization.
        """
        if self.mesh is None:
            self.set_mesh()

        parameterization = self._control_parameterization
        if parameterization is None:
            data = synthetic_data
            if data is None:
                data = self.input_dictionary.get("synthetic_data")
            if not isinstance(data, dict) or "type" not in data:
                raise ValueError(
                    "Input dictionary must contain ['synthetic_data']['type']."
                )
            if data["type"] == "file":
                raise NotImplementedError(
                    "File-based isotropic-elastic material initialization is "
                    "not implemented."
                )
            if data["type"] != "object":
                raise ValueError(
                    f"Invalid synthetic data type: {data['type']}"
                )

            values = {}
            for parameter in ElasticMaterialParameter:
                names = (
                    parameter.value,
                    *_ELASTIC_PARAMETER_ALIASES.get(parameter, ()),
                )
                for name in names:
                    if name in data:
                        values[parameter] = data[name]
                        break

            lame_parameters = {
                ElasticMaterialParameter.DENSITY,
                ElasticMaterialParameter.LAMBDA,
                ElasticMaterialParameter.MU,
            }
            velocity_parameters = {
                ElasticMaterialParameter.DENSITY,
                ElasticMaterialParameter.P_WAVE_VELOCITY,
                ElasticMaterialParameter.S_WAVE_VELOCITY,
            }
            if set(values) == lame_parameters:
                parameterization = ElasticMaterialParameterization.LAME
            elif set(values) == velocity_parameters:
                parameterization = ElasticMaterialParameterization.VELOCITY
            else:
                raise ValueError(
                    "Inconsistent selection of isotropic elastic wave "
                    f"parameters: {set(values)}. The valid options are "
                    "{density, lambda, mu} or "
                    "{density, p_wave_velocity, s_wave_velocity}."
                )

            fields = {
                ElasticMaterialParameter.DENSITY: self.rho,
                ElasticMaterialParameter.LAMBDA: self.lmbda,
                ElasticMaterialParameter.MU: self.mu,
                ElasticMaterialParameter.P_WAVE_VELOCITY: self.c,
                ElasticMaterialParameter.S_WAVE_VELOCITY: self.c_s,
            }
            for parameter, value in values.items():
                source = self._material_parameter_field(value, parameter)
                fields[parameter].assign(source)

            self._control_parameterization = parameterization

        if parameterization is ElasticMaterialParameterization.LAME:
            self.c.interpolate(((self.lmbda + 2*self.mu)/self.rho)**0.5)
            self.c_s.interpolate((self.mu/self.rho)**0.5)
        else:
            self.mu.interpolate(self.rho*self.c_s**2)
            self.lmbda.interpolate(self.rho*self.c**2 - 2*self.mu)
        self._model_parameters_initialized = True

    def _material_parameter_field(
        self,
        value,
        name,
        fast_interpolate=False,
    ):
        """Create a scalar material Function through the shared I/O engine.

        Normalizes ``value`` to the right keyword of
        :meth:`set_material_property` based on its Python type: scalars become
        ``constant``, Functions become ``fire_function``, strings/dicts become
        ``from_file`` (path or grid data), and anything else (a UFL/conditional
        expression) becomes ``conditional``. This is the single funnel through
        which every material field is materialized.

        Parameters
        ----------
        value : scalar, firedrake.Function, str, dict or UFL expression
            The material value to materialize. Its type selects the I/O path.
        name : str or ElasticMaterialParameter
            Property name passed to the I/O engine. Elastic callers use the
            enum; conversion to the public string name happens only here.
        fast_interpolate : bool, optional
            Forwarded to the I/O engine for ``from_file`` inputs; see
            :meth:`initialize_model_parameters`. Default is ``False``.

        Returns
        -------
        firedrake.Function
            The materialized scalar field.

        Raises
        ------
        ValueError
            If no function space exists yet (``set_mesh()`` was not called).
        """
        if self.function_space is None:
            raise ValueError(
                "A function space is required to initialize model parameters. "
                "Call set_mesh() first.",
            )

        property_name = (
            name.value if isinstance(name, ElasticMaterialParameter) else name
        )
        source = {}
        if np.isscalar(value):
            source["constant"] = value
        elif isinstance(value, fire.Function):
            source["fire_function"] = value
        elif isinstance(value, (str, dict)):
            source["from_file"] = value
        else:
            source["conditional"] = value

        return self.set_material_property(
            property_name,
            "scalar",
            fast_interpolate=fast_interpolate,
            **source,
        )

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
        dt_solver = Modal_Solver(self.dimension, method=method, calc_max_dt=True)
        max_dt = dt_solver.estimate_timestep(c, self.function_space, self.final_time,
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

    def set_material_property(self, *args, **kwargs):
        """Wrapper for material_properties_io.set_material_property."""
        return material_properties_io.set_material_property(
            self,
            *args,
            **kwargs
        )

    def set_material_properties(self, *args, **kwargs):
        """Deprecated alias for :meth:`set_material_property`.

        .. deprecated::
            Use :meth:`set_material_property` instead. This wrapper forwards
            every argument unchanged and will be removed in a future release.
        """
        warnings.warn(
            "set_material_properties() is deprecated; use "
            "set_material_property() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.set_material_property(*args, **kwargs)

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
        self.initialize_model_parameters()
        if self.c is None:
            raise ValueError(
                "self.c must be set before enabling automated adjoint."
                "Please set the velocity model using initialize_model_parameters()"
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
        self.adjoint_type = AdjointType.IMPLEMENTED_ADJOINT
        self.store_forward_time_steps = True

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

    def mesh_manager(self):
        """Create the mesh operations manager for the wave solver."""

        # Domain dimensions
        domain_dim = self.domain_dimensions()

        if self.abc_active:  # If ABC scheme is used
            from ..meshing.meshing_habc import HABCMesh
            self.mesh_ops = HABCMesh(domain_dim, dimension=self.dimension,
                                     quadrilateral=self.mesh_parameters.quadrilateral,
                                     comm=self.mesh_parameters.comm)

        else:  # If no ABC scheme is used
            from ..meshing.meshing_operations import MeshOps
            self.mesh_ops = MeshOps(domain_dim, dimension=self.dimension,
                                    quadrilateral=self.mesh_parameters.quadrilateral,
                                    comm=self.mesh_parameters.comm)

    def layer_manager(self):
        """Return the layer operations manager for the wave solver."""

        # Domain dimensions
        domain_dim = self.domain_dimensions()

        # Timestep of the simulation. It is `None` if the response is not 'transient'.
        time_step = None if self.analysis != "transient" else self.dt

        if self.abc_type == AbsorbingBCsType.PML:  # PML
            from ..pml.pml_nsnc import PMLLayer
            self.layer_ops = PMLLayer(domain_dim, frequency=self.frequency,
                                      dt=time_step, dimension=self.dimension,
                                      quadrilateral=self.mesh_parameters.quadrilateral,
                                      func_space_type=self.mesh_ops.func_space_type,
                                      abc_reference_freq=self.abc_reference_freq,
                                      output_folder=self.output_folder, comm=self.comm)

        if self.abc_type == AbsorbingBCsType.HYBRID:  # HABC
            from ..habc.habc import HABCLayer
            self.layer_ops = HABCLayer(domain_dim, frequency=self.frequency,
                                       dt=time_step, dimension=self.dimension,
                                       quadrilateral=self.mesh_parameters.quadrilateral,
                                       func_space_type=self.mesh_ops.func_space_type,
                                       abc_boundary_layer_shape=self.abc_boundary_layer_shape,
                                       abc_reference_freq=self.abc_reference_freq,
                                       abc_degree_type=self.abc_degree_type,
                                       abc_deg_layer=self.abc_deg_layer,
                                       output_folder=self.output_folder, comm=self.comm)

        # Identifier for the current case study
        if self.abc_type in [AbsorbingBCsType.PML, AbsorbingBCsType.HYBRID]:
            self.case_abc = self.layer_ops.case_abc
            self.path_save = self.layer_ops.path_save
            self.path_case_abc = self.layer_ops.path_case_abc

    @abstractmethod
    def get_control_parameters(self):
        """Return inversion controls exposed by a concrete wave solver.

        Subclasses override this method when they can participate in inversion
        workflows. The base class raises because a generic ``spyro.solvers.Wave`` does not
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
