from abc import abstractmethod, ABCMeta
import warnings
import firedrake as fire
import spyro.meshing.meshing_operations as mshops

from .time_integration_central_difference import \
    central_difference as time_integrator
from ..domains.quadrature import quadrature_rules
from ..domains.space import check_function_space_type
from ..io import Model_parameters
from ..io import material_properties_io
from ..io.basicio import ensemble_propagator
from ..io import parallel_print
from ..io.field_logger import FieldLogger
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from ..utils.typing import WaveType
from .solver_parameters import get_default_parameters_for_method
from ..utils import eval_functions_to_ufl
from .modal.modal_sol import Modal_Solver

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
        self._receivers_output = None
        self.current_time = 0.0
        self.set_solver_parameters()

        self.mesh = self.get_mesh()
        self.c = None
        self.sources = None

        # Creating mesh operations manager
        self.mesh_ops = mshops.MeshOps(
            self.domain_dimensions(), dimension=self.dimension,
            quadrilateral=self.mesh_parameters.quadrilateral,
            comm=self.mesh_parameters.comm)

        if self.mesh is not None:
            self.building_mesh_derived_paramenters()
        elif self.mesh_parameters.mesh_type == "firedrake_mesh":
            warnings.warn(
                "No mesh file, Firedrake mesh will be automatically generated."
            )
        else:
            warnings.warn("No mesh found. Please define a mesh.")
        # Expression to define sources through UFL (less efficient)
        self.source_expression = None

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
        boundaries = [self.absorb_top, self.absorb_bottom,
                      self.absorb_right, self.absorb_left]
        if self.dimension == 3:
            boundaries.extend([self.absorb_front,
                               self.absorb_back])

        # Build the boundary ID mapping
        self.mesh_parameters.boundary_idx_map = \
            self.mesh_ops.mapping_boundary_ids(self.mesh, self.function_space,
                                               boundaries, box_domain=True)

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
    def wave_propagator(self, dt=None, final_time=None, source_nums=[0]):
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

        self.current_sources = source_nums
        usol, usol_recv = time_integrator(self, source_nums)

        return usol, usol_recv

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
    def forward_solution_receivers(self):
        return self._receivers_output

    @forward_solution_receivers.setter
    def forward_solution_receivers(self, value):
        self._receivers_output = value

    @property
    def receivers_output(self):
        return self._receivers_output

    @receivers_output.setter
    def receivers_output(self, value):
        self._receivers_output = value
