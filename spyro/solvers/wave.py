from abc import abstractmethod, ABCMeta
import warnings
import firedrake as fire
from numpy import log10, ones
from numpy.random import uniform
from os import getcwd
from os.path import splitext

from .time_integration_central_difference import \
    central_difference as time_integrator
from ..domains.quadrature import quadrature_rules
from ..io import Model_parameters, interpolate
from ..io.basicio import ensemble_propagator
from ..io.field_logger import FieldLogger
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from .solver_parameters import get_default_parameters_for_method
from ..utils import error_management
from ..utils import eval_functions_to_ufl
from .modal.modal_sol import Modal_Solver

try:
    from SeismicMesh import write_velocity_model
    SEISMIC_MESH_AVAILABLE = True
except ImportError:
    SEISMIC_MESH_AVAILABLE = False

fire.set_log_level(fire.ERROR)


class Wave(Model_parameters, metaclass=ABCMeta):
    """
    Base class for wave equation solvers.

    Attributes:
    -----------
    comm: MPI communicator

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
    mesh_z: symbolic coordinate z of the mesh object
    mesh_x: symbolic coordinate x of the mesh object
    mesh_y: symbolic coordinate y of the mesh object
    sources: Sources object
        Contains information about sources
    receivers: Receivers object
        Contains information about receivers

    Methods:
    --------
    set_mesh()
        Sets or calculates new mesh
    set_solver_parameters()
        Sets new or default solver parameters
    get_spatial_coordinates()
        Returns spatial coordinates of mesh
    set_initial_velocity_model()
        Ssets initial velocity model
    get_and_set_maximum_dt()
        Calculates and/or sets maximum dt
    get_mass_matrix_diagonal()
        Returns diagonal of mass matrix
    set_last_solve_as_real_shot_record()
        Sets last solve as real shot record
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

        self.function_space = None
        self.forward_solution_receivers = None
        self.current_time = 0.0
        self.set_solver_parameters()

        self.mesh = self.get_mesh()
        self.c = None
        self.sources = None
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
        """Solves the forward problem."""

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

    def set_mesh(self, user_mesh=None,
                 input_mesh_parameters=None):
        """
        Set the mesh for the solver.

        Args:
            user_mesh (optional): User-defined mesh. Defaults to None.
            mesh_parameters (optional): Parameters for generating a mesh.
            Defaults to None.
        """

        if input_mesh_parameters is None:
            input_mesh_parameters = {}

        super().set_mesh(user_mesh=user_mesh,
                         input_mesh_parameters=input_mesh_parameters)

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
            dg_velocity_model=True):
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
            V = self.function_space
            vp = eval_functions_to_ufl.generate_ufl_functions(
                self.mesh, expression, self.dimension)
            self.initial_velocity_model = fire.Function(
                V, name="velocity").interpolate(vp)

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

    def get_and_set_maximum_dt(self, fraction=0.7,
                               estimate_max_eigenvalue=False):
        '''
        Calculates and sets the maximum stable time step (dt) for the wave solver.

        Args:
            fraction (float, optional):
                Fraction of the estimated time step to use. Defaults to 0.7.
            estimate_max_eigenvalue (bool, optional):
                Whether to estimate the maximum eigenvalue. Defaults to False.

        Returns:
            float: The calculated maximum time step (dt).
        '''

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
        A = self.solver.A
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
        '''
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
        '''
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
        '''
        Return the right-hand side Cofunction without PML DOFs (i.e., only
        the DOFs associated with the subspace of the original problem).
        '''
        pass

    def define_property_function_space(self, func_space_type, dg_property,
                                       shape_func_space=None):
        '''
        Define the function space for a material property.

        Parameters:
        -----------
        func_space_type, `str`
            Type of function space for the material property.
            Options: 'scalar', 'vector' or 'tensor'
        dg_property: `bool`
            If True, uses a DG0 function space for conditional and
            expression inputs
        shape_func_space: `tuple`, optional
            Shape of the function space for only tensorial material property.
            Default is None

        Returns:
        --------
        V: `firedrake function space`
            Function space for the material property
        '''

        # Checking input arguments
        opts_func_space_type = ['scalar', 'vector', 'tensor']
        if func_space_type not in opts_func_space_type:
            error_management.value_parameter_error('func_space_type',
                                                   func_space_type,
                                                   opts_func_space_type)

        # Define the function space parameters
        if self.mesh_parameters.quadrilateral:  # Q_Elements
            base_mesh = self.mesh._base_mesh
            base_cell = base_mesh.ufl_cell()
            element = self.function_space.ufl_element()
            ele_zx = element.sub_elements[0].sub_elements[0]
            ele_y = element.sub_elements[1].sub_elements[1]
            typ_ele_zx = 'DQ' if dg_property else ele_zx.family()
            typ_ele_y = 'DG' if dg_property else ele_y.family()
            dgr_ele = (0, 0) if dg_property else element.degree()
            variant = element.variant()
            element_zx = fire.FiniteElement(typ_ele_zx, base_cell,
                                            dgr_ele[0], variant=variant)
            element_y = fire.FiniteElement(typ_ele_y, fire.interval,
                                           dgr_ele[1], variant=variant)
            tensor_element = fire.TensorProductElement(element_zx, element_y)
            V = fire.FunctionSpace(self.mesh, tensor_element)

            # Function space for the property
            if func_space_type == 'scalar':
                V = fire.FunctionSpace(self.mesh, tensor_element)
            elif func_space_type == 'vector':
                V = fire.VectorFunctionSpace(self.mesh, tensor_element)
            elif func_space_type == 'tensor':
                V = fire.TensorFunctionSpace(self.mesh, tensor_element,
                                             shape=shape_func_space)

        else:  # T_Elements
            typ_ele = 'DG' if dg_property else \
                self.function_space.ufl_element().family()
            dgr_ele = 0 if dg_property else \
                self.function_space.ufl_element().degree()
            V = fire.FunctionSpace(self.mesh, self.ele_type_c0, self.p_c0)

            # Function space for the property
            if func_space_type == 'scalar':
                V = fire.FunctionSpace(self.mesh, typ_ele, dgr_ele)
            elif func_space_type == 'vector':
                V = fire.VectorFunctionSpace(self.mesh, typ_ele, dgr_ele)
            elif func_space_type == 'tensor':
                V = fire.TensorFunctionSpace(self.mesh, typ_ele, dgr_ele,
                                             shape=shape_func_space)

        return V

    def _initialize_material_prop_from_ufl(self, property_name,
                                           func_space_type, V,
                                           shape_func_space=None,
                                           constant=None,
                                           conditional=None,
                                           expression=None):
        '''
        Initialize material property from a UFL input. This method is
        used when the material property is defined by a constant value,
        a conditional or an expression.

        Parameters:
        -----------
        property_name: `str`
            Name of the material property to be set
        func_space_type, `str`
            Type of function space for the material property.
            Options: 'scalar', 'vector' or 'tensor'
        V: `firedrake function space`
            Function space for the material property
        shape_func_space: `tuple`, optional
            Shape of the function space for only tensorial material property.
            Default is None
        constant: `float`, optional
            Constant value for the material property. Default is None
        conditional:  `firedrake conditional`, optional
            Firedrake conditional object. Default is None
        expression: `str`, optional
            If you use an expression, you can use the following variables:
            x, y, z, pi, tanh, sqrt. Ex: "2. + 0.5 * tanh((x - 2.) / 0.1)".
            It will be interpoalte into either the same function space as
            the object or a DG0 function space in the same mesh.
            Default is None

        Returns:
        --------
        mat_property: `firedrake function`
            Material property
        '''

        if constant is not None:
            value = 1 if constant == 0. else abs(constant)
            col = int(abs(log10(abs(value)))) + 2

            print(f"Assigning {property_name} with a "
                  f"constant value of {constant:>{col}}", flush=True)

            if func_space_type == 'vector':
                ufl_input = fire.as_vector((constant,) * self.dimension)

            elif func_space_type == 'tensor':
                ufl_input = fire.as_tensor(constant * ones((shape_func_space)))
            else:
                ufl_input = fire.Constant(constant)

        if conditional is not None:
            print(f"Assigning {property_name} with a conditional "
                  f"field given by {conditional}", flush=True)
            ufl_input = conditional

        if expression is not None:
            print(f"Assigning {property_name} with an expression "
                  f"field given by f = {expression} ", flush=True)
            ufl_input = eval_functions_to_ufl.generate_ufl_functions(
                self.mesh, expression, self.dimension)

        mat_property = fire.Function(
            V, name=property_name).interpolate(ufl_input)

        return mat_property

    def _initialize_material_prop_from_func(self, property_name,
                                            fire_function, V):
        '''
        Initialize material property from a firedrake function.

        Parameters:
        -----------
        property_name: `str`
            Name of the material property to be set.
        fire_function: `firedrake function`
            Firedrake function based on the input object
        V: `firedrake function space`
            Function space for the material property

        Returns:
        --------
        mat_property: `firedrake function`
            Material property
        '''

        ele_orig = self.function_space.ufl_element().family()
        dgr_orig = self.function_space.ufl_element().degree()
        typ_ele = V.ufl_element().family()
        dgr_ele = V.ufl_element().degree()

        print(f"Assigning {property_name} with a firedrake function",
              ("in the same" if typ_ele == ele_orig
               and dgr_ele == dgr_orig else "in another"),
              f"function space: {typ_ele} {dgr_ele}.", flush=True)

        if typ_ele == ele_orig and dgr_ele == dgr_orig:
            # Same function space
            mat_property = fire_function
            mat_property.rename(property_name)

        else:  # Different function space
            mat_property = fire.Function(
                V, name=property_name).interpolate(fire_function)

        return mat_property

    def _initialize_random_material_prop(self, property_name, random, V):
        '''
        Initialize material property from a random distribution.

        Parameters:
        -----------
        property_name: `str`
            Name of the material property to be set.
        random: `tuple`
            If you want to set a random material property, specify the range of
            values as a tuple (min, max)
        V: `firedrake function space`
            Function space for the material property

        Returns:
        --------
        mat_property: `firedrake function`
            Material property
        '''

        col0 = int(abs(log10(abs(random[0])))) + 2
        col1 = int(abs(log10(abs(random[1])))) + 2
        print(f"Assigning {property_name} with a random field "
              f"between ({random[0]:>{col0}},{random[1]:>{col1}})",
              flush=True)

        mat_property = fire.Function(V, name=property_name)
        mat_property.dat.data[:] = uniform(random[0], random[1],
                                           mat_property.dat.data.shape)

        return mat_property

    def _initialize_material_prop_from_file(self, property_name, from_file, V):
        '''
        Initialize material property from a file.

        Parameters:
        -----------
        property_name: `str`
            Name of the material property to be set.
        from_file: `str`
            Name of the file containing the material property
        V: `firedrake function space`
            Function space for the material property

        Returns:
        --------
        mat_property: `firedrake function`
            Material property
        '''

        ele_orig = self.function_space.ufl_element().family()
        dgr_orig = self.function_space.ufl_element().degree()
        typ_ele = V.ufl_element().family()
        dgr_ele = V.ufl_element().degree()

        print(f"Assigning {property_name} from file {from_file}",
              ("in the same" if typ_ele == ele_orig
               and dgr_ele == dgr_orig else "in another"),
              f"function space: {typ_ele} {dgr_ele}.", flush=True)

        if from_file.endswith(".segy"):
            if not SEISMIC_MESH_AVAILABLE:
                raise ImportError(
                    "SeismicMesh is required to convert segy files.")

            mp_filename, mp_filetype = splitext(from_file)
            warnings.warn("Converting segy file to hdf5")
            # ToDo: Change method name
            write_velocity_model(from_file, ofname=mp_filename)
            from_file = mp_filename + ".hdf5"

        if from_file.endswith((".hdf5", ".h5")):
            mat_property = interpolate(self, from_file, V)

        return mat_property

    def _saving_property_to_file(self, mat_property, property_name,
                                 foldername='default'):
        '''
        Save a material property to a pvd file for visualization.

        Parameters:
        -----------
        mat_property: `firedrake function`
            Material property
        property_name: `str`
            Name of the material property to be set.
        foldername : `string`, optional
            Name of the folder where the material property is saved.
            If default is 'default', property is saved in '/property_fields/'

        Returns:
        --------
        None
        '''

        # Path to save data
        self.path_save_matprop = getcwd() + ('/property_fields/'
                                             if foldername == 'default'
                                             else foldername)
        pth_prop = self.path_save_matprop + property_name + ".pvd"
        print(f"Saving {property_name} to {foldername} for visualization.")
        fire.VTKFile(pth_prop).write(mat_property, name=property_name)

    @staticmethod
    def _check_material_property_inputs(val_lst, func_space_type,
                                        shape_func_space, output):
        '''
        Check the inputs for setting a material property.

        Parameters:
        -----------
        val_lst: `list`
            List of values for constant, conditional, expression,
            random, fire_function and from_file.
        func_space_type, `str`
            Type of function space for the material property.
            Options: 'scalar', 'vector' or 'tensor'
        shape_func_space: `tuple`, optional
            Shape of the function space for only tensorial material
            property. Default is None
        output: `bool`, optional
            If True, outputs the material property to a pvd file for
            visualization. Default is False

        Returns:
        --------
        None
        '''

        if sum(value is not None for value in val_lst) > 1:
            name_lst = ["constant", "conditional", "expression",
                        "random", "fire_function", "from_file"]
            name_lst[-1] += " (*.segy or *.hdf5)"
            error_management.mutually_exclusive_parameter_error(name_lst,
                                                                val_lst)
        if shape_func_space is not None:
            if func_space_type != 'tensor':
                raise ValueError("'shape_func_space' can only be specified "
                                 "for tensorial material properties.")
            if shape_func_space[0] * shape_func_space[1] > 9 and output:
                raise ValueError("Output of tensorial material "
                                 "properties with more than 9 "
                                 "components is not supported.")

    def set_material_property(self, property_name, func_space_type,
                              shape_func_space=None, constant=None,
                              conditional=None, expression=None,
                              random=None, fire_function=None,
                              from_file=None, dg_property=False,
                              output=False, foldername='default'):
        '''
        Set a material property(e.g., density, etc.) in the model.

        Parameters:
        -----------
        property_name: `str`
            Name of the material property to be set.
        func_space_type, `str`
            Type of function space for the material property.
            Options: 'scalar', 'vector' or 'tensor'
        shape_func_space: `tuple`, optional
            Shape of the function space for only tensorial material property.
            Default is None
        from_file: `str`, optional
            Name of the file containing the material property. Default is None
        constant: `float`, optional
            Constant value for the material property. Default is None
        conditional: `firedrake conditional`, optional
            Firedrake conditional object. Default is None
        expression: `str`, optional
            If you use an expression, you can use the following variables:
            x, y, z, pi, tanh, sqrt. Ex: "2. + 0.5 * tanh((x - 2.) / 0.1)".
            It will be interpoalte into either the same function space as
            the object or a DG0 function space in the same mesh.
            Default is None
        random: `tuple`, optional
            If you want to set a random material property, specify the range of
            values as a tuple(min, max). Default is None
        fire_function: `firedrake function`, optional
            Firedrake function based on the input object. Default is None.
        dg_property: `bool`, optional
            If True, uses a DG0 function space for conditional and
            expression inputs. Default is False
        output: `bool`, optional
            If True, outputs the material property to a pvd file for
            visualization. Default is False
        foldername: `string`, optional
            Name of the folder where the material property is saved.
            If default is 'default', property is saved in '/property_fields/'

        Returns:
        -----------
        mat_property: `firedrake function`
            Material property
        '''

        # Checking input arguments
        val_lst = [constant, conditional, expression,
                   random, fire_function, from_file]

        self._check_material_property_inputs(val_lst, func_space_type,
                                             shape_func_space, output)

        V = self.define_property_function_space(
            func_space_type, dg_property, shape_func_space=shape_func_space)

        # If no mesh is set, we have to do it beforehand
        if self.mesh is None:
            self.set_mesh()

        try:

            if func_space_type == 'scalar':

                if any(v is not None for v in val_lst[:3]):  # UFL
                    mat_property = self._initialize_material_prop_from_ufl(
                        property_name, func_space_type, V, constant=constant,
                        conditional=conditional, expression=expression)

                if random is not None:  # Random
                    mat_property = self._initialize_random_material_prop(
                        property_name, random, V)

                if fire_function is not None:
                    mat_property = self._initialize_material_prop_from_func(
                        property_name, fire_function, V)

                if from_file is not None:
                    raise NotImplementedError("Initializing property "
                                              "from file is currently "
                                              "not implemented")
                    # mat_property = self._initialize_material_prop_from_file(
                    #     property_name, from_file, V)

            else:

                print("Vectorial and Tensorial material properties are "
                      "defined only either by constants or\nby firedrake "
                      "functions. If use 'constant', define its components "
                      "as scalar material\nproperties using the same property "
                      "name followed by the component at the end.")

                if constant is not None:
                    mat_property = self._initialize_material_prop_from_ufl(
                        property_name, func_space_type, V,
                        shape_func_space=shape_func_space,
                        constant=constant, conditional=None,
                        expression=None)

                if fire_function is not None:
                    mat_property = self._initialize_material_prop_from_func(
                        property_name, fire_function, V)

        except Exception as e:
            raise ValueError(f"Error Setting a Material Property: {e}.")

        if output:
            self._saving_property_to_file(mat_property, property_name,
                                          foldername=foldername)

        return mat_property
