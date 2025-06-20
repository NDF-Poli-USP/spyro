import numpy as np
import uuid
from mpi4py import MPI  # noqa:F401
from firedrake import COMM_WORLD  # noqa:
import warnings
from .. import io
from .. import utils
from .. import meshing


class Model_parameters:
    """
    Class that reads and sanitizes input parameters.

    Attributes
    ----------
    input_dictionary: dictionary
        Contains all input parameters already organized based on examples
        from github.
    cell_type: str
        Type of cell used in meshing. Can be "T" for triangles or "Q" for
        quadrilaterals.
    method: str
        Method used in meshing. Can be "MLT" for mass lumped triangles,
        "spectral_quadrilateral" for spectral quadrilaterals, "DG_triangle"
        for discontinuous Galerkin triangles, or "DG_quadrilateral" for
        discontinuous Galerkin quadrilaterals.
    variant: str
        Variant used in meshing. Can be "lumped" for lumped mass matrices,
        "equispaced" for equispaced nodes, or "DG" for discontinuous Galerkin
        nodes.
    degree: int
        Degree of the basis functions used in the FEM.
    dimension: int
        Dimension of the mesh.
    final_time: float
        Final time of the simulation.
    dt: float
        Time step of the simulation.
    initial_time: float
        Initial time of the simulation.
    output_frequency: int
        Frequency of outputting the solution to pvd files.
    gradient_sampling_frequency: int
        Frequency of saving the solution to RAM.
    number_of_sources: int
        Number of sources used in the simulation.
    source_locations: list
        List of source locations.
    frequency: float
        Frequency of the source.
    amplitude: float
        Amplitude of the source.
    delay: float
        Delay of the source.
    number_of_receivers: int
        Number of receivers used in the simulation.
    receiver_locations: list
        List of receiver locations.
    parallelism_type: str
        Type of parallelism used in the simulation. Can be "automatic" for
        automatic parallelism or "spatial" for spatial parallelism.
    mesh_file: str
        Path to the mesh file.
    length_z: float
        Length of the domain in the z-direction.
    length_x: float
        Length of the domain in the x-direction.
    length_y: float
        Length of the domain in the y-direction.
    user_mesh: spyro.Mesh
        User defined mesh.
    firedrake_mesh: firedrake.Mesh
        Firedrake mesh.
    abc_active: bool
        Whether or not the absorbing boundary conditions are used.
    abc_exponent: int
        Exponent of the absorbing boundary conditions.
    abc_cmax: float
        Maximum acoustic wave velocity in the absorbing boundary conditions.
    abc_R: float
        Theoretical reflection coefficient of the absorbing boundary
        conditions.
    abc_pad_length: float
        Thickness of the absorbing boundary conditions.
    source_type: str
        Type of source used in the simulation. Can be "ricker" for a Ricker
        wavelet or "MMS" for a manufactured solution.
    running_fwi: bool
        Whether or not the simulation is a FWI.
    initial_velocity_model_file: str
        Path to the initial velocity model file.
    fwi_output_folder: str
        Path to the FWI output folder.
    control_output_file: str
        Path to the control output file.
    gradient_output_file: str
        Path to the gradient output file.
    optimization_parameters: dict
        Dictionary of the optimization parameters.
    automatic_adjoint: bool
        Whether or not the adjoint is calculated automatically.
    forward_output: bool
        Whether or not the forward output is saved.
    fwi_velocity_model_output: bool
        Whether or not the FWI velocity model output is saved.
    gradient_output: bool
        Whether or not the gradient output is saved.
    adjoint_output: bool
        Whether or not the adjoint output is saved.
    forward_output_file: str
        Path to the forward output file.
    fwi_velocity_model_output_file: str
        Path to the FWI velocity model output file.
    gradient_output_file: str
        Path to the gradient output file.
    adjoint_output_file: str
        Path to the adjoint output file.
    comm: MPI communicator
        MPI communicator.
    velocity_model_type: str
        Type of velocity model used in the simulation. Can be "file" for a
        file, "conditional" for a conditional, or None for no velocity model.
    velocity_conditional: str
        Conditional used for the velocity model.
    equation_type: str
        Type of equation used in the simulation. Can be "second_order_in_pressure".
    time_integrator: str
        Type of time integrator used in the simulation. Can be "central_difference".

    Methods
    -------
    get_wavelet()
        Returns a wavelet based on the source type.
    set_mesh()
        Sets the mesh.
    get_mesh()
        Reads in a mesh and scatters it between cores.
    """

    def __init__(self, dictionary=None, comm=None):
        """Initializes class that reads and sanitizes input parameters.
        A dictionary can be used.

        Parameters
        ----------
        dictionary: 'dictionary' (optional)
            Contains all input parameters already organized based on examples
            from github.
        comm: MPI communicator (optional)
            MPI comunicator. If None is given model_parameters creates one.

        Returns
        -------
        model_parameters: :class: 'model_parameters' object
        """
        # Converts old dictionary to new one. Deprecated feature
        if "opts" in dictionary:
            warnings.warn("Old deprecated dictionary style in usage.")
            dictionary = io.Dictionary_conversion(dictionary).new_dictionary
        # Saves inout_dictionary internally
        self.input_dictionary = dictionary

        # some default parameters we might use in the future
        self.input_dictionary["time_axis"].setdefault("time_integration_scheme", "central_difference")
        self.input_dictionary.setdefault("equation_type", "second_order_in_pressure")
        self.time_integrator = self.input_dictionary["time_axis"]["time_integration_scheme"]
        self.equation_type = self.input_dictionary["equation_type"]

        # Sanitizes method or cell_type+variant inputs
        options = io.dictionaryio.Read_options(self.input_dictionary["options"])
        self.cell_type = options.cell_type
        self.method = options.method
        self.variant = options.variant
        self.degree = options.degree
        self.dimension = options.dimension
        self.sources = None

        if self.cell_type == "quadrilateral":
            quadrilateral = True
        else:
            quadrilateral = False

        # Checks time inputs
        self.input_dictionary["time_axis"].setdefault("initial_time", 0.0)
        self.initial_time = self.input_dictionary["time_axis"]["initial_time"]
        self.final_time = self.input_dictionary["time_axis"]["final_time"]
        self.dt = self.input_dictionary["time_axis"]["dt"]

        # Checks inversion variables, FWI and velocity model inputs and outputs
        self.real_shot_record = None
        self._sanitize_optimization_and_velocity()

        # Checking source and receiver inputs
        self._sanitize_acquisition()

        # Setting up MPI communicator and checking parallelism:
        self._sanitize_comm(comm)

        # Checking mesh_parameters
        self.input_dictionary["mesh"].setdefault("user_mesh", None)
        self.input_dictionary.setdefault("absorving_boundary_conditions", {})
        self.input_dictionary["absorving_boundary_conditions"].setdefault("pad_length", None)
        self.input_dictionary["absorving_boundary_conditions"].setdefault("status", False)
        self.user_mesh = self.input_dictionary["mesh"]["user_mesh"]
        self.mesh_parameters = meshing.MeshingParameters(
            input_mesh_dictionary=self.input_dictionary["mesh"],
            dimension=self.dimension,
            source_frequency=self.input_dictionary["acquisition"]["frequency"],
            comm=self.comm,
            quadrilateral=quadrilateral,
            method=self.method,
            degree=self.degree,
            abc_pad_length=self.input_dictionary["absorving_boundary_conditions"]["pad_length"],
        )

        # Checking absorving boundary condition parameters
        self._sanitize_absorving_boundary_condition()

        # Checking source and receiver inputs
        self._sanitize_acquisition()

        # Check automatic adjoint
        self.input_dictionary["time_axis"].setdefault("gradient_sampling_frequency", 99999)
        self.input_dictionary["time_axis"].setdefault("output_frequency", 99999)
        self.gradient_sampling_frequency = self.input_dictionary["time_axis"]["output_frequency"]
        self.output_frequency = self.input_dictionary["time_axis"]["output_frequency"]
        self._sanitize_automatic_adjoint()

        # Sanitize output files
        self._sanitize_output()
        self.random_id_string = str(uuid.uuid4())[:10]

    @property
    def source_locations(self):
        return self._source_locations
    
    @source_locations.setter
    def source_locations(self, value):
        if value is not None:
            self.number_of_sources = len(value)
        else:
            self.number_of_sources = 1
        self._source_locations = value

    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        if value is not None:
            if value < 1.0:
                warnings.warn("Frequency of {value} too low for realistic FWI.")
            elif value > 50:
                warnings.warn("Frequency of {value} too high for eficient FWI.")
        self._frequency = value

    @property
    def receiver_locations(self):
        return self._receiver_locations
    
    @receiver_locations.setter
    def receiver_locations(self, value):
        if value is not None:
            self.number_of_receivers = len(value)
        else:
            self.number_of_receivers = 1
        self._receiver_locations = value

    @property
    def initial_time(self):
        return self._initial_time
    
    @initial_time.setter
    def initial_time(self, value):
        if value is None:
            value = 0.0
        self._initial_time = value
    
    @property
    def final_time(self):
        return self._final_time
    
    @final_time.setter
    def final_time(self, value):
        if value < self.initial_time:
            raise ValueError(f"Final time of {value} lower than initial time of {self.initial_time} not allowed.")
        
        self._final_time = value

    @property
    def time_integrator(self):
        return self._time_integrator
    
    @time_integrator.setter
    def time_integrator(self, value):
        if value != "central_difference":
            raise ValueError(f"The time integrator of {value} is not implemented yet")
        self._time_integrator = value
    
    @property
    def equation_type(self):
        return self._equation_type
    
    @equation_type.setter
    def equation_type(self, value):
        if value != "second_order_in_pressure":
            raise ValueError(
                "The equation type specified is not implemented yet"
            )
        self._equation_type = value

    def _sanitize_absorving_boundary_condition(self):
        if "absorving_boundary_conditions" not in self.input_dictionary:
            self.input_dictionary["absorving_boundary_conditions"] = {
                "status": False
            }
        dictionary = self.input_dictionary["absorving_boundary_conditions"]
        self.abc_active = dictionary["status"]

        BL_obj = io.boundary_layer_io.read_boundary_layer(dictionary)
        self.abc_exponent = BL_obj.abc_exponent
        self.abc_cmax = BL_obj.abc_cmax
        self.abc_R = BL_obj.abc_R
        self.abc_pad_length = BL_obj.abc_pad_length
        self.abc_boundary_layer_type = BL_obj.abc_boundary_layer_type

        self.absorb_top = dictionary.get("absorb_top", False)
        self.absorb_bottom = dictionary.get("absorb_bottom", True)
        self.absorb_right = dictionary.get("absorb_right", True)
        self.absorb_left = dictionary.get("absorb_left", True)
        self.absorb_front = dictionary.get("absorb_front", True)
        self.absorb_back = dictionary.get("absorb_back", True)

    def _sanitize_output(self):
        #         default_dictionary["visualization"] = {
        #     "forward_output" : True,
        #     "forward_output_filename": "results/forward.pvd",
        #     "fwi_velocity_model_output": False,
        #     "velocity_model_filename": None,
        #     "gradient_output": False,
        #     "gradient_filename": None,
        #     "adjoint_output": False,
        #     "adjoint_filename": None,
        # }
        # Checking if any output should be saved
        if "visualization" in self.input_dictionary:
            dictionary = self.input_dictionary["visualization"]
        else:
            dictionary = {
                "forward_output": False,
                "fwi_velocity_model_output": False,
                "gradient_output": False,
                "adjoint_output": False,
            }
            self.input_dictionary["visualization"] = dictionary

        self.forward_output = dictionary["forward_output"]

        if "fwi_velocity_model_output" in dictionary:
            self.fwi_velocity_model_output = dictionary[
                "fwi_velocity_model_output"
            ]
        else:
            self.fwi_velocity_model_output = False

        if "gradient_output" in dictionary:
            self.gradient_output = dictionary["gradient_output"]
        else:
            self.gradient_output = False

        if "adjoint_output" in dictionary:
            self.adjoint_output = dictionary["adjoint_output"]
        else:
            self.adjoint_output = False

        # Getting output file names
        self._sanitize_output_files()

    def _sanitize_output_files(self):
        self._sanitize_forward_output_files()
        dictionary = self.input_dictionary["visualization"]

        # Estabilishing velocity model file and setting a default
        if "velocity_model_filename" not in dictionary:
            self.fwi_velocity_model_output_file = (
                "results/fwi_velocity_model.pvd"
            )
        elif dictionary["velocity_model_filename"] is None:
            self.fwi_velocity_model_output_file = dictionary[
                "velocity_model_filename"
            ]
        else:
            self.fwi_velocity_model_output_file = (
                "results/fwi_velocity_model.pvd"
            )

        self._check_debug_output()

    def _sanitize_forward_output_files(self):
        dictionary = self.input_dictionary["visualization"]
        if "forward_output_filename" not in dictionary:
            self.forward_output_file = "results/forward_propogation.pvd"
        elif dictionary["forward_output_filename"] is not None:
            self.forward_output_file = dictionary["forward_output_filename"]
        else:
            self.forward_output_file = "results/forward_propagation.pvd"

    def _sanitize_adjoint_and_gradient_output_files(self):
        dictionary = self.input_dictionary["visualization"]
        # Estabilishing gradient file and setting a default
        if "gradient_filename" not in dictionary:
            self.gradient_output_file = "results/gradient.pvd"
        elif dictionary["gradient_filename"] is None:
            self.gradient_output_file = dictionary["gradient_filename"]
        else:
            self.gradient_output_file = "results/gradient.pvd"

        # Estabilishing adjoint file and setting a default
        if "adjoint_filename" not in dictionary:
            self.adjoint_output_file = "results/adjoint.pvd"
        elif dictionary["adjoint_filename"] is None:
            self.adjoint_output_file = dictionary["adjoint_filename"]
        else:
            self.adjoint_output_file = "results/adjoint.pvd"

    def _check_debug_output(self):
        dictionary = self.input_dictionary["visualization"]
        # Estabilishing debug output
        if "debug_output" not in dictionary:
            self.debug_output = False
        elif dictionary["debug_output"] is None:
            self.debug_output = False
        elif dictionary["debug_output"] is False:
            self.debug_output = False
        elif dictionary["debug_output"] is True:
            self.debug_output = True
        else:
            raise ValueError("Debug output not understood")

    def _sanitize_automatic_adjoint(self):
        dictionary = self.input_dictionary
        if "automatic_adjoint" in dictionary:
            self.automatic_adjoint = True
        else:
            self.automatic_adjoint = False

    def _sanitize_comm(self, comm):
        dictionary = self.input_dictionary
        if "parallelism" in dictionary:
            self.parallelism_type = dictionary["parallelism"]["type"]
        else:
            warnings.warn("No paralellism type listed. Assuming automatic")
            self.parallelism_type = "automatic"

        if self.parallelism_type == "custom":
            self.shot_ids_per_propagation = dictionary["parallelism"]["shot_ids_per_propagation"]
        elif self.parallelism_type == "automatic":
            self.shot_ids_per_propagation = [[i] for i in range(0, self.number_of_sources)]
        elif self.parallelism_type == "spatial":
            self.shot_ids_per_propagation = [[i] for i in range(0, self.number_of_sources)]

        if comm is None:
            self.comm = utils.mpi_init(self)
            self.comm.comm.barrier()
        else:
            self.comm = comm

    def _sanitize_acquisition(self):
        dictionary = self.input_dictionary["acquisition"]
        self.number_of_receivers = len(dictionary["receiver_locations"])
        self.receiver_locations = dictionary["receiver_locations"]

        # Check ricker source:
        self.source_type = dictionary["source_type"]
        if self.source_type == "Ricker":
            self.source_type = "ricker"
        elif self.source_type == "MMS":
            self.source_locations = []
            self.number_of_sources = 1
            self.frequency = None
            self.amplitude = None
            self.delay = None
            return

        # self.number_of_sources = len(dictionary["source_locations"])
        self.source_locations = dictionary["source_locations"]
        self.frequency = dictionary["frequency"]
        if "amplitude" in dictionary:
            self.amplitude = dictionary["amplitude"]
        else:
            self.amplitude = 1.0
        if "delay" in dictionary:
            self.delay = dictionary["delay"]
        else:
            self.delay = 1.5
        self.delay_type = dictionary.get("delay_type", "multiples_of_minimun")
        self.__check_acquisition()

    def _sanitize_optimization_and_velocity(self):
        """
        Checks if we are doing a FWI and sorts velocity model types, inputs,
        and outputs
        """
        dictionary = self.input_dictionary
        self.velocity_model_type = "file"

        # Check if we are doing a FWI and sorting output locations and
        # velocity model inputs
        self.running_fwi = False
        if "inversion" not in dictionary:
            dictionary["inversion"] = {"perform_fwi": False}

        if dictionary["inversion"]["perform_fwi"]:
            self.running_fwi = True

        if self.running_fwi:
            self._sanitize_optimization_and_velocity_for_fwi()
        else:
            self._sanitize_optimization_and_velocity_without_fwi()

        if self.initial_velocity_model_file is None:
            if "velocity_conditional" not in dictionary["synthetic_data"]:
                self.velocity_model_type = None
                warnings.warn(
                    "No velocity model set initially. If using "
                    "user defined conditional or expression, please "
                    "input it in the Wave object."
                )

        if "velocity_conditional" in dictionary["synthetic_data"]:
            self.velocity_model_type = "conditional"
            self.velocity_conditional = dictionary["synthetic_data"][
                "velocity_conditional"
            ]

        self.forward_output_file = "results/forward_output.pvd"

    def _sanitize_optimization_and_velocity_for_fwi(self):
        self._sanitize_optimization_and_velocity_without_fwi()
        dictionary = self.input_dictionary
        try:
            self.initial_velocity_model_file = dictionary["inversion"][
                "initial_guess_model_file"
            ]
        except KeyError:
            self.initial_velocity_model_file = None
        self.fwi_output_folder = "fwi/"
        self.control_output_file = self.fwi_output_folder + "control"
        self.gradient_output_file = self.fwi_output_folder + "gradient"
        if "optimization_parameters" in dictionary["inversion"]:
            self.optimization_parameters = dictionary["inversion"][
                "optimization_parameters"
            ]
        else:
            default_optimization_parameters = {
                "General": {"Secant": {"Type": "Limited-Memory BFGS",
                                       "Maximum Storage": 10}},
                "Step": {
                    "Type": "Augmented Lagrangian",
                    "Augmented Lagrangian": {
                        "Subproblem Step Type": "Line Search",
                        "Subproblem Iteration Limit": 5.0,
                    },
                    "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
                },
                "Status Test": {
                    "Gradient Tolerance": 1e-16,
                    "Iteration Limit": None,
                    "Step Tolerance": 1.0e-16,
                },
            }
            self.optimization_parameters = default_optimization_parameters

        if "shot_record_file" in dictionary["inversion"]:
            if dictionary["inversion"]["shot_record_file"] is not None:
                self.real_shot_record = np.load(dictionary["inversion"]["shot_record_file"])

    def _sanitize_optimization_and_velocity_without_fwi(self):
        dictionary = self.input_dictionary
        if "synthetic_data" in dictionary:
            self.initial_velocity_model_file = dictionary["synthetic_data"][
                "real_velocity_file"
            ]
        else:
            dictionary["synthetic_data"] = {"real_velocity_file": None}
            self.initial_velocity_model_file = None

    def _sanitize_time_inputs(self):
        self.__check_time()

    def __check_acquisition(self):
        for source in self.source_locations:
            if self.dimension == 2:
                source_z, source_x = source
                source_y = 0.0
            elif self.dimension == 3:
                source_z, source_x, source_y = source
            else:
                raise ValueError("Source input type not supported")

    def __check_time(self):
        if self.final_time < 0.0:
            raise ValueError(f"Negative time of {self.final_time} not valid.")
        if self._dt > 1.0:
            warnings.warn(f"Time step of {self.dt} too big.")
        if self._dt is None:
            warnings.warn(
                "Timestep not given. Will calculate internally when user \
                    attemps to propagate wave."
            )

    def set_mesh(
        self,
        user_mesh=None,
        input_mesh_parameters={},
    ):
        """
        Set the mesh for the model.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            The desired mesh. The default is None.
        mesh_parameters : dict, optional
            Additional parameters for setting up the mesh. The default is an empty dictionary.

        Returns
        -------
        None
        """
        if user_mesh is not None:
            self.user_mesh = user_mesh

        pad_length = None
        if self.abc_active:
            pad_length = self.abc_pad_length
        self.mesh_parameters.set_mesh(user_mesh=user_mesh,input_mesh_parameters=input_mesh_parameters, abc_pad_length=pad_length)

        if self.mesh_parameters.automatic_mesh:
            autoMeshing = meshing.AutomaticMesh(
                mesh_parameters=self.mesh_parameters,
            )

            self.user_mesh = autoMeshing.create_mesh()

    def _set_mesh_length(
        self,
        length_z=None,
        length_x=None,
        length_y=None,
    ):
        if length_z is not None:
            self.mesh_parameters.length_z = length_z
        if length_x is not None:
            self.mesh_parameters.length_x = length_x
        if length_y is not None:
            self.mesh_parameters.length_y = length_y

    def get_mesh(self):
        """Reads in an external mesh and scatters it between cores.

        Returns
        -------
        mesh: Firedrake.Mesh object
            The distributed mesh across `ens_comm`
        """
        if self.mesh_parameters.user_mesh is not None:
            self.user_mesh = self.mesh_parameters.user_mesh
            return self.user_mesh
        elif self.mesh_parameters.mesh_file is not None:
            return io.read_mesh(self.mesh_parameters)
        else:
            return self.user_mesh
