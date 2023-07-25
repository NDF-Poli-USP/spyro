import warnings
import spyro

# default_optimization_parameters = {
#     "General": {"Secant": {"Type": "Limited-Memory BFGS",
# "Maximum Storage": 10}},
#     "Step": {
#         "Type": "Augmented Lagrangian",
#         "Augmented Lagrangian": {
#             "Subproblem Step Type": "Line Search",
#             "Subproblem Iteration Limit": 5.0,
#         },
#         "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
#     },
#     "Status Test": {
#         "Gradient Tolerance": 1e-16,
#         "Iteration Limit": None,
#         "Step Tolerance": 1.0e-16,
#     },
# }

# default_dictionary = {}
# default_dictionary["options"] = {
#     "cell_type": "T",  # simplexes such as triangles or tetrahedra (T)
# or quadrilaterals (Q)
#     "variant": 'lumped', # lumped, equispaced or DG, default is lumped
#     "method": "MLT", # (MLT/spectral_quadrilateral/DG_triangle/
# DG_quadrilateral) You can either specify a cell_type+variant or a method
#     "degree": 4,  # p order
#     "dimension": 2,  # dimension
#     "automatic_adjoint": False,
# }

# # Number of cores for the shot. For simplicity, we keep things serial.
# # spyro however supports both spatial parallelism and "shot" parallelism.
# default_dictionary["parallelism"] = {
# # options: automatic (same number of cores for evey processor) or spatial
#     "type": "automatic",
# }

# # Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# # domain and reserve the remaining 250 m for the Perfectly Matched Layer
# # (PML) to absorb
# # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
# default_dictionary["mesh"] = {
#     "Lz": 1.0,  # depth in km - always positive
#     "Lx": 1.0,  # width in km - always positive
#     "Ly": 0.0,  # thickness in km - always positive
#     "mesh_file": None,
# }
# #For use only if you are using a synthetic test model
# #or a forward only simulation -adicionar discrição para modelo direto
# default_dictionary["synthetic_data"] = {
#     "real_mesh_file": None,
#     "real_velocity_file": None,
# }
# default_dictionary["inversion"] = {
#     "perform_fwi": False, # switch to true to make a FWI
#     "initial_guess_model_file": None,
#     "shot_record_file": None,
#     "optimization_parameters": default_optimization_parameters,
# }

# # Specify a 250-m PML on the three sides of the
# # domain to damp outgoing waves.
# default_dictionary["absorving_boundary_conditions"] = {
#     "status": False,  # True or false
# #  None or non-reflective (outer boundary condition)
#     "outer_bc": "non-reflective",
# # polynomial, hyperbolic, shifted_hyperbolic
#     "damping_type": "polynomial",
#     "exponent": 2,  # damping layer has a exponent variation
#     "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
#     "R": 1e-6,  # theoretical reflection coefficient
# # thickness of the PML in the z-direction (km) - always positive
#     "lz": 0.25,
# # thickness of the PML in the x-direction (km) - always positive
#     "lx": 0.25,
# # thickness of the PML in the y-direction (km) - always positive
#     "ly": 0.0,
# }

# # Create a source injection operator. Here we use a single source with a
# # Ricker wavelet that has a peak frequency of 8 Hz injected at the
# # center of the mesh.
# # We also specify to record the solution at 101 microphones near the
# # top of the domain.
# # This transect of receivers is created with the helper function
# # `create_transect`.
# default_dictionary["acquisition"] = {
#     "source_type": "ricker",
#     "source_locations": [(-0.1, 0.5)],
#     "frequency": 5.0,
#     "delay": 1.0,
#     "receiver_locations": spyro.create_transect(
#         (-0.10, 0.1), (-0.10, 0.9), 20
#     ),
# }

# # Simulate for 2.0 seconds.
# default_dictionary["time_axis"] = {
#     "initial_time": 0.0,  #  Initial time for event
#     "final_time": 2.00,  # Final time for event
#     "dt": 0.001,  # timestep size
#     "amplitude": 1,  # the Ricker has an amplitude of 1.
# # how frequently to output solution to pvds
#     "output_frequency": 100,
# # how frequently to save solution to RAM
#     "gradient_sampling_frequency": 100,
# }
# default_dictionary["visualization"] = {
#     "forward_output" : True,
#     "output_filename": "results/forward_output.pvd",
#     "fwi_velocity_model_output": False,
#     "velocity_model_filename": None,
#     "gradient_output": False,
#     "gradient_filename": None,
#     "adjoint_output": False,
#     "adjoint_filename": None,
# }


class Model_parameters:
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
            dictionary = spyro.io.Dictionary_conversion(dictionary).new_dictionary
        # Saves inout_dictionary internally
        self.input_dictionary = dictionary

        # Sanitizes method or cell_type+variant inputs
        self._sanitize_method()

        # Checks time inputs
        self._sanitize_time_inputs()

        # Checks inversion variables, FWI and velocity model inputs and outputs
        self._sanitize_optimization_and_velocity()

        # Checking mesh_parameters
        self._sanitize_mesh()

        # Checking absorving boundary condition parameters
        self._sanitize_absorving_boundary_condition()

        # Checking source and receiver inputs
        self._sanitize_acquisition()

        # Setting up MPI communicator and checking parallelism:
        self._sanitize_comm(comm)

        # Check automatic adjoint
        self._sanitize_automatic_adjoint()

        # Sanitize output files
        self._sanitize_output()

    # default_dictionary["absorving_boundary_conditions"] = {
    #     "status": False,  # True or false
    # #  None or non-reflective (outer boundary condition)
    #     "outer_bc": "non-reflective",
    # # polynomial, hyperbolic, shifted_hyperbolic
    #     "damping_type": "polynomial",
    #     "exponent": 2,  # damping layer has a exponent variation
    #     "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    #     "R": 1e-6,  # theoretical reflection coefficient
    # # thickness of the PML in the z-direction (km) - always positive
    #     "lz": 0.25,
    # # thickness of the PML in the x-direction (km) - always positive
    #     "lx": 0.25,
    # # thickness of the PML in the y-direction (km) - always positive
    #     "ly": 0.0,
    # }
    def _sanitize_absorving_boundary_condition(self):
        if "absorving_boundary_conditions" in self.input_dictionary:
            dictionary = self.input_dictionary["absorving_boundary_conditions"]
        else:
            dictionary = {"status": False}
        self.abc_status = dictionary["status"]

        if "outer_bc" in dictionary:
            self.abc_outer_bc = dictionary["outer_bc"]
        else:
            self.abc_outer_bc = None

        if self.abc_status:
            self.abc_damping_type = dictionary["damping_type"]
            self.abc_exponent = dictionary["exponent"]
            self.abc_cmax = dictionary["cmax"]
            self.abc_R = dictionary["R"]
            self.abc_lz = dictionary["lz"]
            self.abc_lx = dictionary["lx"]
            self.abc_ly = dictionary["ly"]
        else:
            self.abc_damping_type = None
            self.abc_exponent = None
            self.abc_cmax = None
            self.abc_R = None
            self.abc_lz = 0.0
            self.abc_lx = 0.0
            self.abc_ly = 0.0

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

        # Estabilishing forward output file and setting a default
        if "forward_output_filename" not in dictionary:
            self.forward_output_file = "results/forward_propogation.pvd"
        elif dictionary["forward_output_filename"] is not None:
            self.forward_output_file = dictionary["forward_output_filename"]
        else:
            self.forward_output_file = "results/forward_propagation.pvd"

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

    def get_wavelet(self):
        """Returns a wavelet based on the source type.

        Returns
        -------
        wavelet : numpy.ndarray
            Wavelet values in each time step to be used in the simulation.
        """
        if self.source_type == "ricker":
            if "delay_type" in self.input_dictionary["acquisition"]:
                delay_type = self.input_dictionary["acquisition"]["delay_type"]
            else:
                delay_type = "multiples_of_minimun"
            wavelet = spyro.full_ricker_wavelet(
                dt=self.dt,
                final_time=self.final_time,
                frequency=self.frequency,
                delay=self.delay,
                amplitude=self.amplitude,
                delay_type=delay_type,
            )
        elif self.source_type == "mms_source":
            wavelet = None
        else:
            raise ValueError(
                f"Source type of {self.source_type} not yet implemented."
            )

        return wavelet

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

        if comm is None:
            self.comm = spyro.utils.mpi_init(self)
            self.comm.comm.barrier()
        else:
            self.comm = comm

    def _sanitize_acquisition(self):
        dictionary = self.input_dictionary["acquisition"]
        self.number_of_sources = len(dictionary["source_locations"])
        self.source_locations = dictionary["source_locations"]
        self.number_of_receivers = len(dictionary["receiver_locations"])
        self.receiver_locations = dictionary["receiver_locations"]
        self.frequency = dictionary["frequency"]
        if "amplitude" in dictionary:
            self.amplitude = dictionary["amplitude"]
        else:
            self.amplitude = 1.0
        if "delay" in dictionary:
            self.delay = dictionary["delay"]
        else:
            self.delay = 1.5
        self.__check_acquisition()

        # Check ricker source:
        self.source_type = dictionary["source_type"]
        if self.source_type == "Ricker":
            self.source_type = "ricker"

    def _sanitize_mesh(self):
        dictionary = self.input_dictionary
        self.mesh_file = dictionary["mesh"]["mesh_file"]
        if "mesh_type" in dictionary["mesh"]:
            self.mesh_type = dictionary["mesh"]["mesh_type"]
        else:
            self.mesh_type = None

        if "user_mesh" in dictionary["mesh"]:
            if dictionary["mesh"]["user_mesh"]:
                self.user_mesh = dictionary["mesh"]["user_mesh"]
            else:
                self.user_mesh = False
        else:
            self.user_mesh = False

        if (
            "firedrake_mesh" in dictionary["mesh"]
            or self.mesh_type == "firedrake_mesh"
        ):
            self.firedrake_mesh = True
            self.mesh_type = "firedrake_mesh"
        else:
            self.firedrake_mesh = False

        if self.mesh_file == "not_used.msh":
            self.mesh_file = None
        self.length_z = dictionary["mesh"]["Lz"]
        self.length_x = dictionary["mesh"]["Lx"]
        self.length_y = dictionary["mesh"]["Ly"]
        if "user_mesh" in dictionary["mesh"] and self.mesh_file is None:
            self.mesh_type = "user_mesh"
        if self.user_mesh is None and self.mesh_file is None:
            warnings.warn("No mesh yet provided.")

        if self.mesh_file is not None:
            self.mesh_type = "file"

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
        if "inversion" in dictionary:
            if dictionary["inversion"]["perform_fwi"]:
                self.running_fwi = True
        if self.running_fwi:
            self.initial_velocity_model_file = dictionary["inversion"][
                "initial_guess_model_file"
            ]
            self.fwi_output_folder = "fwi/"
            self.control_output_file = self.fwi_output_folder + "control"
            self.gradient_output_file = self.fwi_output_folder + "gradient"
            self.optimization_parameters = dictionary["inversion"][
                "optimization_parameters"
            ]
        else:
            if "synthetic_data" in dictionary:
                self.initial_velocity_model_file = dictionary[
                    "synthetic_data"
                ]["real_velocity_file"]
            else:
                dictionary["synthetic_data"] = {"real_velocity_file": None}
                self.initial_velocity_model_file = None

        if self.initial_velocity_model_file is None:
            if "velocity_conditional" not in dictionary["synthetic_data"]:
                self.velocity_model_type = None
                warnings.warn(
                    "No velocity model set initially. If using \
                        user defined conditional or expression, please \
                            input it in the Wave object."
                )

        if "velocity_conditional" in dictionary["synthetic_data"]:
            self.velocity_model_type = "conditional"
            self.velocity_conditional = dictionary["synthetic_data"][
                "velocity_conditional"
            ]

        self.forward_output_file = "results/forward_output.pvd"

    def _sanitize_time_inputs(self):
        dictionary = self.input_dictionary["time_axis"]
        self.final_time = dictionary["final_time"]
        self.dt = dictionary["dt"]
        if "initial_time" in dictionary:
            self.initial_time = dictionary["initial_time"]
        else:
            self.initial_time = 0.0
        self.output_frequency = dictionary["output_frequency"]
        self.gradient_sampling_frequency = dictionary[
            "gradient_sampling_frequency"
        ]

        self.__check_time()

    def _sanitize_method(self):
        dictionary = self.input_dictionary
        self.cell_type = None
        self.method = None
        self.variant = None
        self.__get_method()

        # Checks if degree is valid
        self.degree = dictionary["options"]["degree"]
        self.dimension = dictionary["options"]["dimension"]
        self.__check_degree()

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
        if self.dt > 1.0:
            warnings.warn(f"Time step of {self.dt} too big.")
        if self.dt is None:
            warnings.warn(
                "Timestep not given. Will calculate internally when user \
                    attemps to propagate wave."
            )

    def __check_degree(self):
        if self.method == "mass_lumped_triangle":
            if self.dimension == 2 and self.degree > 5:
                raise ValueError(
                    f"Degree of {self.degree} not supported by \
                        {self.dimension}D {self.method}."
                )
            if self.dimension == 3 and self.degree > 4:
                raise ValueError(
                    f"Degree of {self.degree} not supported by \
                        {self.dimension}D {self.method}."
                )
            if self.dimension == 3 and self.degree == 4:
                warnings.warn(
                    f"Degree of {self.degree} not supported by \
                        {self.dimension}D {self.method} in main firedrake."
                )

    def __unify_method_input(self):
        unified_method = None
        method = self.method
        if (
            method == "KMV"
            or method == "MLT"
            or method == "mass_lumped_triangle"
            or method == "mass_lumped_tetrahedra"
        ):
            unified_method = "mass_lumped_triangle"
        elif (
            method == "spectral"
            or method == "SEM"
            or method == "spectral_quadrilateral"
        ):
            unified_method = "spectral_quadrilateral"
        elif method == "DG_triangle":
            unified_method = method
        elif method == "DG_quadrilateral":
            unified_method = method
        elif method == "CG":
            unified_method = method
        else:
            warnings.warn(f"Method of {method} not accepted.")
        self.method = unified_method

    def __unify_cell_type_input(self):
        unified_cell_type = None
        cell_type = self.cell_type
        if (
            cell_type == "T"
            or cell_type == "triangles"
            or cell_type == "triangle"
            or cell_type == "tetrahedron"
            or cell_type == "tetrahedra"
        ):
            unified_cell_type = "triangle"
        elif (
            cell_type == "Q"
            or cell_type == "quadrilateral"
            or cell_type == "quadrilaterals"
            or cell_type == "hexahedron"
            or cell_type == "hexahedra"
        ):
            unified_cell_type = "quadrilateral"
        elif cell_type is None:
            unified_cell_type = None
        else:
            warnings.warn(f"Cell type of {cell_type} not accepted.")
        self.cell_type = unified_cell_type

    def __unify_variant_input(self):
        unified_variant = None
        variant = self.variant

        if (
            variant == "spectral"
            or variant == "GLL"
            or variant == "SEM"
            or variant == "lumped"
            or variant == "KMV"
        ):
            unified_variant = "lumped"
        elif variant == "equispaced" or variant == "equis":
            unified_variant = "equispaced"
        elif variant == "DG" or variant == "discontinuous_galerkin":
            unified_variant = "DG"
        else:
            warnings.warn(f"Variant of {variant} not accepted.")
        self.method = unified_variant

    def __get_method_from_cell_type(self):
        cell_type = self.cell_type
        variant = self.variant
        if cell_type == "triangle":
            if variant == "lumped":
                method = "mass_lumped_triangle"
            elif variant == "equispaced":
                method = "CG_triangle"
            elif variant == "DG":
                method = "DG_triangle"
        elif cell_type == "quadrilateral":
            if variant == "lumped":
                method = "spectral_quadrilateral"
            elif variant == "equispaced":
                method = "CG_quadrilateral"
            elif variant == "DG":
                method = "DG_quadrilateral"
        self.method = method

    def __get_method(self):
        dictionary = self.input_dictionary
        # Checking if method/cell_type + variant specified twice:
        if (
            "method" in dictionary["options"]
            and ("cell_type" in dictionary["options"])
            and ("variant" in dictionary["options"])
        ):
            if (
                dictionary["options"]["method"] is not None
                and dictionary["options"]["cell_type"] is not None
            ):
                self.cell_type = dictionary["options"]["cell_type"]
                self.__unify_cell_type_input()
                warnings.warn(
                    "Both methods of specifying method and cell_type with \
                        variant used. Method specification taking priority."
                )
        if (
            "method" in dictionary["options"]
            and dictionary["options"]["method"] is not None
        ):
            self.method = dictionary["options"]["method"]
            self.__unify_method_input()
            # For backwards compatibility
            if "variant" in dictionary["options"]:
                if (
                    dictionary["options"]["variant"] == "spectral"
                    or dictionary["options"]["variant"] == "GLL"
                    and self.method == "CG"
                ):
                    self.method = "spectral_quadrilateral"

        elif (
            ("cell_type" in dictionary["options"])
            and ("variant" in dictionary["options"])
            and dictionary["options"]["cell_type"] is not None
        ):
            self.cell_type = dictionary["options"]["cell_type"]
            self.__unify_cell_type_input()
            self.variant = dictionary["options"]["variant"]
            self.__unify_variant_input()
            self.__get_method_from_cell_type()
        else:
            raise ValueError("Missing options inputs.")

    def set_mesh(
        self,
        dx=None,
        user_mesh=None,
        mesh_file=None,
        length_z=None,
        length_x=None,
        length_y=None,
        periodic=False,
    ):
        """

        Parameters
        ----------
        dx : float, optional
            The desired mesh spacing. The default is None.
        user_mesh : spyro.Mesh, optional
            The desired mesh. The default is None.
        mesh_file : str, optional
            The path to the desired mesh file. The default is None.
        length_z : float, optional
            The length of the domain in the z-direction. The default is None.
        length_x : float, optional
            The length of the domain in the x-direction. The default is None.
        length_y : float, optional
            The length of the domain in the y-direction. The default is None.
        periodic : bool, optional
            Whether the domain is periodic. The default is False.
        """

        if length_z is not None:
            self.length_z = length_z
        if length_x is not None:
            self.length_x = length_x
        if length_y is not None:
            self.length_y = length_y

        if user_mesh is not None:
            self.user_mesh = user_mesh
            self.mesh_type = "user_mesh"
        elif mesh_file is not None:
            self.mesh_file = mesh_file
            self.mesh_type = "file"
        elif self.mesh_type == "firedrake_mesh":
            AutoMeshing = spyro.meshing.AutomaticMesh(dimension=self.dimension, comm=self.comm)
        
        if periodic and self.mesh_type == "firedrake_mesh":
            AutoMeshing.make_periodic()
        elif periodic and self.mesh_type != "firedrake_mesh":
            raise ValueError("Periodic meshes only supported for firedrake meshes.")

        if (
            dx is not None
            and self.mesh_type == "firedrake_mesh"
        ):
            AutoMeshing.set_mesh_size(
                length_z=self.length_z,
                length_x=self.length_x,
                length_y=self.length_y,
            )
            AutoMeshing.set_meshing_parameters(
                dx=dx,
                cell_type=self.cell_type,
                mesh_type=self.mesh_type
            )
            self.user_mesh = AutoMeshing.create_mesh()

        if (
            length_z is None
            or length_x is None
            or (length_y is None and self.dimension == 2)
        ) and self.mesh_type != "firedrake_mesh":
            warnings.warn(
                "Mesh dimensions not completely reset from initial dictionary"
            )

    def get_mesh(self):
        """Reads in an external mesh and scatters it between cores.

        Returns
        -------
        mesh: Firedrake.Mesh object
            The distributed mesh across `ens_comm`
        """
        if self.mesh_file is not None:
            return spyro.io.read_mesh(self)
        elif (
            self.mesh_type == "user_mesh" or self.mesh_type == "firedrake_mesh"
        ):
            return self.user_mesh
