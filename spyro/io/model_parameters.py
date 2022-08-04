from genericpath import exists
import warnings
import spyro
import firedrake as fire
import os.path

default_optimization_parameters = {
    "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
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

default_dictionary = {}
default_dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped', # lumped, equispaced or DG, default is lumped
    "method": "MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    "automatic_adjoint": False,
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
default_dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
default_dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
}
default_dictionary["synthetic_data"] = {    #For use only if you are using a synthetic test model or a forward only simulation -adicionar discrição para modelo direto
    "real_mesh_file": None,
    "real_velocity_file": None,
}
default_dictionary["inversion"] = {
    "perform_fwi": False, # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": default_optimization_parameters,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
default_dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
default_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 0.9), 20
    ),
}

# Simulate for 2.0 seconds.
default_dictionary["time_axis"] = {
    "initial_time": 0.0,  #  Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}

class model_parameters:
    def __init__(self, dictionary=default_dictionary, comm = None):
        '''Initializes class that reads and sanitizes input parameters.
        A dictionary can be used.

        Parameters
        ----------
        dictionary: 'dictionary' (optional)
            Contains all input parameters already organized based on examples from github.
        comm: MPI communicator (optional)
            MPI comunicator. If None is given model_parameters creates one.

        Returns
        -------
        model_parameters: :class: 'model_parameters' object
        '''
        # Converts old dictionary to new one. Deprecated feature
        if 'opts' in dictionary:
            warnings.warn("Old deprecated dictionary style in usage.")
            dictionary = self.__convert_old_dictionary(dictionary)
        # Saves inout_dictionary internally
        self.input_dictionary = dictionary

        #Sanitizes method or cell_type+variant inputs
        self.cell_type = None
        self.method = None
        self.variant = None
        self.__get_method()
        
        #Checks if degree is valid
        self.degree = dictionary["options"]["degree"]
        self.dimension = dictionary["options"]["dimension"]
        self.__check_degree()

        #Checks time inputs
        self.final_time = dictionary["time_axis"]["final_time"]
        self.dt = dictionary["time_axis"]['dt']
        self.__check_time()

        # Check if we are doing a FWI and sorting output locations and velocity model inputs
        self.running_fwi = False
        if "inversion" in dictionary:
            if dictionary["inversion"]["perform_fwi"]:
                self.running_fwi = True
        if self.running_fwi:
            self.initial_velocity_model = dictionary["inversion"]["initial_velocity_model"]
            self.fwi_output_folder = 'fwi/'
            self.control_output_file = self.fwi_output_folder+'control'
            self.gradient_output_file = self.fwi_output_folder+'gradient'
            self.optimization_parameters = dictionary["inversion"]["optimization_parameters"]
        else:
            self.initial_velocity_model = dictionary["synthetic_data"]["real_velocity_file"]

        self.foward_output_file = 'results/forward_output.pvd'

        # Checking mesh_parameters
        self.mesh_file = dictionary["mesh"]["mesh_file"]
        if "user_mesh" in dictionary["mesh"]:
            if dictionary["mesh"]["user_mesh"]:
                self.user_mesh = dictionary["mesh"]["user_mesh"]
            else:
                self.user_mesh = False
        else:
            self.user_mesh = False

        if self.mesh_file == 'not_used.msh':
            self.mesh_file = None
        self.__check_mesh() #Olhar objeto do Firedrake - assumir retangular sempre -só warning se z nao for negativo

        # Checking source and receiver inputs
        self.number_of_sources = len(dictionary["acquisition"]["source_locations"])
        self.source_locations = dictionary["acquisition"]["source_locations"]
        self.number_of_receivers = len(dictionary["acquisition"]["receiver_locations"])
        self.receiver_locations = dictionary["acquisition"]["receiver_locations"]
        self.__check_acquisition()

    # def __check_mesh(self):
        

    def __check_acquisition(self):
        min_z = -self.length_z
        max_z = 0.0
        min_x = 0.0
        max_x = self.length_x
        if self.dimension == 3:
            min_y = 0.0
            max_y = self.length_y
        for source in self.source_locations:
            if self.dimension == 2:
                source_z, source_x = source
                source_y = 0.0
            elif self.dimension == 3:
                source_z, source_x, source_y = source
            else:
                raise ValueError('Source input type not supported')
            if min_z > source_z or source_z > max_z:
                raise ValueError(f'Source of ({source_z},{source_x}, {source_y}) not located inside the mesh.')
            if min_x > source_x or source_x > max_x:
                raise ValueError(f'Source of ({source_z},{source_x}, {source_y}) not located inside the mesh.')
            if (min_y > source_y or source_y > max_y) and self.dimension == 3:
                raise ValueError(f'Source of ({source_z},{source_x}, {source_y}) not located inside the mesh.')

    def __check_time(self):
        if self.final_time < 0.0:
            raise ValueError(f'Negative time of {self.final_time} not valid.')
        if self.dt > 1.0:
            warnings.warn(f'Time step of {self.dt} too big.')
        if self.dt == None:
            warnings.warn('Timestep not given. Will calculate internally when user attemps to propagate wave.')

    def __check_degree(self):
        if self.method == 'mass_lumped_triangle':
            if self.dimension == 2:
                if self.degree > 5:
                    raise ValueError(f'Degree of {self.degree} not supported by {self.dimension}D {self.method}.')
            if self.dimension == 3:
                if self.degree > 4:
                    raise ValueError(f'Degree of {self.degree} not supported by {self.dimension}D {self.method}.')
            if self.dimension == 3:
                if self.degree == 4:
                    warnings.warn(f'Degree of {self.degree} not supported by {self.dimension}D {self.method} in main firedrake.')
        
    def __convert_old_dictionary(self,old_dictionary):
        new_dictionary = {}
        new_dictionary["options"] = {
            "method": old_dictionary["opts"]["method"],
            "variant": old_dictionary["opts"]["quadrature"],
            "degree":old_dictionary["opts"]["degree"],
            "dimension":old_dictionary["opts"]["dimension"],
        }
        new_dictionary["parallelism"] = {
            "type": old_dictionary["parallelism"]["type"],  # options: automatic (same number of cores for evey processor) or spatial
        }
        new_dictionary["mesh"] = {
            "Lz": old_dictionary["mesh"]["Lz"],
            "Lx": old_dictionary["mesh"]["Lx"],
            "Ly": old_dictionary["mesh"]["Ly"],
            "mesh_file": old_dictionary["mesh"]["meshfile"],
        }
        fwi_running = False
        if old_dictionary["mesh"]["initmodel"] != None and old_dictionary["mesh"]["truemodel"] != None:
            if old_dictionary["mesh"]["initmodel"] != "not_used.hdf5" and old_dictionary["mesh"]["truemodel"] != "not_used.hdf5":
                warnings.warn("Assuming parameters set for fwi.")
                fwi_running = True
        if fwi_running == False:
            warnings.warn("Assuming parameters set for forward only propagation, will use velocity model from old_dictionary truemodel.")
        if fwi_running:
            new_dictionary["synthetic_data"] = {
                "real_velocity_file": old_dictionary["mesh"]["truemodel"],
                "real_mesh_file": None,
            }
        else:
            model_file = None
            if old_dictionary["mesh"]["initmodel"] != None and old_dictionary["mesh"]["initmodel"] != "not_used.hdf5":
                model_file = old_dictionary["mesh"]["initmodel"]
            else:
                model_file = old_dictionary["mesh"]["truemodel"]
            new_dictionary["synthetic_data"] = {
                "real_velocity_file": model_file,
                "real_mesh_file": None,
            }        
        if fwi_running:
            warnings.warn("Using default optimization parameters.")
            default_optimization_parameters = {
                "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
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
            old_default_shot_record_file = "shots/shot_record_1.dat"
            shot_record_file = None
            if exists(old_default_shot_record_file):
                shot_record_file = old_default_shot_record_file
            new_dictionary["inversion"] = {
                "perform_fwi": True, # switch to true to make a FWI
                "initial_guess_model_file": old_dictionary["mesh"]["initmodel"],
                "shot_record_file": shot_record_file,
                "optimization_parameters": default_optimization_parameters,
            }
        else:
            new_dictionary["inversion"] = {
                "perform_fwi": False, # switch to true to make a FWI
                "initial_guess_model_file": None,
                "shot_record_file": None,
                "optimization_parameters": None,
            }
        new_dictionary["absorving_boundary_conditions"] = old_dictionary["BCs"]
        new_dictionary["acquisition"] = {
            "source_type": old_dictionary["acquisition"]["source_type"],
            "source_locations": old_dictionary["acquisition"]["source_pos"],
            "frequency": old_dictionary["acquisition"]["frequency"],
            "delay": old_dictionary["acquisition"]["delay"],
            "amplitude": old_dictionary["timeaxis"]["amplitude"],
            "receiver_locations": old_dictionary["acquisition"]["receiver_locations"],
        }
        new_dictionary["time_axis"] = {
            "initial_time": old_dictionary["timeaxis"]["t0"],  #  Initial time for event
            "final_time": old_dictionary["timeaxis"]["tf"],  # Final time for event
            "dt": old_dictionary["timeaxis"]["dt"],  # timestep size
            "nspool": old_dictionary["timeaxis"]["nspool"],  # how frequently to output solution to pvds
            "fspool": old_dictionary["timeaxis"]["fspool"],  # how frequently to save solution to RAM
        }

        return new_dictionary
                
    def __unify_method_input(self):
        unified_method = None
        method = self.method
        if method == 'KMV' or method == 'MLT' or method == 'mass_lumped_triangle' or method == 'mass_lumped_tetrahedra':
            unified_method = 'mass_lumped_triangle'
        elif method == 'spectral' or method == 'SEM' or method == 'spectral_quadrilateral':
            unified_method = 'spectral_quadrilateral'
        elif method == 'DG_triangle':
            unified_method = method
        elif method == 'DG_quadrilateral':
            unified_method = method
        elif method == 'CG':
            unified_method = method
        else:
            warnings.warn(f"Method of {method} not accepted.")
        self.method = unified_method

    def __unify_cell_type_input(self):
        unified_cell_type = None
        cell_type = self.cell_type
        if cell_type == 'T' or cell_type == 'triangles' or cell_type == 'triangle' or cell_type == 'tetrahedron' or cell_type == 'tetrahedra':
            unified_cell_type = 'triangle'
        elif cell_type == 'Q' or cell_type == 'quadrilateral' or cell_type == 'quadrilaterals' or cell_type == 'hexahedron' or cell_type == 'hexahedra':
            unified_cell_type = 'quadrilateral'
        elif cell_type == None:
            unified_cell_type = None
        else:
            warnings.warn(f"Cell type of {cell_type} not accepted.")
        self.cell_type = unified_cell_type

    def __unify_variant_input(self):
        unified_variant = None
        variant = self.variant

        if variant == 'spectral' or variant == 'GLL' or variant == 'SEM' or variant == 'lumped' or variant == 'KMV' :
            unified_variant = 'lumped'
        elif variant == 'equispaced' or variant == 'equis':
            unified_variant = 'equispaced'
        elif variant == 'DG' or variant == 'discontinuous_galerkin':
            unified_variant = 'DG'
        else:
            warnings.warn(f"Variant of {variant} not accepted.")
        self.method = unified_variant

    def __get_method_from_cell_type(self):
        cell_type = self.cell_type
        variant = self.variant
        if cell_type == 'triangle':
            if   variant == 'lumped':
                method = 'mass_lumped_triangle'
            elif variant == 'equispaced':
                method = 'CG_triangle'
            elif variant == 'DG':
                method = 'DG_triangle'
        elif cell_type == 'quadrilateral':
            if   variant == 'lumped':
                method = 'spectral_quadrilateral'
            elif variant == 'equispaced':
                method = 'CG_quadrilateral'
            elif variant == 'DG':
                method = 'DG_quadrilateral'
        self.method = method

    def __get_method(self):
        dictionary = self.input_dictionary
        # Checking if method/cell_type + variant specified twice:
        if "method" in dictionary["options"] and ("cell_type" in dictionary["options"]) and ("variant" in dictionary["options"]):
            if dictionary["options"]["method"] != None and dictionary["options"]["cell_type"] != None:
                warnings.warn("Both methods of specifying method and cell_type with variant used. Method specification taking priority.")
        if "method" in dictionary["options"] and dictionary["options"]["method"] != None:
            self.method = dictionary["options"]["method"]
            self.__unify_method_input()
            # For backwards compatibility
            if "variant" in dictionary["options"]:
                if dictionary["options"]["variant"] == 'spectral' or dictionary["options"]["variant"] == 'GLL' and self.method == 'CG':
                    self.method = 'spectral_quadrilateral'
                
        elif ("cell_type" in dictionary["options"]) and ("variant" in dictionary["options"]) and dictionary["options"]["cell_type"] != None:
            self.cell_type = dictionary["options"]["cell_type"]
            self.__unify_cell_type_input()
            self.variant   = dictionary["options"]["variant"]
            self.__unify_variant_input()
            self.__get_method_from_cell_type()
        else:
            raise ValueError("Missing options inputs.")

    # def get_wavelet(self):
    #     dictionary = self.input_dictionary
    #     source_type = dictionary["acquisition"]

    def get_mesh(self):
        """Reads in an external mesh and scatters it between cores.

        Parameters
        ----------
        model: `dictionary`
            Model options and parameters.
        ens_comm: Firedrake.ensemble_communicator
            An ensemble communicator

        Returns
        -------
        mesh: Firedrake.Mesh object
            The distributed mesh across `ens_comm`
        """
        if self.mesh_file != None:
            return spyro.io.read_mesh(self)
        elif self.mesh_type == 'user_mesh':
            return self.user_mesh

        
        




