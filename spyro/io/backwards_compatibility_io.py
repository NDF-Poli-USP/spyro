from genericpath import exists
import warnings


class Dictionary_conversion:
    """
    Convert the old dictionary to the new one

    Attributes
    ----------
    old_dictionary : dict
        Old dictionary
    new_dictionary : dict
        New dictionary
    fwi_running : bool
        True if fwi is running, False if not

    Methods
    -------
    convert_options()
        Convert the options section of dictionary
    convert_parallelism()
        Convert the parallelism section of dictionary
    convert_mesh()
        Convert the mesh section of dictionary
    check_if_fwi()
        Check if fwi is running
    convert_synthetic_data()
        Convert the synthetic_data section of dictionary
    set_optimization_parameters()
        Set the optimization_parameters section of dictionary
    set_no_inversion()
        Set the no_inversion section of dictionary
    convert_absorving_boundary_conditions()
        Convert the absorving_boundary_conditions section of dictionary
    convert_acquisition()
        Convert the acquisition section of dictionary
    convert_time_axis()
        Convert the time_axis section of dictionary
    """

    def __init__(self, old_dictionary):
        """
        Convert the old dictionary to the new one

        Parameters
        ----------
        old_dictionary : dict
            Old dictionary

        Returns
        -------
        new_dictionary : dict
            New dictionary
        """
        self.new_dictionary = {}
        self.old_dictionary = old_dictionary
        self.fwi_running = False

        self.convert_options()
        self.convert_parallelism()
        self.convert_mesh()
        self.check_if_fwi()
        self.convert_synthetic_data()
        if self.fwi_running:
            self.set_optimization_parameters()
        else:
            self.set_no_inversion()
        self.convert_absorving_boundary_conditions()
        self.convert_acquisition()
        self.convert_time_axis()

    def convert_options(self):
        """
        Convert the options section of dictionary
        """
        self.new_dictionary["options"] = {
            "method": self.old_dictionary["opts"]["method"],
            "variant": self.old_dictionary["opts"]["quadrature"],
            "degree": self.old_dictionary["opts"]["degree"],
            "dimension": self.old_dictionary["opts"]["dimension"],
        }
        variant = self.new_dictionary["options"]["variant"]
        if variant == "GLL" or variant == "KMV":
            self.new_dictionary["options"]["variant"] = "lumped"
        if variant == "KMV":
            self.new_dictionary["options"]["method"] = "mass_lumped_triangle"
        if variant == "GLL":
            self.new_dictionary["options"]["method"] = "spectral_quadrilateral"

    def convert_parallelism(self):
        """
        Convert the parallelism section of dictionary
        """
        self.new_dictionary["parallelism"] = {
            "type": self.old_dictionary["parallelism"][
                # options: automatic (same number of cores for evey processor)
                # or spatial
                "type"
            ],
        }

    def convert_mesh(self):
        """
        Convert the mesh section of dictionary
        """
        self.new_dictionary["mesh"] = {
            "length_z": self.old_dictionary["mesh"]["Lz"],
            "length_x": self.old_dictionary["mesh"]["Lx"],
            "length_y": self.old_dictionary["mesh"]["Ly"],
            "mesh_file": self.old_dictionary["mesh"]["meshfile"],
        }

    def check_if_fwi(self):
        """
        Check if fwi is running
        """
        if (
            self.old_dictionary["mesh"]["initmodel"] is not None
            and self.old_dictionary["mesh"]["truemodel"] is not None
        ) and (
            self.old_dictionary["mesh"]["initmodel"] != "not_used.hdf5"
            and self.old_dictionary["mesh"]["truemodel"] != "not_used.hdf5"
        ):
            warnings.warn("Assuming parameters set for fwi.")
            self.fwi_running = True

        if self.fwi_running is False:
            warnings.warn(
                "Assuming parameters set for forward only propagation, will \
                    use velocity model from old_dictionary truemodel."
            )

    def convert_synthetic_data(self):
        """
        Convert the synthetic_data section of dictionary
        """
        if self.fwi_running:
            self.new_dictionary["synthetic_data"] = {
                "real_velocity_file": self.old_dictionary["mesh"]["truemodel"],
                "real_mesh_file": None,
            }
        else:
            model_file = None
            if (
                self.old_dictionary["mesh"]["initmodel"] is not None
                and self.old_dictionary["mesh"]["initmodel"] != "not_used.hdf5"
            ):
                model_file = self.old_dictionary["mesh"]["initmodel"]
            else:
                model_file = self.old_dictionary["mesh"]["truemodel"]
            self.new_dictionary["synthetic_data"] = {
                "real_velocity_file": model_file,
                "real_mesh_file": None,
            }

    def set_optimization_parameters(self):
        """
        Set the optimization_parameters section of dictionary
        """
        if self.fwi_running is False:
            pass

        warnings.warn("Using default optimization parameters.")
        default_optimization_parameters = {
            "General": {
                "Secant": {
                    "Type": "Limited-Memory BFGS",
                    "Maximum Storage": 10,
                }
            },
            "Step": {
                "Type": "Augmented Lagrangian",
                "Augmented Lagrangian": {
                    "Subproblem Step Type": "Line Search",
                    "Subproblem Iteration Limit": 5.0,
                },
                "Line Search": {
                    "Descent Method": {"Type": "Quasi-Newton Step"}
                },
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
        self.new_dictionary["inversion"] = {
            "perform_fwi": True,  # switch to true to make a FWI
            "initial_guess_model_file": self.old_dictionary["mesh"][
                "initmodel"
            ],
            "shot_record_file": shot_record_file,
            "optimization_parameters": default_optimization_parameters,
        }

    def set_no_inversion(self):
        """
        Set the no_inversion section of dictionary
        """
        self.new_dictionary["inversion"] = {
            "perform_fwi": False,  # switch to true to make a FWI
            "initial_guess_model_file": None,
            "shot_record_file": None,
            "optimization_parameters": None,
        }

    # default_dictionary["absorving_boundary_conditions"] = {
    # # thickness of the PML in the z-direction (km) - always positive
    #     "lz": 0.25,
    # # thickness of the PML in the x-direction (km) - always positive
    #     "lx": 0.25,
    # # thickness of the PML in the y-direction (km) - always positive
    #     "ly": 0.0,
    # }

    def convert_absorving_boundary_conditions(self):
        """
        convert the absorving_boundary_conditions section of dictionary
        """
        old_dictionary = self.old_dictionary["BCs"]
        if old_dictionary["status"]:
            damping_type = "PML"
        else:
            damping_type = None
        self.new_dictionary["absorving_boundary_conditions"] = {
            "status": old_dictionary["status"],
            "damping_type": damping_type,
            "exponent": old_dictionary["exponent"],
            "cmax": old_dictionary["cmax"],
            "R": old_dictionary["R"],
            "pad_length": old_dictionary["lz"],
        }

    def convert_acquisition(self):
        """
        Convert the acquisition section of dictionary
        """
        source_type = self.old_dictionary["acquisition"]["source_type"]
        if source_type == "Ricker":
            source_type = "ricker"
        self.new_dictionary["acquisition"] = {
            "source_type": source_type,
            "source_locations": self.old_dictionary["acquisition"][
                "source_pos"
            ],
            "frequency": self.old_dictionary["acquisition"]["frequency"],
            "delay": self.old_dictionary["acquisition"]["delay"],
            "amplitude": self.old_dictionary["timeaxis"]["amplitude"],
            "receiver_locations": self.old_dictionary["acquisition"][
                "receiver_locations"
            ],
        }

    def convert_time_axis(self):
        """
        Convert the time_axis section of dictionary
        """
        self.new_dictionary["time_axis"] = {
            "initial_time": self.old_dictionary["timeaxis"][
                "t0"
            ],  # Initial time for event
            "final_time": self.old_dictionary["timeaxis"][
                "tf"
            ],  # Final time for event
            "dt": self.old_dictionary["timeaxis"]["dt"],  # timestep size
            "output_frequency": self.old_dictionary["timeaxis"][
                "nspool"
            ],  # how frequently to output solution to pvds
            "gradient_sampling_frequency": self.old_dictionary["timeaxis"][
                "fspool"
            ],  # how frequently to save solution to RAM
        }
