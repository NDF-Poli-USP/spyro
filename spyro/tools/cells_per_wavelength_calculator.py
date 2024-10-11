import numpy as np
import time as timinglib
import copy
from .input_models import create_initial_model_for_meshing_parameter
import spyro


def wave_propagator(dictionary, equation_type):
    if equation_type == "acoustic":
        return spyro.AcousticWave(dictionary)
    elif equation_type == "isotropic_elastic":
        return spyro.IsotropicWave(dictionary)


class Meshing_parameter_calculator:
    """
    A class used to calculate meshing parameter C (cells-per-wavelength).

    ...

    Attributes
    ----------
    parameters_dictionary : dict
        a dictionary containing all the parameters needed for the calculation
    source_frequency : float
        the source frequency in Hz used for the calculation
    minimum_velocity : float
        the minimum velocity in the domain in km/s
    velocity_profile_type : str
        the type of velocity profile, either "homogeneous" or "heterogeneous"
    velocity_model_file_name : str
        the file name of the velocity model .segy file
    FEM_method_to_evaluate : str
        the Finite Element Method to be evaluated, either "mass_lumped_triangle" or "spectral_quadrilateral"
    dimension : int
        the spatial dimension of the problem (either 2 or 3)
    receiver_setup : str
        the setup of the receiver either "near", "line", or "far"
    accepted_error_threshold : float
        the accepted error threshold for the calculation. The cpw calculation stops when
        the error is below this threshold. Usually 0.05.
    desired_degree : int
        the desired polynoial element degree for the calculation
    reference_degree : int
        the polynomial degree to be used for the calculation of the reference case
    cpw_reference : float
        the cells-per-wavelength to be used for mesh generation of the reference solution
    cpw_initial : float
        the initial guess for the cells-per-wavelength parameter
    cpw_accuracy : float
        the accuracy of the cells-per-wavelength parameter
    reduced_obj_for_testing : bool
        a boolean to reduce the object size for testing purposes
    save_reference : bool
        a boolean to chose to save the reference solution
    load_reference : bool
        a boolean to load the reference solution, if used paramters_dictionary should also have a "reference_solution_file" key.
    timestep_calculation : str
        a string to define the time-step calculation method, either "exact", "estimate", or "float".
    fixed_timestep : float
        a float to define the fixed time-step if the time-step calculation method is "float"
    estimate_timestep : bool
        a boolean to define if the time-step should be estimated
    initial_guess_object : spyro.AcousticWave
        the initial guess object for the calculation
    comm : mpi4py.MPI.Intracomm
        the MPI communicator
    reference_solution : np.ndarray
        the reference solution
    initial_dictionary : dict
        the initial dictionary used to build the initial guess object

    Methods
    -------
    _check_velocity_profile_type():
        Checks the type of velocity profile.
    _check_heterogenous_mesh_lengths():
        Checks the lengths of the heterogeneous mesh.
    build_initial_guess_model():
        Builds the initial guess model.
    get_reference_solution():
        Gets or generates the reference solution.
    calculate_reference_solution():
        Calculates the reference solution.
    calculate_analytical_solution():
        Calculates the analytical reference solution if it is possible.
    find_minimum(starting_cpw=None, TOL=None, accuracy=None, savetxt=False):
        Finds the minimum cells-per-wavelength meshing parameter.
    build_current_object(cpw, degree=None):
        Builds the current acoustic wave solver object.
    """
    def __init__(self, parameters_dictionary):
        """
        Initializes the Meshing_parameter_calculator class with a dictionary of parameters.

        Parameters
        ----------
        parameters_dictionary : dict
            A dictionary containing all the parameters needed for the calculation. It should include:
            - "source_frequency": float, the source frequency for the calculation
            - "minimum_velocity_in_the_domain": float, the minimum velocity in the domain for the calculation
            - "velocity_profile_type": str, the type of velocity profile for the calculation
            - "velocity_model_file_name": str, the file name of the velocity model for the calculation
            - "FEM_method_to_evaluate": str, the Finite Element Method to be evaluated for the calculation
            - "dimension": int, the dimension of the problem
            - "receiver_setup": str, the setup of the receiver
            - "accepted_error_threshold": float, the accepted error threshold for the calculation
            - "desired_degree": int, the desired degree for the calculation
        """
        parameters_dictionary.setdefault("source_frequency", 5.0)
        parameters_dictionary.setdefault("minimum_velocity_in_the_domain", 1.5)
        parameters_dictionary.setdefault("velocity_profile_type", "homogeneous")
        parameters_dictionary.setdefault("velocity_model_file_name", None)
        parameters_dictionary.setdefault("FEM_method_to_evaluate", "mass_lumped_triangle")
        parameters_dictionary.setdefault("dimension", 2)
        parameters_dictionary.setdefault("receiver_setup", "near")
        parameters_dictionary.setdefault("load_reference", False)
        parameters_dictionary.setdefault("save_reference", True)
        parameters_dictionary.setdefault("time-step_calculation", "estimate")
        parameters_dictionary.setdefault("C_accuracy", 0.1)
        parameters_dictionary.setdefault("equation_type", "acoustic")
        parameters_dictionary.setdefault("accepted_error_threshold", 0.05)
        parameters_dictionary.setdefault("testing", False)

        self.equation_type = parameters_dictionary["equation_type"]
        self.parameters_dictionary = parameters_dictionary
        self.source_frequency = parameters_dictionary["source_frequency"]
        self.minimum_velocity = parameters_dictionary[
            "minimum_velocity_in_the_domain"
        ]
        self.velocity_profile_type = parameters_dictionary[
            "velocity_profile_type"
        ]
        self.velocity_model_file_name = parameters_dictionary[
            "velocity_model_file_name"
        ]
        self._check_velocity_profile_type()
        self.FEM_method_to_evaluate = parameters_dictionary[
            "FEM_method_to_evaluate"
        ]
        self.dimension = parameters_dictionary["dimension"]
        self.receiver_setup = parameters_dictionary["receiver_setup"]
        self.accepted_error_threshold = parameters_dictionary[
            "accepted_error_threshold"
        ]
        self.desired_degree = parameters_dictionary["desired_degree"]

        # Only for use in heterogenoeus models
        self.reference_degree = parameters_dictionary["reference_degree"]
        self.cpw_reference = parameters_dictionary["C_reference"]

        # Initializing optimization parameters
        self.cpw_initial = parameters_dictionary["C_initial"]
        self.cpw_accuracy = parameters_dictionary["C_accuracy"]

        # Debugging and testing  parameters
        self.reduced_obj_for_testing = self.parameters_dictionary["testing"]

        # Setting up reference file read or load
        self.save_reference = self.parameters_dictionary["save_reference"]
        self.load_reference = self.parameters_dictionary["load_reference"]

        # Setting up time-step attributes
        self._setting_up_time_step()

        self.initial_guess_object = self.build_initial_guess_model()
        self.comm = self.initial_guess_object.comm
        self.reference_solution = self.get_reference_solution()

    def _setting_up_time_step(self):
        if "time-step_calculation" in self.parameters_dictionary:
            self.timestep_calculation = self.parameters_dictionary["time-step_calculation"]
        else:
            self.timestep_calculation = "exact"
        self.fixed_timestep = None

        if self.timestep_calculation == "exact":
            self.estimate_timestep = False
        elif self.timestep_calculation == "estimate":
            self.estimate_timestep = True
        else:
            self.estimate_timestep = None
            self.fixed_timestep = self.parameters_dictionary["time-step"]

    def _check_velocity_profile_type(self):
        if self.velocity_profile_type == "homogeneous":
            if self.velocity_model_file_name is not None:
                raise ValueError(
                    "Velocity model file name should be None for homogeneous models"
                )
        elif self.velocity_profile_type == "heterogeneous":
            self._check_heterogenous_mesh_lengths()
            if self.velocity_model_file_name is None:
                raise ValueError(
                    "Velocity model file name should be defined for heterogeneous models"
                )
        else:
            raise ValueError(
                "Velocity profile type is not homogeneous or heterogeneous"
            )

    def _check_heterogenous_mesh_lengths(self):
        parameters = self.parameters_dictionary
        if "length_z" not in parameters:
            raise ValueError("Length in z direction not defined")
        if "length_x" not in parameters:
            raise ValueError("Length in x direction not defined")
        if parameters["length_z"] is None:
            raise ValueError("Length in z direction not defined")
        if parameters["length_x"] is None:
            raise ValueError("Length in x direction not defined")
        if parameters["length_z"] < 0.0:
            parameters["length_z"] = abs(parameters["length_z"])
        if parameters["length_x"] < 0.0:
            raise ValueError("Length in x direction must be positive")

    def build_initial_guess_model(self):
        """
        Builds the initial guess spyro acoustic wave solver object.

        Returns
        -------
        spyro.AcousticWave
            the initial guess spyro acoustic wave solver object
        """
        dictionary = create_initial_model_for_meshing_parameter(self)
        self.initial_dictionary = dictionary
        return wave_propagator(dictionary, self.equation_type)

    def get_reference_solution(self):
        """
        Calculates or loads the reference solution to be used for error calculation.

        Returns
        -------
        np.ndarray
            the reference solution
        """
        if self.load_reference:
            if "reference_solution_file" in self.parameters_dictionary:
                filename = self.parameters_dictionary["reference_solution_file"]
            else:
                filename = "reference_solution.npy"
            return np.load(filename)
        elif self.velocity_profile_type == "heterogeneous" or self.equation_type == "isotropic_elastic":
            return self.calculate_reference_solution()
        elif self.velocity_profile_type == "homogeneous" and self.equation_type == "acoustic":
            return self.calculate_analytical_solution()

    def calculate_reference_solution(self):
        """
        Calculates the numerical reference solution for heterogeneous models, using cpw and degree values in parameters dictionary.

        Returns
        -------
        np.ndarray
            the reference solution
        """
        Wave_obj = self.build_current_object(self.cpw_reference, degree=self.reference_degree)

        Wave_obj.forward_solve()
        p_receivers = Wave_obj.forward_solution_receivers

        if self.save_reference:
            np.save("reference_solution.npy", p_receivers)

        return p_receivers

    def calculate_analytical_solution(self):
        """
        Calculates the analytical reference solution for homogeneous models.

        Returns
        -------
        np.ndarray
            the reference solution
        """
        # Initializing array
        Wave_obj = self.initial_guess_object
        number_of_receivers = Wave_obj.number_of_receivers
        dt = Wave_obj.dt
        final_time = Wave_obj.final_time
        num_t = int(final_time / dt + 1)
        analytical_solution = np.zeros((num_t, number_of_receivers))

        # Solving analytical solution for each receiver
        receiver_locations = Wave_obj.receiver_locations
        source_locations = Wave_obj.source_locations
        source_location = source_locations[0]
        sz, sx = source_location
        i = 0
        for receiver in receiver_locations:
            rz, rx = receiver
            offset = np.sqrt((rz - sz) ** 2 + (rx - sx) ** 2)
            r_sol = spyro.utils.nodal_homogeneous_analytical(
                Wave_obj, offset, self.minimum_velocity
            )
            analytical_solution[:, i] = r_sol
            print(i)
            i += 1
        analytical_solution = analytical_solution / (self.minimum_velocity**2)

        if self.save_reference:
            np.save("reference_solution.npy", analytical_solution)

        return analytical_solution

    def find_minimum(self, starting_cpw=None, TOL=None, accuracy=None, savetxt=False):
        """
        Finds the minimum cells-per-wavelength meshing parameter that is still below the error threshold.

        Parameters
        ----------
        starting_cpw : float (optional)
            the starting cells-per-wavelength parameter to be used in the search. If None,
            the value from paramters_dictionary is used.
        TOL : float (optional)
            the accepted error threshold for the calculation. The cpw calculation stops when
            the error is below this threshold. Usually 0.05. If None, the value from paramters_dictionary is used.
        accuracy : float (optional)
            the accuracy of the cells-per-wavelength parameter. If None, the value from paramters_dictionary is used.
        savetxt : bool (optional)
            a boolean to chose to save the results to a text file

        Returns
        -------
        cpw : float
            the minimum cells-per-wavelength parameter that is still below the error threshold
        """
        if starting_cpw is None:
            starting_cpw = self.cpw_initial
        if TOL is None:
            TOL = self.accepted_error_threshold
        if accuracy is None:
            accuracy = self.cpw_accuracy

        error = 100.0
        cpw = starting_cpw
        print("Starting line search", flush=True)
        cpws = []
        dts = []
        errors = []
        runtimes = []

        self.fast_loop = True
        # fast_loop = False
        dif = 0.0
        cont = 0
        while error > TOL:
            print("Trying cells-per-wavelength = ", cpw, flush=True)

            # Running forward model
            Wave_obj = self.build_current_object(cpw)
            Wave_obj._initialize_model_parameters() # TO REVIEW: call to protected method

            # Setting up time-step
            if self.timestep_calculation != "float":
                Wave_obj.get_and_set_maximum_dt(
                    fraction=0.2,
                    estimate_max_eigenvalue=self.estimate_timestep
                )
            else:
                Wave_obj.dt = self.fixed_timestep
            print("Maximum dt is ", Wave_obj.dt, flush=True)

            t0 = timinglib.time()
            Wave_obj.forward_solve()
            t1 = timinglib.time()
            p_receivers = Wave_obj.forward_solution_receivers
            spyro.io.save_shots(
                Wave_obj, file_name="test_shot_record" + str(cpw)
            )

            error = error_calc(
                p_receivers, self.reference_solution, Wave_obj.dt
            )
            print("Error is ", error, flush=True)
            cpws.append(cpw)
            dts.append(Wave_obj.dt)
            errors.append(error)
            runtimes.append(t1 - t0)

            cpw, error, dif = self._updating_cpw_error_and_dif(cpw, error, dif)

            cont += 1

        self._saving_file(savetxt, np.transpose([cpws, dts, errors, runtimes]))

        return cpw - dif

    def build_current_object(self, cpw, degree=None):
        """
        Builds the current acoustic wave solver object.

        Parameters
        ----------
        cpw : float
            the current cells-per-wavelength parameter
        degree : int (optional)
            the polynomial degree to be used in the calculation. If None, the value from paramters_dictionary is used.

        Returns
        -------
        spyro.AcousticWave
            the current acoustic wave solver object
        """
        dictionary = copy.deepcopy(self.initial_dictionary)
        dictionary["mesh"]["cells_per_wavelength"] = cpw
        if degree is not None:
            dictionary["options"]["degree"] = degree
        Wave_obj = wave_propagator(dictionary, self.equation_type)
        if self.velocity_profile_type == "homogeneous":
            lba = self.minimum_velocity / self.source_frequency
            edge_length = lba / cpw
            Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})
            Wave_obj.set_initial_velocity_model(constant=self.minimum_velocity)
        elif self.velocity_profile_type == "heterogeneous":
            Wave_obj.set_mesh(mesh_parameters={"cells_per_wavelength": cpw})
        return Wave_obj

    def _saving_file(self, savetxt, info):
        """
        Saves the results to a text file.
        """
        if savetxt:
            np.savetxt(
                "p"+str(self.initial_guess_object.degree)+"_cpw_results.txt",
                info,
            )

    def _updating_cpw_error_and_dif(self, cpw, error, dif):
        """
        Updates the cells-per-wavelength parameter.
        """
        if error < self.accepted_error_threshold and dif > self.cpw_accuracy:
            cpw -= dif
            error = 100.0
            # Flooring CPW to the neartest decimal point inside accuracy
            cpw = np.round(
                (cpw + 1e-6) // self.cpw_accuracy * self.cpw_accuracy,
                int(-np.log10(self.cpw_accuracy)),
            )
            self.fast_loop = False
        else:
            dif = calculate_dif(cpw, self.cpw_accuracy, fast_loop=self.fast_loop)
            cpw += dif

        return cpw, error, dif


def calculate_dif(cpw, accuracy, fast_loop=False):
    """
    Calculates the difference between consecutive cells-per-wavelength to be used in the search.

    Parameters
    ----------
    cpw : float
        the current cells-per-wavelength parameter
    accuracy : float
        the accuracy of the cells-per-wavelength parameter
    fast_loop : bool
        a boolean to chose to use a fast loop or not

    Returns
    -------
    dif : float
        the difference between consecutive cells-per-wavelength to be used in the search
    """
    if fast_loop:
        dif = max(0.1 * cpw, accuracy)
    else:
        dif = accuracy

    return dif


def error_calc(receivers, analytical, dt):
    """
    Calculates the error between the numerical and analytical solutions.

    Parameters
    ----------
    receivers : np.ndarray
        the numerical solution to be evaluated
    analytical : np.ndarray
        the analytical or reference solution
    dt : float
        the time-step used in the numerical solution

    Returns
    -------
    error : float
        the error between the numerical and analytical solutions
    """
    rec_len, num_rec = np.shape(receivers)

    # Interpolate analytical solution into numerical dts
    final_time = dt * (rec_len - 1)
    time_vector_rec = np.linspace(0.0, final_time, rec_len)
    time_vector_ana = np.linspace(0.0, final_time, len(analytical[:, 0]))
    ana = np.zeros(np.shape(receivers))
    for i in range(num_rec):
        ana[:, i] = np.interp(
            time_vector_rec, time_vector_ana, analytical[:, i]
        )

    total_numerator = 0.0
    total_denumenator = 0.0
    for i in range(num_rec):
        diff = receivers[:, i] - ana[:, i]
        diff_squared = np.power(diff, 2)
        numerator = np.trapz(diff_squared, dx=dt)
        ref_squared = np.power(ana[:, i], 2)
        denominator = np.trapz(ref_squared, dx=dt)
        total_numerator += numerator
        total_denumenator += denominator

    squared_error = total_numerator / total_denumenator

    error = np.sqrt(squared_error)
    return error
