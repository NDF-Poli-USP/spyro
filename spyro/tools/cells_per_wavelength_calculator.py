import numpy as np
from scipy import interpolate
import time as timinglib
import copy
from .input_models import create_initial_model_for_meshing_parameter
import spyro


class Meshing_parameter_calculator:
    def __init__(self, parameters_dictionary):
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
        if "testing" in parameters_dictionary:
            self.reduced_obj_for_testing = parameters_dictionary["testing"]
        else:
            self.reduced_obj_for_testing = False

        if "save_reference" in parameters_dictionary:
            self.save_reference = parameters_dictionary["save_reference"]
        else:
            self.save_reference = False

        if "load_reference" in parameters_dictionary:
            self.load_reference = parameters_dictionary["load_reference"]
        else:
            self.load_reference = False

        self.initial_guess_object = self.build_initial_guess_model()
        self.reference_solution = self.get_reference_solution()

    def build_initial_guess_model(self):
        dictionary = create_initial_model_for_meshing_parameter(self)
        self.initial_dictionary = dictionary
        return spyro.AcousticWave(dictionary)

    def get_reference_solution(self):
        if self.load_reference:
            if "reference_solution_file" in self.parameters_dictionary:
                filename = self.parameters_dictionary["reference_solution_file"]
            else:
                filename = "reference_solution.npy"
            return np.load(filename)
        elif self.velocity_profile_type == "heterogeneous":
            raise NotImplementedError("Not yet implemented")
            # return self.get_referecen_solution_from refined_mesh()
        elif self.velocity_profile_type == "homogeneous":
            return self.calculate_analytical_solution()

    def calculate_analytical_solution(self):
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
        analytical_solution = analytical_solution/(self.minimum_velocity**2)

        if self.save_reference:
            np.save("reference_solution.npy", analytical_solution)

        return analytical_solution

    def find_minimum(self, starting_cpw=None, TOL=None, accuracy=None):
        if starting_cpw is None:
            starting_cpw = self.cpw_initial
        if TOL is None:
            TOL = self.accepted_error_threshold
        if accuracy is None:
            accuracy = self.cpw_accuracy

        error = 100.0
        cpw = starting_cpw
        print("Starting line search", flush=True)

        fast_loop = True
        dif = 0.0
        cont = 0
        while error > TOL:

            print("Trying cells-per-wavelength = ", cpw, flush=True)

            # Running forward model
            Wave_obj = self.build_current_object(cpw)
            # Wave_obj.get_and_set_maximum_dt(fraction=0.2)
            Wave_obj.forward_solve()
            p_receivers = Wave_obj.forward_solution_receivers
            spyro.io.save_shots(Wave_obj, file_name="test_shot_record"+str(cpw))

            error = error_calc(p_receivers, self.reference_solution, Wave_obj.dt)
            print("Error is ", error, flush=True)

            if error < TOL and dif > accuracy:
                cpw -= dif
                error = 100.0
                # Flooring CPW to the neartest decimal point inside accuracy
                cpw = np.round((cpw+1e-6) // accuracy * accuracy, int(-np.log10(accuracy)))
                fast_loop = False
            else:
                dif = calculate_dif(cpw, accuracy, fast_loop=fast_loop)
                cpw += dif

            cont += 1

        return cpw - dif

    def build_current_object(self, cpw):
        dictionary = copy.deepcopy(self.initial_dictionary)
        dictionary["mesh"]["cells_per_wavelength"] = cpw
        Wave_obj = spyro.AcousticWave(dictionary)
        lba = self.minimum_velocity / self.source_frequency

        edge_length = lba/cpw
        Wave_obj.set_mesh(edge_length=edge_length)
        Wave_obj.set_initial_velocity_model(constant=self.minimum_velocity)
        return Wave_obj


def calculate_dif(cpw, accuracy, fast_loop=False):
    if fast_loop:
        dif = max(0.1 * cpw, accuracy)
    else:
        dif = accuracy

    return dif


def error_calc(receivers, analytical, dt):
    rec_len, num_rec = np.shape(receivers)

    # Interpolate analytical solution into numerical dts
    final_time = dt*(rec_len-1)
    time_vector_rec = np.linspace(0.0, final_time, rec_len)
    time_vector_ana = np.linspace(0.0, final_time, len(analytical[:, 0]))
    ana = np.zeros(np.shape(receivers))
    for i in range(num_rec):
        ana[:, i] = np.interp(time_vector_rec, time_vector_ana, analytical[:, i])

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

    squared_error = total_numerator/total_denumenator

    error = np.sqrt(squared_error)
    return error
