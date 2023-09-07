import numpy as np
from scipy import interpolate
import copy
import spyro


class Meshing_parameter_calculator():
    def __init__(self, parameters_dictionary):
        self.parameters_dictionary = parameters_dictionary
        self.source_frequency = parameters_dictionary["source_frequency"]
        self.minimum_velocity = parameters_dictionary["minimum_velocity_in_the_domain"]
        self.velocity_profile_type = parameters_dictionary["velocity_profile_type"]
        self.velocity_model_file_name = parameters_dictionary["velocity_model_file_name"]
        self.FEM_method_to_evaluate = parameters_dictionary["FEM_method_to_evaluate"]
        self.dimension = parameters_dictionary["dimension"]
        self.receiver_setup = parameters_dictionary["receiver_setup"]
        self.accepted_error_threshold = parameters_dictionary["accepted_error_threshold"]
        self.desired_degree = parameters_dictionary["desired_degree"]

        # Only for use in heterogenoeus models
        self.reference_degree = parameters_dictionary["reference_degree"]
        self.c_reference = parameters_dictionary["C_reference"]

        # Initializing optimization parameters
        self.c_initial = parameters_dictionary["C_initial"]
        self.c_accuracy = parameters_dictionary["C_accuracy"]

        # Debugging and testing  parameters
        if "testing" in parameters_dictionary:
            self.reduced_obj_for_testing = parameters_dictionary["testing"]
        else:
            self.reduced_obj_for_testing = False

        self.initial_guess_object = self.build_initial_guess_model()
        self.reference_solution = self.get_reference_solution()

    def build_initial_guess_model(self):
        from temp_input_models import create_initial_model_for_meshing_parameter
        dictionary = create_initial_model_for_meshing_parameter(self)
        return spyro.AcousticWave(dictionary)

    def get_reference_solution(self):
        if self.velocity_profile_type == "heterogeneous":
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
        num_t = int(final_time/dt + 1)
        analytical_solution = np.zeros((num_t, number_of_receivers))

        # Solving analytical solution for each receiver
        receiver_locations = Wave_obj.receiver_locations
        source_locations = Wave_obj.source_locations
        source_location = source_locations[0]
        sz, sx = source_location
        i = 0
        for receiver in receiver_locations:
            rz, rx = receiver
            offset = np.sqrt((rz-sz)**2+(rx-sx)**2)
            r_sol = spyro.utils.nodal_homogeneous_analytical(
                Wave_obj,
                offset,
                self.minimum_velocity
            )
            analytical_solution[:, i] = r_sol
            print(i)
            i += 1

        # np.save("reference_solution.npy", analytical_solution)
        return analytical_solution


def error_calc(p_exact, p, model, comm=False):
    """ Calculates the error between the exact and the numerical solution

    Parameters
    ----------
    p_exact : `firedrake.Function`
        The exact pressure field
    p : `firedrake.Function`
        The numerical pressure field
    model : `dictionary`
        Contains simulation parameters and options.
    comm : Firedrake.ensemble_communicator, optional
        An ensemble communicator

    Returns
    -------
    error : `float`
        The error between the exact and the numerical solution

    """
    # p0 doesn't necessarily have the same dt as p_exact
    # therefore we have to interpolate the missing points
    # to have them at the same length
    # testing shape
    times_p_exact, _ = p_exact.shape
    times_p, _ = p.shape
    if times_p_exact > times_p:  # then we interpolate p_exact
        times, receivers = p.shape
        dt = model["timeaxis"]["tf"] / times
        p_exact = time_interpolation(p_exact, p, model)
    elif times_p_exact < times_p:  # then we interpolate p
        times, receivers = p_exact.shape
        dt = model["timeaxis"]["tf"] / times
        p = time_interpolation(p, p_exact, model)
    else:  # then we dont need to interpolate
        times, receivers = p.shape
        dt = model["timeaxis"]["tf"] / times
    # p = time_interpolation(p, p_exact, model)

    max_absolute_diff = 0.0
    max_percentage_diff = 0.0

    if comm.ensemble_comm.rank == 0:
        numerator = 0.0
        denominator = 0.0
        for receiver in range(receivers):
            numerator_time_int = 0.0
            denominator_time_int = 0.0
            for t in range(times - 1):
                top_integration = (p_exact[t, receiver] - p[t, receiver]) ** 2 * dt
                bot_integration = (p_exact[t, receiver]) ** 2 * dt

                # Adding 1e-25 filter to receivers to eliminate noise
                numerator_time_int += top_integration

                denominator_time_int += bot_integration

                diff = p_exact[t, receiver] - p[t, receiver]
                if abs(diff) > 1e-15 and abs(diff) > max_absolute_diff:
                    max_absolute_diff = copy.deepcopy(diff)

                if abs(diff) > 1e-15 and abs(p_exact[t, receiver]) > 1e-15:
                    percentage_diff = abs(diff / p_exact[t, receiver]) * 100
                    if percentage_diff > max_percentage_diff:
                        max_percentage_diff = copy.deepcopy(percentage_diff)

            numerator += numerator_time_int
            denominator += denominator_time_int

    if denominator > 1e-15:
        error = np.sqrt(numerator / denominator)

    # if numerator < 1e-15:
    #     print('Warning: error too small to measure correctly.', flush = True)
    #     error = 0.0
    if denominator < 1e-15:
        print("Warning: receivers don't appear to register a shot.", flush=True)
        error = 0.0

    # print("ERROR IS ", flush = True)
    # print(error, flush = True)
    # print("Maximum absolute error ", flush = True)
    # print(max_absolute_diff, flush = True)
    # print("Maximum percentage error ", flush = True)
    # print(max_percentage_diff, flush = True)
    return error


def error_calc_line(p_exact, p, model, comm=False):
    # p0 doesn't necessarily have the same dt as p_exact
    # therefore we have to interpolate the missing points
    # to have them at the same length
    # testing shape
    (times_p_exact,) = p_exact.shape
    (times_p,) = p.shape
    if times_p_exact > times_p:  # then we interpolate p_exact
        (times,) = p.shape
        dt = model["timeaxis"]["tf"] / times
        p_exact = time_interpolation_line(p_exact, p, model)
    elif times_p_exact < times_p:  # then we interpolate p
        (times,) = p_exact.shape
        dt = model["timeaxis"]["tf"] / times
        p = time_interpolation_line(p, p_exact, model)
    else:  # then we dont need to interpolate
        (times,) = p.shape
        dt = model["timeaxis"]["tf"] / times

    if comm.ensemble_comm.rank == 0:
        numerator_time_int = 0.0
        denominator_time_int = 0.0
        # Integrating with trapezoidal rule
        for t in range(times - 1):
            numerator_time_int += (p_exact[t] - p[t]) ** 2
            denominator_time_int += (p_exact[t]) ** 2
        numerator_time_int -= (
            (p_exact[0] - p[0]) ** 2 + (p_exact[times - 1] - p[times - 1]) ** 2
        ) / 2
        numerator_time_int *= dt
        denominator_time_int -= (p_exact[0] ** 2 + p_exact[times - 1] ** 2) / 2
        denominator_time_int *= dt

        # if denominator_time_int > 1e-15:
        error = np.sqrt(numerator_time_int / denominator_time_int)

        if denominator_time_int < 1e-15:
            print("Warning: receivers don't appear to register a shot.", flush=True)
            error = 0.0

    return error


def time_interpolation(p_old, p_exact, model):
    times, receivers = p_exact.shape
    dt = model["timeaxis"]["tf"] / times

    times_old, rec = p_old.shape
    dt_old = model["timeaxis"]["tf"] / times_old
    time_vector_old = np.zeros((1, times_old))
    for ite in range(times_old):
        time_vector_old[0, ite] = dt_old * ite

    time_vector_new = np.zeros((1, times))
    for ite in range(times):
        time_vector_new[0, ite] = dt * ite

    p = np.zeros((times, receivers))
    for receiver in range(receivers):
        f = interpolate.interp1d(time_vector_old[0, :], p_old[:, receiver])
        p[:, receiver] = f(time_vector_new[0, :])

    return p


def time_interpolation_line(p_old, p_exact, model):
    (times,) = p_exact.shape
    dt = model["timeaxis"]["tf"] / times

    (times_old,) = p_old.shape
    dt_old = model["timeaxis"]["tf"] / times_old
    time_vector_old = np.zeros((1, times_old))
    for ite in range(times_old):
        time_vector_old[0, ite] = dt_old * ite

    time_vector_new = np.zeros((1, times))
    for ite in range(times):
        time_vector_new[0, ite] = dt * ite

    p = np.zeros((times,))
    f = interpolate.interp1d(time_vector_old[0, :], p_old[:])
    p[:] = f(time_vector_new[0, :])

    return p
