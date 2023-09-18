import spyro
import pickle
import numpy as np
import copy
from scipy import interpolate


def new_error_calc(rec, ana, dt):
    _, num_rec = np.shape(rec)
    total_numerator = 0.0
    total_denumenator = 0.0
    for i in range(num_rec):
        diff = rec[:, i] - ana[:, i]
        diff_squared = np.power(diff, 2)
        numerator = np.trapz(diff_squared, dx=dt)
        ref_squared = np.power(ana[:, i], 2)
        denominator = np.trapz(ref_squared, dx=dt)
        total_numerator += numerator
        total_denumenator += denominator

    squared_error = total_numerator/total_denumenator

    error = np.sqrt(squared_error)
    return error


def error_calc(p_exact, p, model):
    """Calculates the error between the exact and the numerical solution

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
        dt = model["time_axis"]["final_time"] / times
        p_exact = time_interpolation(p_exact, p, model)
    elif times_p_exact < times_p:  # then we interpolate p
        times, receivers = p_exact.shape
        dt = model["time_axis"]["final_time"] / times
        p = time_interpolation(p, p_exact, model)
    else:  # then we dont need to interpolate
        times, receivers = p.shape
        dt = model["time_axis"]["final_time"] / times
    # p = time_interpolation(p, p_exact, model)

    max_absolute_diff = 0.0
    max_percentage_diff = 0.0

    numerator = 0.0
    denominator = 0.0
    for receiver in range(receivers):
        numerator_time_int = 0.0
        denominator_time_int = 0.0
        for t in range(times - 1):
            top_integration = (
                p_exact[t, receiver] - p[t, receiver]
            ) ** 2 * dt
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


def error_calc_line(p_exact, p, model):
    # p0 doesn't necessarily have the same dt as p_exact
    # therefore we have to interpolate the missing points
    # to have them at the same length
    # testing shape
    (times_p_exact,) = p_exact.shape
    (times_p,) = p.shape
    if times_p_exact > times_p:  # then we interpolate p_exact
        (times,) = p.shape
        dt = model["time_axis"]["final_time"] / times
        p_exact = time_interpolation_line(p_exact, p, model)
    elif times_p_exact < times_p:  # then we interpolate p
        (times,) = p_exact.shape
        dt = model["time_axis"]["final_time"] / times
        p = time_interpolation_line(p, p_exact, model)
    else:  # then we dont need to interpolate
        (times,) = p.shape
        dt = model["time_axis"]["final_time"] / times

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
        print(
            "Warning: receivers don't appear to register a shot.",
            flush=True,
        )
        error = 0.0

    return error


def time_interpolation(p_old, p_exact, model):
    times, receivers = p_exact.shape
    dt = model["time_axis"]["final_time"] / times

    times_old, rec = p_old.shape
    dt_old = model["time_axis"]["final_time"] / times_old
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
    dt = model["time_axis"]["final_time"] / times

    (times_old,) = p_old.shape
    dt_old = model["time_axis"]["final_time"] / times_old
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


grid_point_calculator_parameters = {
    # Experiment parameters
    # Here we define the frequency of the Ricker wavelet source
    "source_frequency": 5.0,
    # The minimum velocity present in the domain.
    "minimum_velocity_in_the_domain": 1.5,
    # if an homogeneous test case is used this velocity will be defined in
    # the whole domain.
    # Either or heterogeneous. If heterogeneous is
    "velocity_profile_type": "homogeneous",
    # chosen be careful to have the desired velocity model below.
    "velocity_model_file_name": None,
    # FEM to evaluate such as `KMV` or `spectral`
    # (GLL nodes on quads and hexas)
    "FEM_method_to_evaluate": "mass_lumped_triangle",
    "dimension": 2,  # Domain dimension. Either 2 or 3.
    # Either near or line. Near defines a receiver grid near to the source,
    "receiver_setup": "near",
    # line defines a line of point receivers with pre-established near and far
    # offsets.
    # Line search parameters
    "load_reference": True,
    "save_reference": False,
    "reference_degree": None,  # Degree to use in the reference case (int)
    # grid point density to use in the reference case (float)
    "C_reference": None,
    "desired_degree": 4,  # degree we are calculating G for. (int)
    "C_initial": 6.0,  # Initial G for line search (float)
    "accepted_error_threshold": 0.05,
    "C_accuracy": 1e-1,
}

Cpw_calc = spyro.tools.Meshing_parameter_calculator(grid_point_calculator_parameters)
Wave_obj = Cpw_calc.initial_guess_object
with open("testing_rec.pck", "rb") as f:
    array = np.asarray(pickle.load(f), dtype=float)
    Wave_obj.forward_solution_receivers = array

rec_num = Wave_obj.forward_solution_receivers
rec_ana = Cpw_calc.reference_solution/(1.5**2)
error = new_error_calc(rec_num, rec_ana, Wave_obj.dt)
old_error = error_calc(rec_num, rec_ana, Wave_obj.input_dictionary)
print("END")
