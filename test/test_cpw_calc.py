import numpy as np
import spyro


def test_cpw_calc():
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
        "reference_solution_file": "test/inputfiles/reference_solution_cpw.npy",
        "save_reference": False,
        "time-step_calculation": "estimate",
        "reference_degree": None,  # Degree to use in the reference case (int)
        # grid point density to use in the reference case (float)
        "C_reference": None,
        "desired_degree": 4,  # degree we are calculating G for. (int)
        "C_initial": 2.2,  # Initial G for line search (float)
        "accepted_error_threshold": 0.05,
        "C_accuracy": 0.1,
    }

    Cpw_calc = spyro.tools.Meshing_parameter_calculator(
        grid_point_calculator_parameters
    )

    # Check correct offset
    source_location = Cpw_calc.initial_guess_object.source_locations[0]
    receiver_location = Cpw_calc.initial_guess_object.receiver_locations[1]
    sz, sx = source_location
    rz, rx = receiver_location
    offset = np.sqrt((sz - rz) ** 2 + (sx - rx) ** 2)
    expected_offset_value = 2.6580067720004026
    test1 = np.isclose(offset, expected_offset_value)
    print(f"Checked if offset calculation is correct: {test1}")

    # Check if analytical solution has the correct peak location
    analytical_solve_one_receiver = Cpw_calc.reference_solution[:, 1]
    peak_indice = np.argmax(analytical_solve_one_receiver)
    expected_peak_indice = 4052  # 2804
    test2 = expected_peak_indice == peak_indice
    print(f"Checked if reference solution seems correct: {test2}")

    # Check if cpw is within error TOL, starting search at min
    min = Cpw_calc.find_minimum()
    print(f"Minimum of {min}")
    test3 = np.isclose(2.3, min)

    print("END")
    assert all([test1, test2, test3])


if __name__ == "__main__":
    test_cpw_calc()
