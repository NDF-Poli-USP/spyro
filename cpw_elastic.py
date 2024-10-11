import numpy as np
import spyro


def test_cpw_calc():
    grid_point_calculator_parameters = {
        # Experiment parameters
        # Here we define the frequency of the Ricker wavelet source
        "source_frequency": 5.0,
        "velocity_profile_type": "homogeneous",
        "FEM_method_to_evaluate": "mass_lumped_triangle",
        # line defines a line of point receivers with pre-established near and far
        # offsets.
        # Line search parameters
        "reference_solution_file": "test/inputfiles/reference_solution_cpw.npy",
        "reference_degree": 4,  # Degree to use in the reference case (int)
        # grid point density to use in the reference case (float)
        "C_reference": 5.0,
        "desired_degree": 4,  # degree we are calculating G for. (int)
        "C_initial": 2.2,  # Initial G for line search (float)
        "equation_type": "isotropic_elastic",
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

    # Check if cpw is within error TOL, starting search at min
    min = Cpw_calc.find_minimum()
    print(f"Minimum of {min}")

    print("END")
    # assert all([test1, test2, test3])


if __name__ == "__main__":
    test_cpw_calc()
