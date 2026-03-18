import numpy as np
import spyro


def test_misfit_2d():
    default_optimization_parameters = {
        "General": {
            "Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}
        },
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

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped
        "method": "MLT",  # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 1,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "length_z": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
        "length_x": 3.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
    # We also specify to record the solution at 101 microphones near the top of the domain.
    # This transect of receivers is created with the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 1.5)],
        "frequency": 5.0,
        "delay": 1.5,
        "delay_type": "multiples_of_minimum",
        "receiver_locations": spyro.create_transect((-2.9, 0.1), (-2.9, 2.9), 100),
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.00,  # Final time for event
        "dt": 0.001,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": True,
        "gradient_filename": "results/Gradient.pvd",
        "adjoint_output": False,
        "adjoint_filename": None,
        "debug_output": True,
    }
    dictionary["inversion"] = {
        "perform_fwi": True,
        "initial_guess_model_file": None,
        "shot_record_file": None,
        "optimization_parameters": default_optimization_parameters,
    }

    # Using FWI Object
    FWI_obj = spyro.FullWaveformInversion(dictionary=dictionary)
    FWI_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.05})
    FWI_obj.set_real_velocity_model(
        expression="4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((x - 1.5) ** 2 + (z + 1.5) ** 2)))",
    )
    FWI_obj.generate_real_shot_record()

    FWI_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.05})
    FWI_obj.set_guess_velocity_model(constant=4.0)
    misfit = FWI_obj.calculate_misfit()

    # Using only wave objects
    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.05})
    Wave_obj_exact.set_initial_velocity_model(
        expression="4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((x - 1.5) ** 2 + (z + 1.5) ** 2)))",
        output=True
    )
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.receivers_output

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.05})
    Wave_obj_guess.set_initial_velocity_model(constant=4.0)
    Wave_obj_guess.forward_solve()
    rec_out_guess = Wave_obj_guess.receivers_output

    misfit_second_calc = rec_out_exact - rec_out_guess

    arevaluesclose = np.isclose(misfit, misfit_second_calc)
    test = arevaluesclose.all()

    print(f"Misfit calculated with FWI object is close to the individually calculated: {test}")

    assert test


if __name__ == "__main__":
    test_misfit_2d()
