import spyro
import matplotlib.pyplot as plt
import numpy as np
import time as timer
import firedrake as fire
import pickle


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def run_forward():
    dt = 0.0001

    t0 = timer.time()

    final_time = 1.4

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
    # We also specify to record the solution at a microphone near the top of the domain.
    # This transect of receivers is created with the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.1, 0.5)],
        "frequency": 5.0,
        "delay": 0.3,
        "receiver_locations": spyro.create_transect(
            (-0.15, 0.1), (-0.15, 0.9), 50
        ),
        "delay_type": "time",
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": dt,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 200,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 200,  # how frequently to save solution to RAM
    }

    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }

    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/extended_pml_propagation.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    Wave_obj = spyro.solvers.AcousticWavePML(dictionary=dictionary)
    Wave_obj.set_mesh(dx=0.02)

    z = Wave_obj.mesh_z
    cond = fire.conditional(z > -0.333,
                            1.5,
                            fire.conditional(z > -0.667,
                                             3.0,
                                             4.5)
                            )
    Wave_obj.set_initial_velocity_model(conditional=cond)
    Wave_obj.forward_solve()

    t1 = timer.time()
    print("Time elapsed: ", t1-t0)
    nt = int(final_time/dt)+1
    p_r = Wave_obj.forward_solution_receivers

    return p_r, nt


def test_pml():
    """Test that the second order time convergence
    of the central difference method is achieved"""

    p_r, nt = run_forward()
    with open("inputfiles/extended_pml_receveirs.pck", "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)
        extended_p_r = array

    error = error_calc(extended_p_r, p_r, nt)
    print(f"Error of {error}")
    assert np.abs(error) < 0.05


if __name__ == "__main__":
    test_pml()
