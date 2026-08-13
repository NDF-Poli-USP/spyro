from pytest import mark
import spyro
from firedrake import COMM_WORLD as comm, conditional
from numpy import asarray
from pickle import load
from spyro.tools.error_measure import MeasureError
from spyro.utils.cost import comp_cost
from spyro.io.basicio import parallel_print as pprint


def run_forward():
    dt = 0.0001

    # Reference to resource usage
    tRef = comp_cost("tini")

    final_time = 1.4

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
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
        "length_z": 1.0,  # depth in km - always positive
        "length_x": 1.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
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
        "abc_type": "PML",
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

    wave = spyro.solvers.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.02})

    z = wave.mesh_z
    cond = conditional(
        z > -0.333, 1.5, conditional(z > -0.667, 3.0, 4.5)
    )
    wave.set_initial_velocity_model(conditional=cond)
    wave.forward_solve()

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, save_time=False)

    p_r = wave.forward_solution_receivers

    return p_r, wave.dt


@mark.slow
@mark.skip(reason="Ruben is implementing a right PML formulation")
def test_pml():
    """Test that the second order time convergence
    of the central difference method is achieved"""

    p_r, dt = run_forward()
    with open("tests/inputfiles/extended_pml_receveirs.pck", "rb") as f:
        array = asarray(load(f), dtype=float)
        extended_p_r = array

    # Computing errors
    measure_error = MeasureError()
    errPk = measure_error.peak_error(p_r, extended_p_r)[0]
    errIt = measure_error.integral_error(p_r, extended_p_r, dt)
    eNRMS = measure_error.normalized_root_mean_square_error(p_r, extended_p_r)

    pprint(f"NRMS Error = {eNRMS:.4e}", comm=comm)
    pprint(f"Integral Error = {errIt:.4e}", comm=comm)
    pprint(f"Peak Error = {errPk:.4e}", comm=comm)

    assert eNRMS < 0.05 and errIt < 0.05 and errPk < 0.05, "Error is too high for PML test."


if __name__ == "__main__":
    test_pml()
