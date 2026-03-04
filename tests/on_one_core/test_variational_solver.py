from copy import deepcopy

import firedrake as fire
import numpy as np
import spyro

from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave
from .model import dictionary as acoustic_model


def test_mass_matrix_diagonal_from_lhs():
    model = deepcopy(acoustic_model)
    model["time_axis"]["final_time"] = model["time_axis"]["dt"]
    model["time_axis"]["output_frequency"] = 1
    model["time_axis"]["gradient_sampling_frequency"] = 1

    wave = spyro.AcousticWave(dictionary=model)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave.set_initial_velocity_model(constant=1.5)
    wave._initialize_model_parameters()
    wave.matrix_building()

    diagonal = wave.get_mass_matrix_diagonal()
    assert diagonal.size > 0
    assert np.all(diagonal > 0.0)


def test_pml_3d_matrix_building_variational_setup():
    model = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 3,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "Lz": 1.0,
            "Lx": 1.0,
            "Ly": 1.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.2, 0.5, 0.5)],
            "frequency": 5.0,
            "delay": 1.0,
            "receiver_locations": [(-0.2, 0.6, 0.5)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.001,
            "dt": 0.001,
            "amplitude": 1.0,
            "output_frequency": 1,
            "gradient_sampling_frequency": 1,
        },
        "absorving_boundary_conditions": {
            "status": True,
            "damping_type": "PML",
            "exponent": 2,
            "cmax": 4.5,
            "R": 1e-6,
            "pad_length": 0.2,
        },
        "visualization": {
            "forward_output": False,
            "output_filename": "results/forward_output.pvd",
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": None,
        },
    }

    wave = spyro.AcousticWave(dictionary=model)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave.length_x = wave.mesh_parameters.length_x
    wave.length_y = wave.mesh_parameters.length_y
    wave.length_z = wave.mesh_parameters.length_z
    wave.set_initial_velocity_model(constant=1.5)
    wave._initialize_model_parameters()
    wave.matrix_building()

    assert isinstance(wave.solver, fire.LinearVariationalSolver)
    assert wave.source_function is not None
    assert wave.rhs_no_pml_source() is not None


def test_isotropic_rhs_source_accessor():
    model = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "Lz": 1.0,
            "Lx": 1.0,
            "Ly": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.2, 0.5)],
            "frequency": 5.0,
            "delay": 1.0,
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": [(-0.2, 0.6)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.001,
            "dt": 0.001,
            "output_frequency": 1,
            "gradient_sampling_frequency": 1,
        },
        "synthetic_data": {
            "type": "object",
            "density": 1.0,
            "lambda": 1.0,
            "mu": 1.0,
            "real_velocity_file": None,
        },
        "visualization": {
            "forward_output": False,
            "time": False,
            "mechanical_energy": False,
        },
    }

    wave = IsotropicWave(model)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave._initialize_model_parameters()
    wave.matrix_building()

    assert wave.rhs_no_pml_source() is wave.source_function
