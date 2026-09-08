import firedrake as fire
import numpy as np
import pytest

from spyro.solvers.elastic_wave.anisotropic_VTI_wave import AnisotropicVTIWave
from spyro.solvers.elastic_wave.anisotropic_TTI_wave import AnisotropicTTIWave

base_dict = {
    "options": {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 2,
        "dimension": 2,
    },
    "parallelism": {"type": "automatic"},
    "mesh": {
        "length_z": 1.0,
        "length_x": 1.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    },
    "acquisition": {
        "receiver_locations": [],
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.5)],
        "frequency": 5.0,
        "amplitude": np.array([0.0, 5]),
    },
    "time_axis": {
        "final_time": 0.5,
        "dt": 0.001,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    },
    "visualization": {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": "results/Gradient.pvd",
        "adjoint_output": False,
        "adjoint_filename": None,
        "debug_output": False,
    },
    "synthetic_data": {
        "type": "object",
        "density": 1.0,
        "p_wave_velocity": 1.5,
        "s_wave_velocity": 1.0,
        "real_velocity_file": None,
        'epsilon': 0.2, 
        'gamma': 0.1,
        'delta': 0.15,
        'anisotropy': 'exact'
    }
}


def test_VTI():
    wave = AnisotropicVTIWave(base_dict)

def test_TTI():
    base_dict["synthetic_data"]['theta'] = 30.0,
    base_dict["synthetic_data"]['phi'] = 0.0,
    wave = AnisotropicTTIWave(base_dict)