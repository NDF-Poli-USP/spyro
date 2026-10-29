import numpy as np


from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave
from spyro.solvers.elastic_wave.anisotropic_VTI_wave import AnisotropicVTIWave
from spyro.solvers.elastic_wave.anisotropic_TTI_wave import AnisotropicTTIWave

base_dict = {
    "options": {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 2,
        "dimension": 3,
    },
    "parallelism": {"type": "automatic"},
    "mesh": {
        "length_z": 1.0,
        "length_x": 1.0,
        "length_y": 1.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    },
    "acquisition": {
        "receiver_locations": [],
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.5, 0.5)],
        "frequency": 5.0,
        "amplitude": np.array([0.0, 0.0, 5]),
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
        "viscoelastic": True,
        "density": 1.0,
        "p_wave_velocity": 1.5,
        "s_wave_velocity": 1.0,
        "Q_vp": 1 / 30,   # reciprocal of the P-wave quality factor
        "Q_vs": 1 / 30,   # reciprocal of the S-wave quality factor
        "real_velocity_file": None
    }
}

def test_viscoelastic_isotropic():
    wave = IsotropicWave(base_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2, "periodic": True})
    wave.forward_solve()

def test_viscoelastic_VTI():
    base_dict["synthetic_data"]['epsilon'] = 0.2
    base_dict["synthetic_data"]['gamma'] = 0.1
    base_dict["synthetic_data"]["delta"] = 0.15
    base_dict["synthetic_data"]["anisotropy"] = 'exact'
    wave = AnisotropicVTIWave(base_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2, "periodic": True})
    wave.forward_solve()

def test_viscoelastic_TTI():
    base_dict["synthetic_data"]['epsilon'] = 0.2
    base_dict["synthetic_data"]['gamma'] = 0.1
    base_dict["synthetic_data"]["delta"] = 0.15
    base_dict["synthetic_data"]['gamma'] = 0.1
    base_dict["synthetic_data"]["theta"] = 30.0
    base_dict["synthetic_data"]["phi"] = 0.0
    wave = AnisotropicTTIWave(base_dict)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2, "periodic": True})
    wave.forward_solve()