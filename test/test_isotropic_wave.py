import pytest

from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave

# TO REVIEW: it is extra work to have to define this dictionary everytime
# Here I listed only the required parameters for running to get a view of
# what is currently necessary. Note that the dictionary is not even complete
dummy_dict = {
    "options": {
        "cell_type": "T",
        "variant": "lumped",
    },
    "time_axis": {
        "final_time": 1,
        "dt": 0.001,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    },
    "mesh": {},
    "acquisition": {
        "receiver_locations": [],
        "source_type": "ricker",
        "source_locations": [(0, 0)],
        "frequency": 5.0,
    },
}

def test_initialize_model_parameters_from_object_missing_parameters():
    synthetic_dict = {
        "type": "object",
    }
    wave = IsotropicWave(dummy_dict)
    with pytest.raises(Exception) as e:
        wave.initialize_model_parameters_from_object(synthetic_dict)

def test_initialize_model_parameters_from_object_first_option():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lambda": 2,
        "mu": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.initialize_model_parameters_from_object(synthetic_dict)

def test_initialize_model_parameters_from_object_second_option():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "p_wave_velocity": 2,
        "s_wave_velocity": 3,
    }
    wave = IsotropicWave(dummy_dict)
    wave.initialize_model_parameters_from_object(synthetic_dict)

def test_initialize_model_parameters_from_object_redundant():
    synthetic_dict = {
        "type": "object",
        "density": 1,
        "lmbda": 2,
        "mu": 3,
        "p_wave_velocity": 2,
        "s_wave_velocity": 3,
    }
    wave = IsotropicWave(dummy_dict)
    with pytest.raises(Exception) as e:
        wave.initialize_model_parameters_from_object(synthetic_dict)