import numpy as np
import firedrake as fire

import spyro


def build_acoustic_dictionary():
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5)],
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.02,
            "dt": 0.002,
            "amplitude": 1.0,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def build_elastic_dictionary():
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5)],
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "synthetic_data": {
            "type": "object",
            "density": 1.0,
            "p_wave_velocity": 2.5,
            "s_wave_velocity": 1.0,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.02,
            "dt": 0.002,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def make_elastic_controls(fwi):
    V = fwi.get_control_parameter_function_space()
    rho = fire.Function(V, name="density").assign(1.0)
    lmbda = fire.Function(V, name="lambda")
    mu = fire.Function(V, name="mu").assign(1.0)
    z = fwi.mesh_z
    x = fwi.mesh_x
    lmbda.interpolate(
        fire.conditional(
            (z + 0.5) ** 2 + (x - 0.5) ** 2 < 0.12 ** 2,
            5.0,
            4.0,
        ),
    )
    return {
        "density": rho,
        "lambda": lmbda,
        "mu": mu,
    }


def test_full_waveform_inversion_uses_composition():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())

    assert not isinstance(fwi, spyro.Wave)
    assert isinstance(fwi.wave, spyro.AcousticWave)


def test_elastic_controls_roundtrip_with_zero_functional():
    fwi = spyro.FullWaveformInversion(
        dictionary=build_elastic_dictionary(),
        wave_class=spyro.IsotropicWave,
    )

    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_control(make_elastic_controls(fwi))
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(make_elastic_controls(fwi))

    control_vector = fwi.get_control_vector()
    assert control_vector.size == sum(
        control.dat.data.size for control in fwi.guess_control.values()
    )

    misfit = fwi.calculate_misfit(c=control_vector)
    functional = fwi.get_functional(c=control_vector)

    assert set(fwi.wave.get_control_parameters()) == {"density", "lambda", "mu"}
    assert np.allclose(misfit, 0.0, atol=1e-12)
    assert np.isclose(functional, 0.0, atol=1e-12)
