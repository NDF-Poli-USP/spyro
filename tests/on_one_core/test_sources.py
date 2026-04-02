import math
from copy import deepcopy
import firedrake as fire
import numpy as np
import spyro
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.solvers.elastic_wave.isotropic_wave import IsotropicWave

"""Read in an external mesh and interpolate velocity to it"""
from ..inputfiles.Model1_2d_CG import model as oldmodel


def test_ricker_varies_in_time():
    """This test ricker time variation when applied to a time-
    dependent PDE (acoustic wave second order in pressure) in
    firedrake. It tests if the right hand side varies in time
    and if the applied ricker function behaves correctly
    """

    # initial ricker tests
    modelRicker = deepcopy(oldmodel)
    frequency = 2
    amplitude = 3

    # tests if ricker starts at zero
    delay = 1.5 * math.sqrt(6.0) / (math.pi * frequency)
    t = 0.0
    r0 = spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude)
    test1 = math.isclose(
        r0,
        0,
        abs_tol=1e-3,
    )

    # tests if the minimum value is correct and occurs at correct locations
    minimum = -amplitude * 2 / math.exp(3.0 / 2.0)
    t = 0.0 + delay + math.sqrt(6.0) / (2.0 * math.pi * frequency)
    rmin1 = spyro.sources.timedependentSource(
        modelRicker, t, frequency, amplitude
    )
    test2 = math.isclose(
        rmin1,
        minimum,
    )

    t = 0.0 + delay - math.sqrt(6.0) / (2.0 * math.pi * frequency)
    rmin2 = spyro.sources.timedependentSource(
        modelRicker, t, frequency, amplitude
    )
    test3 = math.isclose(rmin2, minimum)

    # tests if maximum value in correct and occurs at correct location
    t = 0.0 + delay
    rmax = spyro.sources.timedependentSource(
        modelRicker, t, frequency, amplitude
    )
    test4 = math.isclose(
        rmax,
        amplitude,
    )

    assert all([test1, test2, test3, test4])


def _base_time_axis():
    return {
        "initial_time": 0.0,
        "final_time": 0.001,
        "dt": 0.001,
        "output_frequency": 1,
        "gradient_sampling_frequency": 1,
    }


def _base_mesh(dimension):
    mesh = {
        "length_z": 1.0,
        "length_x": 1.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    if dimension == 3:
        mesh["length_y"] = 1.0
    else:
        mesh["length_y"] = 0.0
    return mesh


def _base_visualization():
    return {
        "forward_output": False,
        "gradient_output": False,
        "adjoint_output": False,
        "debug_output": False,
    }


def _build_elastic_wave(amplitude, source_locations, dimension=2):
    receiver_location = (-0.2, 0.6) if dimension == 2 else (-0.2, 0.6, 0.6)
    model = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": dimension,
        },
        "parallelism": {"type": "custom", "shot_ids_per_propagation": [list(range(len(source_locations)))]},
        "mesh": _base_mesh(dimension),
        "acquisition": {
            "source_type": "ricker",
            "source_locations": source_locations,
            "frequency": 5.0,
            "delay": 1.0,
            "amplitude": amplitude,
            "receiver_locations": [receiver_location],
            "use_vertex_only_mesh": True,
        },
        "time_axis": _base_time_axis(),
        "synthetic_data": {
            "type": "object",
            "density": 1.0,
            "lambda": 1.0,
            "mu": 1.0,
            "real_velocity_file": None,
        },
        "visualization": _base_visualization(),
    }

    wave = IsotropicWave(model)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave._initialize_model_parameters()
    wave.matrix_building()
    wave.sources.current_sources = list(range(len(source_locations)))
    return wave


def _build_acoustic_wave(amplitude, source_locations):
    model = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {"type": "custom", "shot_ids_per_propagation": [list(range(len(source_locations)))]},
        "mesh": _base_mesh(2),
        "acquisition": {
            "source_type": "ricker",
            "source_locations": source_locations,
            "frequency": 5.0,
            "delay": 1.0,
            "amplitude": amplitude,
            "receiver_locations": [(-0.2, 0.6)],
            "use_vertex_only_mesh": True,
        },
        "time_axis": _base_time_axis(),
        "visualization": _base_visualization(),
    }

    wave = AcousticWave(model)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave.set_initial_velocity_model(constant=1.5)
    wave._initialize_model_parameters()
    wave.matrix_building()
    wave.sources.current_sources = list(range(len(source_locations)))
    return wave


def _source_action(source_cofunction, field):
    return fire.assemble(fire.action(source_cofunction, field))


def test_elastic_vertex_only_mesh_preserves_amplitude_vector():
    amplitude = [0.25, -0.5]
    wave = _build_elastic_wave(amplitude, [(-0.2, 0.5)])

    source_cofunction = wave.sources.source_cofunction()
    field = fire.Function(wave.function_space).interpolate(fire.as_vector((2.0, 3.0)))

    expected = np.dot(np.asarray(amplitude, dtype=float), np.array([2.0, 3.0]))
    assert np.isclose(_source_action(source_cofunction, field), expected)


def test_elastic_vertex_only_mesh_sums_multiple_active_sources():
    amplitude = np.array([0.25, -0.5])
    source_locations = [(-0.2, 0.5), (-0.8, 0.5)]
    wave = _build_elastic_wave(amplitude, source_locations)

    source_cofunction = wave.sources.source_cofunction()
    field = fire.Function(wave.function_space).interpolate(fire.as_vector((2.0, 3.0)))

    expected_per_source = np.dot(amplitude, np.array([2.0, 3.0]))
    assert np.isclose(
        _source_action(source_cofunction, field),
        len(source_locations) * expected_per_source,
    )


def test_acoustic_vertex_only_mesh_uses_scalar_amplitude():
    amplitude = 2.5
    source_locations = [(-0.2, 0.5), (-0.8, 0.5)]
    wave = _build_acoustic_wave(amplitude, source_locations)

    source_cofunction = wave.sources.source_cofunction()
    field = fire.Function(wave.function_space).assign(3.0)

    assert np.isclose(_source_action(source_cofunction, field), len(source_locations) * amplitude * 3.0)


if __name__ == "__main__":
    test_ricker_varies_in_time()
