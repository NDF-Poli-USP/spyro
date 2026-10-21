import math
from copy import deepcopy
import pytest
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


def _build_wave(
    wave_class,
    amplitude,
    source_locations,
    *,
    dimension,
    degree,
    synthetic_data=None,
    initial_velocity=None,
    cell_type="T",
    use_vertex_only_mesh=True,
):
    receiver_location = (-0.2, 0.6) if dimension == 2 else (-0.2, 0.6, 0.6)
    model = {
        "options": {
            "cell_type": cell_type,
            "variant": "lumped",
            "degree": degree,
            "dimension": dimension,
        },
        "parallelism": {
            "type": "custom",
            "shot_ids_per_propagation": [list(range(len(source_locations)))],
        },
        "mesh": _base_mesh(dimension),
        "acquisition": {
            "source_type": "ricker",
            "source_locations": source_locations,
            "frequency": 5.0,
            "delay": 1.0,
            "amplitude": amplitude,
            "receiver_locations": [receiver_location],
            "use_vertex_only_mesh": use_vertex_only_mesh,
        },
        "time_axis": _base_time_axis(),
        "visualization": _base_visualization(),
    }

    if synthetic_data is not None:
        model["synthetic_data"] = synthetic_data

    wave = wave_class(model)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    if initial_velocity is not None:
        wave.set_initial_velocity_model(constant=initial_velocity)
    wave.sources.current_sources = list(range(len(source_locations)))
    return wave


def _build_elastic_wave(amplitude, source_locations, dimension=2, cell_type="T"):
    return _build_wave(
        IsotropicWave,
        amplitude,
        source_locations,
        dimension=dimension,
        degree=3,
        synthetic_data={
            "type": "object",
            "density": 1.0,
            "lambda": 1.0,
            "mu": 1.0,
            "real_velocity_file": None,
        },
    )


def _build_acoustic_wave(amplitude, source_locations, cell_type="T", use_vertex_only_mesh=True):
    return _build_wave(
        AcousticWave,
        amplitude,
        source_locations,
        dimension=2,
        degree=4,
        initial_velocity=1.5,
        cell_type=cell_type,
        use_vertex_only_mesh=use_vertex_only_mesh,
    )


def _random_source_locations(dimension, count=5):
    locations = []
    for _ in range(count):
        point = [-np.random.rand()]
        point.extend(np.random.rand(dimension - 1))
        locations.append(tuple(point))
    return locations


def _check_cofunction_values(wave, source_locations, test_exprs):
    """Check that the cofunction action on each test function matches the
    expected value for each source individually."""
    for source_id, sl in enumerate(source_locations):
        wave.sources.point_locations = [sl]
        wave.sources.current_sources = [0]
        cofunction = wave.sources.source_cofunction()

        for f, expected_fn in test_exprs:
            expected = expected_fn(sl)
            action = np.dot(cofunction.dat.data_ro.ravel(), f.dat.data_ro.ravel())
            assert math.isclose(action, expected, abs_tol=1e-6), \
                f"Source {source_id} at {sl}: <c, f> = {action}, expected {expected}"


@pytest.mark.parametrize("cell_type", ["T", "Q"])
def test_cofunction_values_acoustic(cell_type):
    """Test if the cofunction correctly represents delta functions at source
    locations by checking its action on known functions."""
    np.random.seed(42)
    source_locations = _random_source_locations(2)
    wave = _build_acoustic_wave(1.0, source_locations, cell_type=cell_type)
    V = wave.function_space
    x = fire.SpatialCoordinate(wave.mesh)

    test_exprs = [
        (fire.Function(V).interpolate(1.0), lambda sl: 1.0),
        (fire.Function(V).interpolate(x[0]), lambda sl: sl[0]),
        (fire.Function(V).interpolate(x[1]), lambda sl: sl[1]),
        (fire.Function(V).interpolate(x[0]**2 + x[1]**2), lambda sl: sl[0]**2 + sl[1]**2),
    ]
    _check_cofunction_values(wave, source_locations, test_exprs)


@pytest.mark.parametrize("cell_type", ["T", "Q"])
def test_cofunction_values_elastic(cell_type):
    """Test if the elastic cofunction correctly represents delta functions at
    source locations by checking its action on known vector functions."""
    np.random.seed(42)
    source_locations = _random_source_locations(2)
    wave = _build_elastic_wave(1.0, source_locations, cell_type=cell_type)
    V = wave.function_space
    x = fire.SpatialCoordinate(wave.mesh)

    test_exprs = [
        (fire.Function(V).interpolate(fire.as_vector([1.0, 0.0])), lambda sl: 1.0),
        (fire.Function(V).interpolate(fire.as_vector([0.0, 1.0])), lambda sl: 1.0),
        (fire.Function(V).interpolate(fire.as_vector([x[0], 0.0])), lambda sl: sl[0]),
        (fire.Function(V).interpolate(fire.as_vector([0.0, x[1]])), lambda sl: sl[1]),
        (fire.Function(V).interpolate(fire.as_vector([x[0], x[1]])), lambda sl: sl[0] + sl[1]),
    ]
    _check_cofunction_values(wave, source_locations, test_exprs)


@pytest.mark.parametrize("cell_type", ["T", "Q"])
def test_cofunction_values_elastic_3d(cell_type):
    """Test if the 3D elastic cofunction correctly represents delta functions
    at source locations by checking its action on known vector functions."""
    np.random.seed(42)
    source_locations = _random_source_locations(3)
    wave = _build_elastic_wave(1.0, source_locations, dimension=3, cell_type=cell_type)
    V = wave.function_space
    x = fire.SpatialCoordinate(wave.mesh)

    test_exprs = [
        (
            fire.Function(V).interpolate(fire.as_vector([1.0, 0.0, 0.0])),
            lambda sl: 1.0,
        ),
        (
            fire.Function(V).interpolate(fire.as_vector([0.0, 1.0, 0.0])),
            lambda sl: 1.0,
        ),
        (
            fire.Function(V).interpolate(fire.as_vector([0.0, 0.0, 1.0])),
            lambda sl: 1.0,
        ),
        (
            fire.Function(V).interpolate(fire.as_vector([x[0], 0.0, 0.0])),
            lambda sl: sl[0],
        ),
        (
            fire.Function(V).interpolate(fire.as_vector([0.0, x[1], 0.0])),
            lambda sl: sl[1],
        ),
        (
            fire.Function(V).interpolate(fire.as_vector([0.0, 0.0, x[2]])),
            lambda sl: sl[2],
        ),
        (
            fire.Function(V).interpolate(fire.as_vector([x[0], x[1], x[2]])),
            lambda sl: sl[0] + sl[1] + sl[2],
        ),
    ]
    _check_cofunction_values(wave, source_locations, test_exprs)


if __name__ == "__main__":
    test_ricker_varies_in_time()


def _first_cell_centroid(mesh: fire.MeshGeometry) -> tuple[float, ...]:
    """Return the centroid of the cell numbered 0 on this rank.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        The mesh.

    Returns
    -------
    tuple of float
        The centroid coordinates.
    """
    cell_nodes = mesh.coordinates.cell_node_map().values[0]
    return tuple(mesh.coordinates.dat.data_ro[cell_nodes].mean(axis=0))


def test_source_in_first_cell_is_applied() -> None:
    """A source located in cell 0 is still injected into the right hand side.

    Cell 0 is falsy, and was once mistaken for a source outside this rank.
    """
    wave = _build_acoustic_wave(1.0, [(-0.5, 0.5)], use_vertex_only_mesh=False)
    wave.sources.point_locations = [_first_cell_centroid(wave.mesh)]
    wave.sources.build_maps()
    assert wave.sources.is_local[0] == 0

    rhs_forcing = fire.Cofunction(wave.function_space.dual())
    peak_step = int(np.argmax(np.abs(wave.sources.wavelet)))
    wave.sources.apply_source(rhs_forcing, peak_step)

    assert np.abs(rhs_forcing.dat.data_ro).max() > 0.0


def test_receiver_in_first_cell_is_injected() -> None:
    """A receiver located in cell 0 still injects its residual as a source.

    Cell 0 is falsy, and was once mistaken for a receiver outside this rank.
    """
    wave = _build_acoustic_wave(1.0, [(-0.5, 0.5)], use_vertex_only_mesh=False)
    wave.receivers.point_locations = [_first_cell_centroid(wave.mesh)]
    wave.receivers.build_maps()
    assert wave.receivers.is_local[0] == 0

    rhs_forcing = fire.Cofunction(wave.function_space.dual())
    wave.receivers.apply_receivers_as_source(rhs_forcing, [[1.0]], 0)

    assert np.abs(rhs_forcing.dat.data_ro).max() > 0.0
