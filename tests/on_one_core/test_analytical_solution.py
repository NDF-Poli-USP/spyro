import numpy as np
import pytest
import spyro

from spyro.io.basicio import parallel_print as pprint
from spyro.tools.error_measure import MeasureError
from spyro.utils.analytical_solution_nodal import (
    analytical_explosive_source,
    analytical_force_source_3d,
    analytical_solution_elastic,
)


@pytest.mark.parametrize("use_vertex_only_mesh", [False, True])
def test_analytical_solution(use_vertex_only_mesh):
    frequency = 5.0
    offset = 0.5
    c_value = 1.5
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "status": False,
        "abc_type": None,
        "exponent": None,
        "cmax": None,
        "R": None,
        "pad_length": None,
    }
    dictionary["mesh"] = {
        "length_z": 3.0,  # depth in km - always positive
        "length_x": 3.0,  # width in km - always positive
    }
    dictionary["acquisition"] = {
        "delay_type": "time",
        "frequency": frequency,
        "delay": c_value / frequency,
        "source_locations": [(-1.5, 1.5)],
        "receiver_locations": [(-1.5 - offset, 1.5)],
        "use_vertex_only_mesh": use_vertex_only_mesh,
    }
    wave = spyro.examples.Rectangle_acoustic(
        dictionary=dictionary, periodic=True
    )
    wave.set_initial_velocity_model(constant=c_value)
    analytical_p = spyro.utils.nodal_homogeneous_analytical(
        wave, offset, c_value
    )

    wave.forward_solve()
    numerical_p = wave.forward_solution_receivers
    numerical_p = numerical_p.flatten()

    # Computing errors
    peak_error = MeasureError.calculate_peak_error(numerical_p, analytical_p)[0]
    integral_error = MeasureError.calculate_integral_error(numerical_p, analytical_p, wave.dt)
    normalized_l2_error = MeasureError.calculate_normalized_L2_error(numerical_p, analytical_p)

    vom_label = "VOM" if use_vertex_only_mesh else "NO VOM"
    pprint(f"Normalized L2 Error ({vom_label}) = {normalized_l2_error:.4e}", comm=wave.comm)
    pprint(f"Integral Error ({vom_label}) = {integral_error:.4e}", comm=wave.comm)
    pprint(f"Peak Error ({vom_label}) = {peak_error:.4e}", comm=wave.comm)

    assert normalized_l2_error < 1e-3 and integral_error < 1e-3 and peak_error < 1e-3


def test_analytical_force_source_3d_offset_symmetry():
    """Tests analytical force source.
    
    It should have the same result with the same abs distance from
    the source.
    """
    time_vector = np.linspace(0.0, 3.0, 101)

    common_parameters = dict(
        time_vector=time_vector,
        p_wave_velocity=1.5,
        s_wave_velocity=1.0,
        density=0.1,
        amplitude=1.0,
        frequency=5.0,
        time_delay=0.2,
        force_direction=0,
        displacement_direction=0,
    )

    result_positive = analytical_force_source_3d(
        offsets=np.array([1.0, 0.5, 0.3]),
        **common_parameters,
    )
    assert np.max(np.abs(result_positive)) > 1e-12

    result_negative = analytical_force_source_3d(
        offsets=np.array([-1.0, -0.5, -0.3]),
        **common_parameters,
    )

    np.testing.assert_allclose(
        result_negative,
        result_positive,
    )


def test_analytical_explosive_source_direction():
    """Tests if explosive source is zero outsie of the main direction."""
    offsets = np.array([1.0, 0.0, 0.0])
    time_vector = np.linspace(0.0, 2.0, 201)

    common_parameters = dict(
        offsets=offsets,
        time_vector=time_vector,
        p_wave_velocity=1.0,
        density=1.0,
        amplitude=1.0,
        frequency=5.0,
        time_delay=0.2,
    )

    ux = analytical_explosive_source(
        displacement_direction=0,
        **common_parameters,
    )

    uy = analytical_explosive_source(
        displacement_direction=1,
        **common_parameters,
    )

    uz = analytical_explosive_source(
        displacement_direction=2,
        **common_parameters,
    )

    assert np.any(np.abs(ux) > 1e-5)

    np.testing.assert_allclose(uy, 0.0)
    np.testing.assert_allclose(uz, 0.0)


if __name__ == "__main__":
    test_analytical_solution(False)
    test_analytical_solution(True)
    test_analytical_force_source_3d_offset_symmetry()
    test_analytical_explosive_source_direction()
