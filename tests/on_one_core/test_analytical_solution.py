from pytest import mark
from firedrake import COMM_WORLD as comm
from numpy import linspace
import spyro
from spyro.io.basicio import parallel_print as pprint
from spyro.tools.error_measure import MeasureError


@mark.parametrize("use_vertex_only_mesh", [False, True])
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
    errPk = MeasureError().peak_error(numerical_p, analytical_p)[0]
    errIt = MeasureError().integral_error(numerical_p, analytical_p, wave.dt)
    eNRMS = MeasureError().normalized_root_mean_square_error(numerical_p, analytical_p)

    vom_label = "VOM" if use_vertex_only_mesh else "NO VOM"
    pprint(f"NRMS Error ({vom_label}) = {eNRMS:.4e}", comm=comm)
    pprint(f"Integral Error ({vom_label}) = {errIt:.4e}", comm=comm)
    pprint(f"Peak Error ({vom_label}) = {errPk:.4e}", comm=comm)

    assert eNRMS < 1e-3 and errIt < 1e-3 and errPk < 1e-3, \
        "Error is too high for analytical solution test."


if __name__ == "__main__":
    test_analytical_solution(False)
    test_analytical_solution(True)
