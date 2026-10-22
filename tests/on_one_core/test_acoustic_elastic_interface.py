"""Reflection of a plane wave at the fluid-solid interface.

A plane pressure pulse in the fluid hits the interface at normal incidence.
The pressure it reflects is the incident one times

.. math::

    R = \\frac{\\rho_s v_p - \\rho_f c_f}{\\rho_s v_p + \\rho_f c_f},

so the test checks the impedance of both media enters the interface
condition, the fluid density included.
"""

import firedrake as fire
import numpy as np
import pytest

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave


pytestmark = pytest.mark.newer_firedrake

DENSITY_FLUID = 2.0
VELOCITY_FLUID = 1.5
DENSITY_SOLID = 2.5
S_WAVE_VELOCITY = 1.5
# A vanishing first Lame parameter leaves no lateral stress in a plane P
# wave, so the traction-free sides of the narrow domain do not disturb it.
P_WAVE_VELOCITY = np.sqrt(2.0) * S_WAVE_VELOCITY
PULSE_CENTER = -0.2
PULSE_WIDTH = 0.04
RECEIVER_Z = -0.3


def make_dictionary() -> dict:
    """Build a narrow fluid-over-solid column for a plane wave.

    The fluid fills ``-0.5 < z < 0`` and the solid ``-1 < z < -0.5``. The
    point source is switched off: the wave comes from the initial pressure.

    Returns
    -------
    dict
        Spyro model dictionary.
    """
    return {
        "options": {
            "cell_type": "Q",
            "variant": "lumped",
            "degree": 2,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 0.1,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
            "edge_length": 0.02,
            "interface_z": -0.5,
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(PULSE_CENTER, 0.05)],
            "frequency": 5.0,
            "delay": 0.2,
            "delay_type": "time",
            "amplitude": 0.0,
            "receiver_locations": [(RECEIVER_Z, 0.05)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.45,
            "dt": 0.0005,
            "output_frequency": 100000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
        "synthetic_data": {
            "type": "object",
            "velocity_fluid": VELOCITY_FLUID,
            "density_fluid": DENSITY_FLUID,
            "density_solid": DENSITY_SOLID,
            "p_wave_velocity": P_WAVE_VELOCITY,
            "s_wave_velocity": S_WAVE_VELOCITY,
            "real_velocity_file": None,
        },
    }


@pytest.mark.parametrize("use_monolithic", [False, True],
                         ids=["sequential", "monolithic"])
def test_normal_incidence_reflection_coefficient(use_monolithic: bool):
    """The reflected pressure matches the impedance contrast.

    Parameters
    ----------
    use_monolithic : bool
        Whether to solve the coupled step monolithically or sequentially.
    """
    wave = AcousticElasticWave(make_dictionary())
    wave.use_monolithic = use_monolithic
    wave.initialize_physical_parameters()
    wave.matrix_building()

    # A Gaussian pulse travelling down, towards the interface: at the
    # previous time step its centre was one step higher up.
    z = fire.SpatialCoordinate(wave.submesh_fluid)[0]
    previous_center = PULSE_CENTER + VELOCITY_FLUID * wave.dt
    wave.X_n.sub(0).interpolate(fire.exp(-((z - PULSE_CENTER) / PULSE_WIDTH)**2))
    wave.X_nm1.sub(0).interpolate(
        fire.exp(-((z - previous_center) / PULSE_WIDTH)**2),
    )
    wave.wave_propagator()

    pressure = np.asarray(wave.forward_solution_receivers)[:, 0]
    time = np.arange(len(pressure)) * wave.dt
    # The incident pulse passes the receiver around t = 0.07 and its
    # reflection around t = 0.33; the wave transmitted into the solid only
    # comes back after both.
    incident = pressure[time < 0.2]
    reflected = pressure[(time > 0.25) & (time < 0.42)]
    measured = (
        reflected[np.argmax(np.abs(reflected))]
        / incident[np.argmax(np.abs(incident))]
    )

    impedance_fluid = DENSITY_FLUID * VELOCITY_FLUID
    impedance_solid = DENSITY_SOLID * P_WAVE_VELOCITY
    expected = (
        (impedance_solid - impedance_fluid)
        / (impedance_solid + impedance_fluid)
    )
    assert measured == pytest.approx(expected, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__])
