"""Automated-adjoint gradient of the acoustic wave with non-reflecting boundaries.

``abc_type: "nrbc"`` adds the first-order absorbing term ``(1/c) du/dt v ds``
on the sides selected by ``absorb_<side>``. The term depends on the control
``c`` and on the two most recent wavefields, so it has to be recorded on the
pyadjoint tape for the gradient to be right. The other automated-adjoint
tests run either without absorbing boundaries or with the PML, so this is
where the NRBC path of the tape is verified.
"""

import numpy as np
import pytest

import firedrake as fire
import spyro
from spyro.utils.typing import AbsorbingBCsType, AdjointType


# Deliberately small: the gradient check is exact, so it does not need a
# well-resolved model - only enough time for the wavefront to cross the
# domain, reach the absorbing sides and come back to the receivers.
FINAL_TIME = 0.8
EDGE_LENGTH = 0.2
DT = 0.004

# name -> ``absorb_<side>`` flags added to the boundary condition entry
ABSORBING_SIDES = {
    # spyro's default: free surface at the top, absorbing elsewhere
    "free_top": {},
    "all_sides": {
        "absorb_top": True,
        "absorb_bottom": True,
        "absorb_right": True,
        "absorb_left": True,
    },
}


def build_dictionary(absorbing_sides: dict | None = None) -> dict:
    """Build the model dictionary shared by every test in this module.

    Parameters
    ----------
    absorbing_sides : dict, optional
        ``absorb_<side>`` flags of the non-reflecting boundary condition.
        ``None`` builds the model without absorbing boundaries.

    Returns
    -------
    dict
        A spyro model dictionary describing a small 2D acoustic problem.
    """
    dictionary = {
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
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5)],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": spyro.create_transect(
                (-0.8, 0.1), (-0.8, 0.9), 5
            ),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": FINAL_TIME,
            "dt": DT,
            "amplitude": 1,
            "output_frequency": 100000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "forward_output_filename": None,
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": None,
            "adjoint_output": False,
            "adjoint_filename": None,
            "debug_output": False,
        },
    }
    if absorbing_sides is not None:
        dictionary["absorving_boundary_conditions"] = {
            "status": True,
            "abc_type": "nrbc",
            **absorbing_sides,
        }
    return dictionary


def guess_wave(absorbing_sides: dict | None) -> spyro.AcousticWave:
    """Build the constant-velocity guess model.

    Parameters
    ----------
    absorbing_sides : dict or None
        Passed on to :func:`build_dictionary`.

    Returns
    -------
    spyro.AcousticWave
        The wave object, with mesh and velocity model set, not yet solved.
    """
    wave = spyro.AcousticWave(dictionary=build_dictionary(absorbing_sides))
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(constant=2.0)
    return wave


def observed_record(absorbing_sides: dict | None) -> np.ndarray:
    """Run the two-layer 'exact' model and return its receiver data.

    Parameters
    ----------
    absorbing_sides : dict or None
        Passed on to :func:`build_dictionary`, so the observed data is
        propagated under the same boundary condition as the guess.

    Returns
    -------
    numpy.ndarray
        Receiver time series used as the real shot record.
    """
    wave = spyro.AcousticWave(dictionary=build_dictionary(absorbing_sides))
    wave.set_mesh(input_mesh_parameters={"edge_length": EDGE_LENGTH})
    wave.set_initial_velocity_model(
        conditional=fire.conditional(wave.mesh_z > -0.5, 1.5, 3.5),
        dg_velocity_model=False,
    )
    wave.forward_solve()
    return wave.forward_solution_receivers


@pytest.fixture(scope="module")
def reflecting_guess_record() -> np.ndarray:
    """Guess-model record without absorbing boundaries.

    Module scoped: it is the reference every absorbing case is compared
    against to show the boundary term actually acted on the solution.

    Returns
    -------
    numpy.ndarray
        Receiver time series of the guess model with reflecting boundaries.
    """
    wave = guess_wave(None)
    wave.forward_solve()
    return wave.forward_solution_receivers


@pytest.mark.slow
@pytest.mark.newer_firedrake
@pytest.mark.parametrize("sides", list(ABSORBING_SIDES))
def test_taylor_test_nrbc(reflecting_guess_record, sides: str) -> None:
    """Taylor-test the gradient with the NRBC on the selected sides.

    Parameters
    ----------
    reflecting_guess_record : numpy.ndarray
        Guess-model receiver data without absorbing boundaries.
    sides : str
        Key into :data:`ABSORBING_SIDES` selecting the absorbing sides.
    """
    absorbing_sides = ABSORBING_SIDES[sides]
    wave = guess_wave(absorbing_sides)
    wave.real_shot_record = observed_record(absorbing_sides)
    wave.enable_automated_adjoint()
    try:
        wave.forward_solve()
        assert wave.abc_type is AbsorbingBCsType.NRBC

        # A Taylor test only says something about the boundary term if the
        # term changed the solution the functional is built from.
        assert not np.allclose(
            wave.forward_solution_receivers, reflecting_guess_record
        ), "the NRBC did not change the receiver data"

        dJ = wave.gradient_solve(adjoint_type=AdjointType.AUTOMATED_ADJOINT)
        assert isinstance(dJ, fire.Function)

        size, = np.shape(wave.c.dat.data_ro[:])
        direction = fire.Function(
            wave.c.function_space(),
            val=np.random.default_rng(0).random(size),
        )
        rate = wave.automated_adjoint.verify_gradient(
            wave.c, direction=direction, dJdm=dJ
        )
        assert rate > 1.9, (
            f"Taylor convergence rate {rate} with NRBC on '{sides}'"
        )
    finally:
        # Clear on the failing path too, so a failure here does not leave a
        # tape for the next test in this process to annotate on top of.
        wave.automated_adjoint.clear_tape()
