"""FWI over latent controls.

``run_fwi(latent=True)`` optimizes over latent controls, inside the bounds by
construction. The test inverts the constant acoustic model of
``test_fwi_automated_adjoint`` and checks that the run stays within the bounds
and lowers the misfit.
"""
import numpy as np
import pytest

import spyro
from spyro.utils.typing import AdjointType

from .test_fwi_automated_adjoint import (
    ACOUSTIC_GUESS,
    ACOUSTIC_REAL,
    build_dictionary,
)

VMIN, VMAX = 2.0, 3.5


def acoustic_fwi() -> spyro.FullWaveformInversion:
    """Return the acoustic inversion of ``test_fwi_automated_adjoint``.

    Returns
    -------
    spyro.FullWaveformInversion
        Driver with observed data and the constant starting model.
    """
    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(ACOUSTIC_REAL)
    fwi.generate_real_shot_record(save_shot_record=False)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=ACOUSTIC_GUESS)
    return fwi


@pytest.mark.newer_firedrake
def test_latent_run(tmp_path, monkeypatch):
    """The model stays within the bounds and the misfit goes down."""
    monkeypatch.chdir(tmp_path)
    fwi = acoustic_fwi()
    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=VMIN, vmax=VMAX, maxiter=2, save_controls=False, latent=True,
    )

    values = result.dat.data_ro
    assert values.min() >= VMIN and values.max() <= VMAX
    assert not np.allclose(values, ACOUSTIC_GUESS)
    assert fwi.functional_history[-1] < fwi.functional_history[0]


def test_latent_needs_the_automated_adjoint():
    """The implemented adjoint's scipy path has no latent option."""
    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())
    with pytest.raises(ValueError, match="automated adjoint"):
        fwi.run_fwi(adjoint_type=AdjointType.IMPLEMENTED_ADJOINT, latent=True)
