"""Run the elastic Marmousi2 forward propagation demo on a coarse mesh."""

import os

import numpy as np
import pytest

from spyro.tools.cells_per_wavelength_calculator import (
    MARMOUSI2_DENSITY_FILE,
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
    MARMOUSI2_S_WAVE_VELOCITY_FILE,
)

MARMOUSI2_FILES = (
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
    MARMOUSI2_S_WAVE_VELOCITY_FILE,
    MARMOUSI2_DENSITY_FILE,
)


@pytest.mark.skipif(
    not all(os.path.isfile(file_name) for file_name in MARMOUSI2_FILES),
    reason=f"{', '.join(MARMOUSI2_FILES)} are not distributed with spyro",
)
def test_elastic_marmousi_forward_propagation_demo(tmp_path, monkeypatch) -> None:
    """The demo reads the Marmousi2 material and propagates the wave."""
    from demos.elastic_marmousi_forward_propagation import run

    p_wave_velocity_file, s_wave_velocity_file, density_file = (
        os.path.abspath(file_name) for file_name in MARMOUSI2_FILES
    )
    # The demo writes its fields and shot record to the working directory
    monkeypatch.chdir(tmp_path)
    os.mkdir("shots")
    wave = run(
        degree=2,
        edge_length=0.25,
        final_time=0.1,
        p_wave_velocity_file=p_wave_velocity_file,
        s_wave_velocity_file=s_wave_velocity_file,
        density_file=density_file,
    )

    # S-wave velocities read with segyio straight from the Marmousi2 SEG-Y,
    # at mesh nodes that are SEG-Y samples
    for point, s_wave_velocity in [
        # Water
        ((-0.25, 8.0), 0.0),
        ((-2.0, 9.0), 1.1804295),
    ]:
        assert np.isclose(
            wave.c_s.at(point), s_wave_velocity, rtol=1e-5, atol=1e-6,
        )

    receivers = np.asarray(wave.forward_solution_receivers)
    assert receivers.shape[1:] == (100, 2)
    assert np.all(np.isfinite(receivers))
    assert np.abs(receivers).max() > 0.0
    for file_name in ("vp.pvd", "vs.pvd", "rho.pvd"):
        assert os.path.isfile(file_name), file_name
    assert len(os.listdir("shots")) == 1
