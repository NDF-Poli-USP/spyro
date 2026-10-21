"""Run the acoustic Marmousi2 cells-per-wavelength demo on a small setup."""

import os

import pytest

from spyro.tools.cells_per_wavelength_calculator import (
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.isfile(MARMOUSI2_P_WAVE_VELOCITY_FILE),
    reason=f"{MARMOUSI2_P_WAVE_VELOCITY_FILE} is not distributed with spyro",
)
def test_acoustic_marmousi2_cpw_demo(tmp_path, monkeypatch) -> None:
    """The demo search stops at the first cpw that meets the threshold."""
    from demos.acoustic_marmousi2_cpw import run

    velocity_file = os.path.abspath(MARMOUSI2_P_WAVE_VELOCITY_FILE)
    # The demo saves its reference and search history to the working directory
    monkeypatch.chdir(tmp_path)
    threshold = 0.05
    calculator = run(
        method="spectral_quadrilateral",
        degree=4,
        accepted_error_threshold=threshold,
        cpw_initial=1.5,
        cpw_max=2.5,
        reference_degree=4,
        cpw_reference=3.0,
        velocity_file=velocity_file,
    )

    errors = {e.cpw: e.error for e in calculator.search_history}
    passing = [cpw for cpw, error in errors.items() if error <= threshold]
    assert passing
    minimum = min(passing)
    assert all(error > threshold for cpw, error in errors.items() if cpw < minimum)
    assert (tmp_path / "p4_cpw_results.txt").is_file()
    assert (tmp_path / "acoustic_marmousi2_cpw_reference.npy").is_file()
