"""Run the acoustic homogeneous cells-per-wavelength demo on a small setup."""

import numpy as np
import pytest


@pytest.mark.slow
def test_acoustic_homogeneous_cpw_demo(tmp_path, monkeypatch) -> None:
    """The demo search runs and finds the known minimum of a reduced case."""
    from demos.acoustic_homogeneous_cpw import run

    # The demo saves its search history to the working directory
    monkeypatch.chdir(tmp_path)
    calculator = run(
        method="spectral_quadrilateral",
        degree=4,
        accepted_error_threshold=0.01,
        cpw_initial=2.5,
        cpw_max=2.5,
        fixed_timestep=0.002,
    )

    errors = {e.cpw: e.error for e in calculator.search_history}
    cpw = min(cpw for cpw, error in errors.items() if error <= 0.01)
    assert np.isclose(cpw, 2.1)
    assert (tmp_path / "p4_cpw_results.txt").is_file()
