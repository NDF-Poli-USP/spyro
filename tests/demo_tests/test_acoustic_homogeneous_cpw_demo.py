"""Run the acoustic homogeneous cells-per-wavelength demo on a small setup."""

import numpy as np
import pytest


@pytest.mark.slow
def test_acoustic_homogeneous_cpw_demo(tmp_path, monkeypatch) -> None:
    """The demo search runs and finds the known minimum of a reduced case."""
    from demos.acoustic_homogeneous_cpw import run

    # The working directory holds no reference, so the demo must find the
    # stored one itself. It saves its search history there.
    monkeypatch.chdir(tmp_path)
    threshold = 0.002
    calculator = run(
        method="spectral_quadrilateral",
        degree=4,
        accepted_error_threshold=threshold,
        cpw_initial=1.5,
        cpw_max=2.5,
        load_reference=True,
    )

    errors = {e.cpw: e.error for e in calculator.search_history}
    cpw = min(cpw for cpw, error in errors.items() if error <= threshold)
    assert np.isclose(cpw, 1.9)
    assert (tmp_path / "spectral_quadrilateral_p4_cpw_results.txt").is_file()
