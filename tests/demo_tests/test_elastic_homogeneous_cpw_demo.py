"""Run the elastic homogeneous cells-per-wavelength demo on a small setup."""

import pytest


@pytest.mark.slow
def test_elastic_homogeneous_cpw_demo(tmp_path, monkeypatch) -> None:
    """The demo search stops at the first cpw that meets the threshold."""
    from demos.elastic_homogeneous_cpw import run

    # The demo saves its search history to the working directory
    monkeypatch.chdir(tmp_path)
    threshold = 0.05
    calculator = run(
        method="spectral_quadrilateral",
        degree=4,
        accepted_error_threshold=threshold,
        cpw_initial=1.5,
        cpw_max=3.0,
    )

    errors = {e.cpw: e.error for e in calculator.search_history}
    passing = [cpw for cpw, error in errors.items() if error <= threshold]
    assert passing
    minimum = min(passing)
    assert all(error > threshold for cpw, error in errors.items() if cpw < minimum)
    assert (tmp_path / "p4_cpw_results.txt").is_file()
