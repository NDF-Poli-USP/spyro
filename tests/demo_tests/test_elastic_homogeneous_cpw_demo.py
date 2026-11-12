"""Run the elastic homogeneous cells-per-wavelength demo on a small setup."""

import pytest


@pytest.mark.slow
def test_elastic_homogeneous_cpw_demo(tmp_path, monkeypatch) -> None:
    """The demo search stops at the first cpw that meets the threshold."""
    from demos.elastic_homogeneous_cpw import run

    # The working directory holds no reference, so the demo must find the
    # stored one itself. It saves its search history there.
    monkeypatch.chdir(tmp_path)
    threshold = 0.02
    calculator = run(
        method="spectral_quadrilateral",
        degree=4,
        accepted_error_threshold=threshold,
        cpw_initial=0.8,
        cpw_max=1.6,
        load_reference=True,
    )

    errors = {e.cpw: e.error for e in calculator.search_history}
    passing = [cpw for cpw, error in errors.items() if error <= threshold]
    assert passing
    minimum = min(passing)
    assert all(error > threshold for cpw, error in errors.items() if cpw < minimum)
    assert (tmp_path / "spectral_quadrilateral_p4_cpw_results.txt").is_file()
