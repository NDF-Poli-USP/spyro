"""Run the elastic homogeneous cells-per-wavelength demo on a small setup."""

import shutil
from pathlib import Path

import pytest

#: Stored analytical receiver data of the demo acquisition.
REFERENCE_FILE = (
    Path(__file__).parents[1] / "inputfiles" / "elastic_homogeneous_cpw_reference.npy"
)


@pytest.mark.slow
def test_elastic_homogeneous_cpw_demo(tmp_path, monkeypatch) -> None:
    """The demo search stops at the first cpw that meets the threshold."""
    from demos.elastic_homogeneous_cpw import run

    # The demo reads its reference from, and saves its search history to,
    # the working directory
    shutil.copy(REFERENCE_FILE, tmp_path)
    monkeypatch.chdir(tmp_path)
    threshold = 0.02
    calculator = run(
        method="spectral_quadrilateral",
        degree=4,
        accepted_error_threshold=threshold,
        cpw_initial=1.0,
        cpw_max=1.6,
        load_reference=True,
    )

    errors = {e.cpw: e.error for e in calculator.search_history}
    passing = [cpw for cpw, error in errors.items() if error <= threshold]
    assert passing
    minimum = min(passing)
    assert all(error > threshold for cpw, error in errors.items() if cpw < minimum)
    assert (tmp_path / "p4_cpw_results.txt").is_file()
