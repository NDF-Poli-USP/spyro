"""Run the acoustic homogeneous cells-per-wavelength demo on a small setup."""

import shutil
from pathlib import Path

import numpy as np
import pytest

#: Stored analytical receiver data of the demo acquisition.
REFERENCE_FILE = (
    Path(__file__).parents[1] / "inputfiles" / "acoustic_homogeneous_cpw_reference.npy"
)


@pytest.mark.slow
def test_acoustic_homogeneous_cpw_demo(tmp_path, monkeypatch) -> None:
    """The demo search runs and finds the known minimum of a reduced case."""
    from demos.acoustic_homogeneous_cpw import run

    # The demo reads its reference from, and saves its search history to,
    # the working directory
    shutil.copy(REFERENCE_FILE, tmp_path)
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
    assert (tmp_path / "p4_cpw_results.txt").is_file()
