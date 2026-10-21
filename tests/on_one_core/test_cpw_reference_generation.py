"""Generate the homogeneous analytical receiver data in reduced (testing) mode.

The reduced acquisitions are subsets of the full ones, with the same time
axis, so the generated data is compared against the matching receivers of
the full data stored in ``tests/inputfiles``.
"""

from pathlib import Path

import numpy as np
import pytest

import spyro

#: Directory holding the stored full receiver data.
INPUT_FILES = Path(__file__).parents[1] / "inputfiles"

#: Parameters shared by every physics, with a fixed threshold so that no
#: threshold discretization is solved.
COMMON_PARAMETERS = {
    "mesh_frequency_factor": 2.0,
    "velocity_profile_type": "homogeneous",
    "FEM_method_to_evaluate": "spectral_quadrilateral",
    "dimension": 2,
    "testing": True,
    "desired_degree": 4,
    "C_initial": 1.0,
    "C_accuracy": 0.1,
    "accepted_error_threshold": 0.01,
}


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameters, stored_file, receivers",
    [
        pytest.param(
            {"source_frequency": 5.0, "minimum_velocity_in_the_domain": 1.5},
            "acoustic_homogeneous_cpw_reference.npy",
            # Corners of the 6 by 6 grid
            [0, 5, 30, 35],
            id="acoustic",
        ),
    ],
)
def test_reduced_reference_matches_stored(
    parameters: dict, stored_file: str, receivers: list[int]
) -> None:
    """The reduced reference equals the stored data at its receivers.

    Parameters
    ----------
    parameters : dict
        Physics-specific calculator parameters.
    stored_file : str
        Name of the stored full receiver data in ``tests/inputfiles``.
    receivers : list of int
        Indices of the reduced receivers in the full acquisition.
    """
    calculator = spyro.tools.MeshingParameterCalculator(
        {**COMMON_PARAMETERS, **parameters}
    )

    stored = np.load(INPUT_FILES / stored_file)
    reference = calculator.reference_solution
    assert reference.shape[1] == len(receivers)
    assert np.allclose(reference, stored[:, receivers], rtol=0.0, atol=1e-12)
