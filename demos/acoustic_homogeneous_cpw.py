"""Demo: minimum cells-per-wavelength for the 2D homogeneous acoustic case.

The calculator compares the receiver data of each mesh against the
analytical solution of a Ricker point source in a homogeneous medium, and
searches for the coarsest mesh, in cells-per-wavelength, whose error stays
below a threshold. The threshold is the error of a degree 4 spectral
quadrilateral mesh at 2 cells-per-wavelength, so the result is the
resolution at which the evaluated element matches that accuracy.

The domain is a periodic box, sized so that no wave from a periodic image
of the source reaches a receiver before the final time. The search starts
at ``C_initial``, which must fail the threshold, and steps up until the
error passes it, then refines the result to ``C_accuracy``.

Run it with::

    python3 demos/acoustic_homogeneous_cpw.py

It reads the analytical reference from :data:`REFERENCE_FILE`, so it runs
from any working directory, and saves the search history to
``p<degree>_cpw_results.txt``. A full search takes several minutes per
evaluated mesh.
"""

from pathlib import Path

import spyro
from spyro.io.dictionaryio import Method
from spyro.tools.cells_per_wavelength_calculator import (
    MAX_CPW,
    ThresholdDiscretization,
)

#: Peak frequency of the Ricker source, in Hz.
FREQUENCY = 5.0
#: Acoustic velocity of the medium, in km/s.
VELOCITY = 1.5
#: Stored analytical receiver data of this demo's acquisition.
REFERENCE_FILE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "inputfiles"
    / "acoustic_homogeneous_cpw_reference.npz"
)


def run(
    method: Method | str = Method.MASS_LUMPED_TRIANGLE,
    degree: int = 4,
    accepted_error_threshold: float | None = None,
    threshold_discretization: ThresholdDiscretization | bool | dict | None = None,
    cpw_initial: float = 0.8,
    cpw_max: float = MAX_CPW,
    load_reference: bool = True,
) -> spyro.tools.MeshingParameterCalculator:
    """Search for the minimum cells-per-wavelength of one discretization.

    Parameters
    ----------
    method : Method or str, optional
        Finite element method to evaluate, `Method.MASS_LUMPED_TRIANGLE`
        or `Method.SPECTRAL_QUADRILATERAL`, or any of their string aliases.
        Default is `Method.MASS_LUMPED_TRIANGLE`.
    degree : int, optional
        Polynomial degree to evaluate. Default is 5.
    accepted_error_threshold : float or None, optional
        Largest accepted receiver error. Default is None.
    threshold_discretization : ThresholdDiscretization, bool, dict or None, optional
        Discretization whose error is used as the threshold. Default is
        None, which, when `accepted_error_threshold` is also None, uses a
        degree 4 spectral quadrilateral mesh at cpw 2.0. Giving both is
        rejected by the calculator.
    cpw_initial : float, optional
        Starting cells-per-wavelength, whose error must exceed the
        threshold. Default is 0.8.
    cpw_max : float, optional
        Largest cells-per-wavelength evaluated. Default is
        :data:`~spyro.tools.cells_per_wavelength_calculator.MAX_CPW`.
    load_reference : bool, optional
        Load the reference receiver data from :data:`REFERENCE_FILE`. If
        False, it is computed and saved to
        `acoustic_homogeneous_cpw_reference.npz` in the working
        directory. Default is True.

    Returns
    -------
    spyro.tools.MeshingParameterCalculator
        The calculator after the search, holding its
        `search_history`.
    """
    method = Method(method)
    threshold = {}
    if accepted_error_threshold is not None:
        threshold["accepted_error_threshold"] = accepted_error_threshold
    if threshold_discretization is not None:
        threshold["threshold_discretization"] = threshold_discretization
    if not threshold:
        # Error of a degree 4 spectral quadrilateral mesh at cpw 2.0
        threshold["threshold_discretization"] = True

    parameters = {
        "source_frequency": FREQUENCY,
        "mesh_frequency_factor": 2.0,
        "minimum_velocity_in_the_domain": VELOCITY,
        "velocity_profile_type": "homogeneous",
        "FEM_method_to_evaluate": method.value,
        "dimension": 2,
        "receiver_setup": "near",
        "timestep_calculation": "estimate",
        "desired_degree": degree,
        "C_initial": cpw_initial,
        "C_max": cpw_max,
        "C_accuracy": 0.1,
        "load_reference": load_reference,
        "save_reference": not load_reference,
        **threshold,
    }
    if load_reference:
        # A computed reference goes to the working directory instead, so the
        # stored file is never overwritten
        parameters["reference_solution_file"] = str(REFERENCE_FILE)

    calculator = spyro.tools.MeshingParameterCalculator(parameters)
    cpw = calculator.find_minimum(savetxt=True)

    for evaluation in calculator.search_history:
        print(
            f"cpw = {evaluation.cpw:.2f}  |  dt = {evaluation.dt:.3e}  |  "
            f"error = {evaluation.error:.4e}  |  "
            f"runtime = {evaluation.runtime:.1f} s",
            flush=True,
        )
    print(
        f"Minimum cells-per-wavelength for {method.value} P{degree}: {cpw} "
        f"(error threshold {calculator.accepted_error_threshold:.4e})",
        flush=True,
    )
    return calculator


if __name__ == "__main__":
    run()
