"""Demo: minimum cells-per-wavelength for the 2D homogeneous acoustic case.

The calculator compares the receiver data of each mesh against the
analytical solution of a Ricker point source in a homogeneous medium, and
searches for the coarsest mesh, in cells-per-wavelength, whose error stays
below a threshold. The threshold is the error of a degree 4 spectral
quadrilateral mesh at 2 cells-per-wavelength, so the result is the
resolution at which the evaluated element matches that accuracy.

The domain is a periodic box, sized so that no wave from a periodic image
of the source reaches a receiver before the final time. The search starts
at ``C_initial`` and steps down until the error passes the threshold, then
refines the result to ``C_accuracy``.

Run it with::

    python3 demos/acoustic_homogeneous_cpw.py

It saves the search history to ``p<degree>_cpw_results.txt``. A full
search takes several minutes per evaluated mesh.
"""

import spyro

#: Peak frequency of the Ricker source, in Hz.
FREQUENCY = 5.0
#: Acoustic velocity of the medium, in km/s.
VELOCITY = 1.5


def run(
    method: str = "mass_lumped_triangle",
    degree: int = 4,
) -> spyro.tools.MeshingParameterCalculator:
    """Search for the minimum cells-per-wavelength of one discretization.

    Parameters
    ----------
    method : str, optional
        Finite element method to evaluate, ``"mass_lumped_triangle"`` or
        ``"spectral_quadrilateral"``. Default is ``"mass_lumped_triangle"``.
    degree : int, optional
        Polynomial degree to evaluate. Default is 4.

    Returns
    -------
    spyro.tools.MeshingParameterCalculator
        The calculator after the search, holding its
        ``search_history``.
    """
    parameters = {
        "source_frequency": FREQUENCY,
        "minimum_velocity_in_the_domain": VELOCITY,
        "velocity_profile_type": "homogeneous",
        "FEM_method_to_evaluate": method,
        "dimension": 2,
        "receiver_setup": "near",
        "time-step_calculation": "estimate",
        "reference_degree": None,
        "C_reference": None,
        "desired_degree": degree,
        # Error of a degree 4 spectral quadrilateral mesh at cpw 2.0
        "threshold_discretization": True,
        "C_initial": 4.0,
        "C_accuracy": 0.1,
    }

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
        f"Minimum cells-per-wavelength for {method} P{degree}: {cpw} "
        f"(error threshold {calculator.accepted_error_threshold:.4e})",
        flush=True,
    )
    return calculator


if __name__ == "__main__":
    run()
