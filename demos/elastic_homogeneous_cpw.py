"""Demo: minimum cells-per-wavelength for the 2D homogeneous isotropic elastic case.

The calculator compares the receiver displacement of each mesh against the
analytical solution of a horizontal Ricker point force in a homogeneous
isotropic elastic medium, and searches for the coarsest mesh, in
cells-per-wavelength, whose error stays below a threshold. The wavelength
is the S-wavelength, the shortest in the medium. The threshold is the error
of a degree 4 spectral quadrilateral mesh at 2 cells-per-wavelength, so the
result is the resolution at which the evaluated element matches that
accuracy.

Material and acquisition follow Lyu et al. (2024), doi:
10.1029/2023JB027576: the receivers lie on a horizontal line 7
S-wavelengths below the source, with horizontal offsets from 0 to 7
S-wavelengths. The domain is a periodic box, sized so that no P-wave from a
periodic image of the source reaches a receiver before the final time. The
search starts at ``C_initial``, which must fail the threshold, and steps up
until the error passes it, then refines the result to ``C_accuracy``.

Run it with::

    python3 demos/elastic_homogeneous_cpw.py

It saves the search history to ``p<degree>_cpw_results.txt``. A full
search takes several minutes per evaluated mesh.
"""

import spyro
from spyro.tools.cells_per_wavelength_calculator import (
    MAX_CPW,
    ThresholdDiscretization,
)
from spyro.utils.typing import WaveType

#: Peak frequency of the Ricker source, in Hz.
FREQUENCY = 10.0
#: P-wave velocity of the medium, in km/s.
P_WAVE_VELOCITY = 5.6
#: S-wave velocity of the medium, in km/s.
S_WAVE_VELOCITY = 2.75
#: Density of the medium, in g/cm^3.
DENSITY = 2.0


def run(
    method: str = "mass_lumped_triangle",
    degree: int = 4,
    accepted_error_threshold: float | None = None,
    threshold_discretization: ThresholdDiscretization | bool | dict | None = None,
    cpw_initial: float = 1.0,
    cpw_max: float = MAX_CPW,
) -> spyro.tools.MeshingParameterCalculator:
    """Search for the minimum cells-per-wavelength of one discretization.

    Parameters
    ----------
    method : str, optional
        Finite element method to evaluate, ``"mass_lumped_triangle"`` or
        ``"spectral_quadrilateral"``. Default is ``"mass_lumped_triangle"``.
    degree : int, optional
        Polynomial degree to evaluate. Default is 4.
    accepted_error_threshold : float or None, optional
        Largest accepted receiver error. Default is None.
    threshold_discretization : ThresholdDiscretization, bool, dict or None, optional
        Discretization whose error is used as the threshold. Default is
        None, which, when ``accepted_error_threshold`` is also None, uses a
        degree 4 spectral quadrilateral mesh at cpw 2.0. Giving both is
        rejected by the calculator.
    cpw_initial : float, optional
        Starting cells-per-wavelength, whose error must exceed the
        threshold. Default is 1.0.
    cpw_max : float, optional
        Largest cells-per-wavelength evaluated. Default is
        :data:`~spyro.tools.cells_per_wavelength_calculator.MAX_CPW`.

    Returns
    -------
    spyro.tools.MeshingParameterCalculator
        The calculator after the search, holding its
        ``search_history``.
    """
    threshold = {}
    if accepted_error_threshold is not None:
        threshold["accepted_error_threshold"] = accepted_error_threshold
    if threshold_discretization is not None:
        threshold["threshold_discretization"] = threshold_discretization
    if not threshold:
        # Error of a degree 4 spectral quadrilateral mesh at cpw 2.0
        threshold["threshold_discretization"] = True

    parameters = {
        "wave_type": WaveType.ISOTROPIC_ELASTIC,
        "source_frequency": FREQUENCY,
        "mesh_frequency_factor": 2.0,
        "p_wave_velocity": P_WAVE_VELOCITY,
        "s_wave_velocity": S_WAVE_VELOCITY,
        "density": DENSITY,
        "velocity_profile_type": "homogeneous",
        "FEM_method_to_evaluate": method,
        "dimension": 2,
        "timestep_calculation": "estimate",
        "desired_degree": degree,
        "C_initial": cpw_initial,
        "C_max": cpw_max,
        "C_accuracy": 0.1,
        "save_reference": True,
        **threshold,
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
