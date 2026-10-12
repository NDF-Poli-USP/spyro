"""Demo: minimum cells-per-wavelength for the 2D isotropic elastic Marmousi2 case.

The calculator compares the receiver displacement of each mesh against a
numerical reference on a finer, higher degree mesh, and searches for the
coarsest mesh, in cells-per-wavelength, whose error stays below a
threshold. The threshold is the error of a degree 4 spectral quadrilateral
mesh at 2 cells-per-wavelength, so the result is the resolution at which
the evaluated element matches that accuracy. The wavelength is that of the
slowest wave in the model: the slowest S-wave in the rock, or the P-wave in
the water, which has no S-wave.

The model is the 17 km by 3.5 km Marmousi2 of Martin, Wiley and Marfurt
(2006), doi: 10.1190/1.2172306, with its 0.45 km water layer as an elastic
material of zero S-wave velocity. The acquisition is the towed-streamer
shot of the acoustic Marmousi2 demo, a source 6 m below the sea surface
and a streamer 8 m deep trailing it, with receiver groups every 12.5 m at
offsets from 2 km to 10 km. The source is the horizontal point force of the
elastic calculator. Every boundary has a first-order Stacey absorbing
condition.

The P-wave velocity, S-wave velocity and density SEG-Ys are not
distributed with spyro. Save them as
``velocity_models/vp_marmousi-ii.segy``,
``velocity_models/vs_marmousi-ii.segy`` and
``velocity_models/density_marmousi-ii.segy``, see
:class:`~spyro.tools.cells_per_wavelength_calculator.ElasticMarmousi2Case`,
or give other files to :func:`run`. Then run, from the repository root::

    python3 demos/elastic_marmousi2_cpw.py

The material is resampled to ``elastic_marmousi2_vp.segy``,
``elastic_marmousi2_vs.segy`` and ``elastic_marmousi2_rho.segy`` in the
working directory. The reference is saved to
``elastic_marmousi2_cpw_reference.npy``, and ``run(load_reference=True)``
reuses it. The search history is saved to ``p<degree>_cpw_results.txt``.
The mesh is sized by the shortest wavelength in the model, so the
reference and every evaluated mesh take hours in serial; run it in
parallel with ``mpiexec -n N_CORES``.
"""

import spyro
from spyro.tools.cells_per_wavelength_calculator import (
    MARMOUSI2_DENSITY_FILE,
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
    MARMOUSI2_S_WAVE_VELOCITY_FILE,
    MAX_CPW,
    ElasticMarmousi2Case,
    ThresholdDiscretization,
)
from spyro.utils.typing import WaveType

#: Peak frequency of the Ricker source, in Hz.
FREQUENCY = 5.0


def run(
    method: str = "mass_lumped_triangle",
    degree: int = 4,
    accepted_error_threshold: float | None = None,
    threshold_discretization: ThresholdDiscretization | bool | dict | None = None,
    cpw_initial: float = 1.0,
    cpw_max: float = MAX_CPW,
    reference_degree: int = 5,
    cpw_reference: float = 3.0,
    load_reference: bool = False,
    p_wave_velocity_file: str = MARMOUSI2_P_WAVE_VELOCITY_FILE,
    s_wave_velocity_file: str = MARMOUSI2_S_WAVE_VELOCITY_FILE,
    density_file: str = MARMOUSI2_DENSITY_FILE,
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
    reference_degree : int, optional
        Polynomial degree of the numerical reference. Default is 5.
    cpw_reference : float, optional
        Cells-per-wavelength of the numerical reference. Default is 3.0,
        15 grid points per wavelength at degree 5.
    load_reference : bool, optional
        If True, load the reference saved by an earlier run instead of
        solving it. Default is False.
    p_wave_velocity_file : str, optional
        Marmousi2 P-wave velocity SEG-Y, in m/s or km/s. Default is
        :data:`~spyro.tools.cells_per_wavelength_calculator.MARMOUSI2_P_WAVE_VELOCITY_FILE`.
    s_wave_velocity_file : str, optional
        Marmousi2 S-wave velocity SEG-Y, in m/s or km/s. Default is
        :data:`~spyro.tools.cells_per_wavelength_calculator.MARMOUSI2_S_WAVE_VELOCITY_FILE`.
    density_file : str, optional
        Marmousi2 density SEG-Y, in kg/m^3 or g/cm^3. Default is
        :data:`~spyro.tools.cells_per_wavelength_calculator.MARMOUSI2_DENSITY_FILE`.

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

    case = ElasticMarmousi2Case(
        segy_files={
            "p_wave_velocity": p_wave_velocity_file,
            "s_wave_velocity": s_wave_velocity_file,
            "density": density_file,
        }
    )
    parameters = {
        "wave_type": WaveType.ISOTROPIC_ELASTIC,
        "source_frequency": FREQUENCY,
        "mesh_frequency_factor": 2.0,
        "heterogeneous_case": case,
        "FEM_method_to_evaluate": method,
        "dimension": 2,
        "timestep_calculation": "estimate",
        "desired_degree": degree,
        "reference_degree": reference_degree,
        "C_reference": cpw_reference,
        "C_initial": cpw_initial,
        "C_max": cpw_max,
        "C_accuracy": 0.1,
        "save_reference": not load_reference,
        "load_reference": load_reference,
        "reference_solution_file": f"{case.name}_cpw_reference.npy",
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
        f"Minimum cells-per-wavelength for {method} P{degree} on elastic "
        f"Marmousi2: {cpw} "
        f"(error threshold {calculator.accepted_error_threshold:.4e})",
        flush=True,
    )
    return calculator


if __name__ == "__main__":
    run()
