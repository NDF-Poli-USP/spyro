# from mpi4py.MPI import COMM_WORLD
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()

"""Demo: minimum cells-per-wavelength for the 2D acoustic Marmousi2 case.

The calculator compares the receiver data of each mesh against a numerical
reference on a finer, higher degree mesh, and searches for the coarsest
mesh, in cells-per-wavelength, whose error stays below a threshold. The
threshold is the error of a degree 4 spectral quadrilateral mesh at 2
cells-per-wavelength, so the result is the resolution at which the
evaluated element matches that accuracy. The wavelength is that of the
water, the slowest P-wave velocity in the model.

The triangle meshes are adapted to the velocity model with SeismicMesh: each
cell is sized for ``cpw`` cells per local P-wavelength, with a limited size
gradation between slow and fast regions. The quadrilateral meshes of the
default threshold discretization cannot be adapted and stay uniform.

The model is the 17 km by 3.5 km Marmousi2 of Martin, Wiley and Marfurt
(2006), doi: 10.1190/1.2172306, with its 0.45 km water layer. The
acquisition is a single shot of a towed-streamer survey: an air gun 6 m
below the sea surface and a streamer 8 m deep trailing it, with receiver
groups every 12.5 m at offsets from 2 km to 10 km. The domain is
surrounded by a PML one wavelength thick, except at the sea surface.

The P-wave velocity SEG-Y is not distributed with spyro. Save it as
``velocity_models/vp_marmousi-ii.segy``, see
:data:`~spyro.tools.cells_per_wavelength_calculator.MARMOUSI2_P_WAVE_VELOCITY_FILE`,
or give another file to :func:`run`. Then run, from the repository root::

    python3 demos/acoustic_marmousi2_cpw.py

The reference is saved to ``acoustic_marmousi2_cpw_reference.npy``, and
``run(load_reference=True)`` reuses it. The search history is saved to
``p<degree>_cpw_results.txt``. The model is large and the final time long,
so the reference and every evaluated mesh take hours in serial; run it in
parallel with ``mpiexec -n N_CORES``.
"""

import spyro
from spyro.tools.cells_per_wavelength_calculator import (
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
    MAX_CPW,
    AcousticMarmousi2Case,
    ThresholdDiscretization,
)

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
    velocity_file: str = MARMOUSI2_P_WAVE_VELOCITY_FILE,
    velocity_adapted_mesh: bool = True,
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
    velocity_file : str, optional
        Marmousi2 P-wave velocity SEG-Y, in m/s or km/s. Default is
        :data:`~spyro.tools.cells_per_wavelength_calculator.MARMOUSI2_P_WAVE_VELOCITY_FILE`.
    velocity_adapted_mesh : bool, optional
        If True, size the triangles by the local P-wavelength; if False,
        uniformly by the minimum one. Default is True.

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

    case = AcousticMarmousi2Case(material_files={"p_wave_velocity": velocity_file})
    parameters = {
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
        "velocity_adapted_mesh": velocity_adapted_mesh,
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
        f"Minimum cells-per-wavelength for {method} P{degree} on Marmousi2: "
        f"{cpw} (error threshold {calculator.accepted_error_threshold:.4e})",
        flush=True,
    )
    return calculator


if __name__ == "__main__":
    run()
