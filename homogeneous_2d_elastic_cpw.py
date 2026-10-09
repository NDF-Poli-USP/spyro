"""Find the minimum cells-per-wavelength for the 2D homogeneous isotropic
elastic case.

Material and acquisition follow Lyu et al. (2024), doi: 10.1029/2023JB027576.
"""
import spyro
from spyro.utils.typing import WaveType

degree = 4

parameters = {
    "wave_type": WaveType.ISOTROPIC_ELASTIC,
    "source_frequency": 10.0,
    "p_wave_velocity": 5.6,
    "s_wave_velocity": 2.75,
    "density": 2.0,
    "velocity_profile_type": "homogeneous",
    "velocity_model_file_name": None,
    "FEM_method_to_evaluate": "mass_lumped_triangle",
    "dimension": 2,
    "receiver_setup": "line",
    "testing": False,
    "time-step_calculation": "estimate",
    "reference_degree": None,
    "C_reference": None,
    "desired_degree": degree,
    "C_initial": 2.0,
    "threshold_case": True,
    "C_accuracy": 0.1,
    "load_reference": True,
}

calculator = spyro.tools.MeshingParameterCalculator(parameters)
cpw = calculator.find_minimum(savetxt=True)

for evaluation in calculator.search_history:
    print(
        f"cpw = {evaluation.cpw:.2f}  |  dt = {evaluation.dt:.3e}  |  "
        f"error = {evaluation.error:.4e}  |  runtime = {evaluation.runtime:.1f} s"
    )
print(f"Minimum cells-per-wavelength for P{degree}: {cpw}")
