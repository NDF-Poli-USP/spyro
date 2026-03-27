from copy import deepcopy
import math
import numpy as np
import firedrake as fire
import spyro
import pytest
import warnings
from spyro.utils.typing import RieszMapType


warnings.filterwarnings("ignore")


def is_rol_installed():
    try:
        import ROL  # noqa:F401
        return True
    except ImportError:
        return False


def get_dof_coordinates_in_function_space(function, mesh_z, mesh_x):
    space = function.function_space()
    if isinstance(function, fire.Cofunction):
        space = space.dual()
    z_values = fire.Function(space).interpolate(mesh_z).dat.data_ro.copy()
    x_values = fire.Function(space).interpolate(mesh_x).dat.data_ro.copy()
    return z_values, x_values


final_time = 0.9
CENTER_Z = -1.0
CENTER_X = 1.0
ANOMALY_RADIUS = 0.2
MASK_BOUNDARIES = {
    "z_min": -1.3,
    "z_max": -0.7,
    "x_min": 0.7,
    "x_max": 1.3,
}
CENTER_BOX = {
    "z_min": -1.1,
    "z_max": -0.9,
    "x_min": 0.9,
    "x_max": 1.1,
}
MIN_CENTER_GRADIENT_RATIO = 0.1

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "length_z": 2.0,  # depth in km - always positive
    "length_x": 2.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 2),
    "frequency": 5.0,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.45, 0.7), (-1.45, 1.3), 200),
    "use_vertex_only_mesh": True,
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}
dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}
dictionary["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "automated_adjoint": False,
}


def get_center_dofs(function, mesh_z, mesh_x):
    dof_z, dof_x = get_dof_coordinates_in_function_space(function, mesh_z, mesh_x)
    return (
        (dof_z > CENTER_BOX["z_min"])
        & (dof_z < CENTER_BOX["z_max"])
        & (dof_x > CENTER_BOX["x_min"])
        & (dof_x < CENTER_BOX["x_max"])
    )


def build_serialshot_fwi(automated_adjoint, load_real_shot=False):
    test_dictionary = deepcopy(dictionary)
    test_dictionary["inversion"]["automated_adjoint"] = automated_adjoint

    if load_real_shot is False:
        FWI_obj = spyro.FullWaveformInversion(dictionary=test_dictionary)

        FWI_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.1})
        mesh_z = FWI_obj.mesh_z
        mesh_x = FWI_obj.mesh_x
        cond = fire.conditional(
            (mesh_z - CENTER_Z) ** 2 + (mesh_x - CENTER_X) ** 2 < ANOMALY_RADIUS ** 2,
            3.0,
            2.5,
        )

        FWI_obj.set_real_velocity_model(
            conditional=cond, output=False, dg_velocity_model=False
        )
        FWI_obj.generate_real_shot_record(
            plot_model=False,
            model_filename="True_experiment.png",
            abc_points=[(-0.5, 0.5), (-1.5, 0.5), (-1.5, 1.5), (-0.5, 1.5)],
            save_shot_record=False,
        )
    else:
        test_dictionary["inversion"]["shot_record_file"] = "real_shot_record.npy"
        FWI_obj = spyro.FullWaveformInversion(dictionary=test_dictionary)

    FWI_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
    FWI_obj.set_guess_velocity_model(constant=2.5)
    FWI_obj.set_gradient_mask(boundaries=MASK_BOUNDARIES)
    return FWI_obj


def build_single_source_vom_dictionary():
    single_source_dictionary = deepcopy(dictionary)
    single_source_dictionary["acquisition"]["source_locations"] = [
        dictionary["acquisition"]["source_locations"][0]
    ]
    return single_source_dictionary


def get_single_source_vom_real_shot_record():
    exact_wave = spyro.AcousticWave(dictionary=build_single_source_vom_dictionary())
    exact_wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    cond = fire.conditional(
        (exact_wave.mesh_z - CENTER_Z) ** 2
        + (exact_wave.mesh_x - CENTER_X) ** 2
        < ANOMALY_RADIUS ** 2,
        3.0,
        2.5,
    )
    exact_wave.set_initial_velocity_model(conditional=cond)
    exact_wave.forward_solve()
    return exact_wave.forward_solution_receivers.copy()


def get_single_source_vom_derivative(automated_adjoint, real_shot_record):
    wave = spyro.AcousticWave(dictionary=build_single_source_vom_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=2.5)
    wave.real_shot_record = real_shot_record

    if automated_adjoint:
        wave.enable_automated_adjoint()
        derivative = wave.gradient_solve(riesz_map=RieszMapType.l2)
        return wave.functional_value, derivative.dat.data_ro.copy()

    wave.enable_implemented_adjoint()
    wave.store_forward_time_steps = True
    wave.forward_solve()
    gradient = wave.gradient_solve(forward_solution=deepcopy(wave.forward_solution))
    trial = fire.TrialFunction(wave.function_space)
    test = fire.TestFunction(wave.function_space)
    derivative = fire.assemble(
        fire.action(
            fire.inner(trial, test) * fire.dx(**wave.quadrature_rule),
            gradient,
        )
    )
    return wave.functional_value, derivative.dat.data_ro.copy()


@pytest.mark.slow
@pytest.mark.parametrize("automated_adjoint", [True, False])
def test_fwi(automated_adjoint, load_real_shot=False, use_rol=False):
    """
    Run the Full Waveform Inversion (FWI) test.

    Parameters
    ----------
        load_real_shot (bool, optional): Whether to load a real shot record or not. Defaults to False.
        automated_adjoint (bool, optional): Whether to use the automated adjoint. Defaults to False.
    """
    if automated_adjoint:
        use_rol = False  # Only for scipy.

    FWI_obj = build_serialshot_fwi(
        automated_adjoint=automated_adjoint,
        load_real_shot=load_real_shot,
    )
    if use_rol:
        FWI_obj.run_fwi_rol(vmin=2.5, vmax=3.0, maxiter=2)
    else:
        FWI_obj.run_fwi(vmin=2.5, vmax=3.0, maxiter=5)

    # Verify the mask exactly where it is applied and check for non-zero
    # gradient energy inside the anomaly region.
    grad_test = FWI_obj.gradient
    masked_dofs = FWI_obj.mask_obj.mask_dofs[0]
    masked_values = grad_test.dat.data_ro[masked_dofs]
    test0 = masked_values.size > 0 and np.allclose(masked_values, 0.0)
    print(f"Masked gradient DoFs are zero: {test0}", flush=True)
    gradient_values = np.abs(grad_test.dat.data_ro)
    center_dofs = get_center_dofs(grad_test, FWI_obj.mesh_z, FWI_obj.mesh_x)
    center_values = gradient_values[center_dofs]
    center_max = np.max(center_values) if center_values.size > 0 else 0.0
    global_max = np.max(gradient_values) if gradient_values.size > 0 else 0.0
    test1 = (
        center_values.size > 0
        and global_max > 0.0
        and center_max >= MIN_CENTER_GRADIENT_RATIO * global_max
    )
    print(
        "Center box captures a meaningful share of the gradient: "
        f"{test1} (center/global={center_max / global_max if global_max else 0.0})",
        flush=True,
    )

    initial_functional = FWI_obj.functional_history[0]
    final_functional = FWI_obj.functional_history[-1]
    test2 = np.isfinite(final_functional) and final_functional >= 0.0
    print(f"Last functional finite and non-negative: {test2}", flush=True)
    test3 = final_functional <= initial_functional * (1.0 + 1e-6) + 1e-15
    print(f"Optimizer keeps the functional stable: {test3}", flush=True)

    print("END", flush=True)
    assert all([test0, test1, test2, test3])


@pytest.mark.slow
def test_non_ad_serialshot_derivative_matches_finite_difference():
    FWI_obj = build_serialshot_fwi(automated_adjoint=False)
    control = FWI_obj.initial_velocity_model.dat.data_ro.copy()
    J0, dJ = FWI_obj.return_functional_and_gradient(control)

    rng = np.random.default_rng(0)
    direction = rng.standard_normal(control.size)
    center_dofs = get_center_dofs(
        FWI_obj.initial_velocity_model,
        FWI_obj.mesh_z,
        FWI_obj.mesh_x,
    )
    direction[~center_dofs] = 0.0
    direction /= np.linalg.norm(direction)

    step = 1e-4
    J1 = FWI_obj.get_functional(c=control + step * direction)
    directional_fd = (J1 - J0) / step
    directional_ad = np.dot(dJ, direction)
    relative_error = np.abs(directional_fd - directional_ad) / (
        np.abs(directional_ad) + 1e-30
    )

    assert relative_error < 1e-2


@pytest.mark.slow
def test_single_source_vom_parity_between_adjoint_modes():
    real_shot_record = get_single_source_vom_real_shot_record()

    implemented_value, implemented_derivative = get_single_source_vom_derivative(
        automated_adjoint=False,
        real_shot_record=real_shot_record,
    )
    automated_value, automated_derivative = get_single_source_vom_derivative(
        automated_adjoint=True,
        real_shot_record=real_shot_record,
    )

    relative_derivative_error = np.linalg.norm(
        implemented_derivative - automated_derivative
    ) / max(
        np.linalg.norm(implemented_derivative),
        np.linalg.norm(automated_derivative),
        1e-30,
    )
    cosine_similarity = np.dot(
        implemented_derivative,
        automated_derivative,
    ) / max(
        np.linalg.norm(implemented_derivative)
        * np.linalg.norm(automated_derivative),
        1e-30,
    )

    assert math.isclose(
        implemented_value,
        automated_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert relative_derivative_error < 2e-3
    assert cosine_similarity > 0.99999


@pytest.mark.slow
@pytest.mark.skip(reason="ROL is not working for spyro")
def test_fwi_with_rol(load_real_shot=False, use_rol=True):
    test_fwi(
        automated_adjoint=False,
        load_real_shot=load_real_shot,
        use_rol=use_rol,
    )


if __name__ == "__main__":
    test_fwi(load_real_shot=False, automated_adjoint=False)
    # test_fwi_with_rol()
