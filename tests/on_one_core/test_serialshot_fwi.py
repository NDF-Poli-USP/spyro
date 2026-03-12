import warnings

import firedrake as fire
import numpy as np
import pytest
import spyro


warnings.filterwarnings("ignore")

FINAL_TIME = 0.9


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    _ = set_test_tape


def build_dictionary(automatic_adjoint):
    source_count = 6
    receiver_count = 200
    if automatic_adjoint:
        source_count = 2
        receiver_count = 100

    acquisition = {
        "source_type": "ricker",
        "source_locations": spyro.create_transect(
            (-0.55, 0.7), (-0.55, 1.3), source_count
        ),
        "frequency": 5.0,
        "delay": 0.2,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect(
            (-1.45, 0.7), (-1.45, 1.3), receiver_count
        ),
    }
    if automatic_adjoint:
        acquisition["use_vertex_only_mesh"] = True

    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 4,
            "dimension": 2,
            "automatic_adjoint": automatic_adjoint,
        },
        "parallelism": {
            "type": "spatial",
        },
        "mesh": {
            "Lz": 2.0,
            "Lx": 2.0,
            "Ly": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": acquisition,
        "time_axis": {
            "initial_time": 0.0,
            "final_time": FINAL_TIME,
            "dt": 0.001,
            "amplitude": 1,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "forward_output_filename": "results/forward_output.pvd",
            "fwi_velocity_model_output": False,
            "velocity_model_filename": None,
            "gradient_output": False,
            "gradient_filename": "results/Gradient.pvd",
            "adjoint_output": False,
            "adjoint_filename": None,
            "debug_output": False,
        },
        "inversion": {
            "perform_fwi": True,
            "initial_guess_model_file": None,
            "shot_record_file": None,
        },
    }


def run_serialshot_fwi_case(
    automatic_adjoint,
    maxiter=5,
    scipy_options=None,
):
    dictionary = build_dictionary(automatic_adjoint)
    fwi_obj = spyro.FullWaveformInversion(dictionary=dictionary)

    try:
        fwi_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.1})
        center_z = -1.0
        center_x = 1.0
        cond = fire.conditional(
            (fwi_obj.mesh_z - center_z) ** 2 + (fwi_obj.mesh_x - center_x) ** 2 < 0.2 ** 2,
            3.0,
            2.5,
        )

        fwi_obj.set_real_velocity_model(
            conditional=cond,
            output=True,
            dg_velocity_model=False,
        )
        fwi_obj.generate_real_shot_record(
            plot_model=True,
            model_filename="True_experiment.png",
            abc_points=[(-0.5, 0.5), (-1.5, 0.5), (-1.5, 1.5), (-0.5, 1.5)],
        )

        fwi_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
        fwi_obj.set_guess_velocity_model(constant=2.5)
        fwi_obj.set_gradient_mask(
            boundaries={
                "z_min": -1.3,
                "z_max": -0.7,
                "x_min": 0.7,
                "x_max": 1.3,
            }
        )
        if automatic_adjoint:
            initial_model = fwi_obj.initial_velocity_model.dat.data_ro.copy()
            fwi_obj.get_gradient(save=False)
            gradient_data = fwi_obj.gradient.dat.data_ro.copy()
            gradient_scale = np.max(np.abs(gradient_data))
            trial_model = np.clip(
                initial_model - 0.5 * gradient_data / gradient_scale,
                2.5,
                3.0,
            )
            fwi_obj.get_functional(c=trial_model)
            return fwi_obj

        run_fwi_kwargs = {"vmin": 2.5, "vmax": 3.0}
        if scipy_options is None:
            run_fwi_kwargs["maxiter"] = maxiter
        else:
            run_fwi_kwargs["scipy_options"] = scipy_options
        fwi_obj.run_fwi(**run_fwi_kwargs)
        return fwi_obj
    finally:
        spyro.io.delete_tmp_files(fwi_obj)


@pytest.mark.slow
@pytest.mark.parametrize(
    "automatic_adjoint,maxiter,scipy_options,last_functional_max,reduction_max",
    [
        pytest.param(False, 5, None, 1e-3, 1e-2, id="implemented-adjoint"),
        pytest.param(True, None, None, 1e-3, 2e-1, id="automatic-adjoint"),
    ],
)
def test_serialshot_fwi(
    automatic_adjoint,
    maxiter,
    scipy_options,
    last_functional_max,
    reduction_max,
):
    fwi_obj = run_serialshot_fwi_case(
        automatic_adjoint,
        maxiter=maxiter,
        scipy_options=scipy_options,
    )

    gradient = fwi_obj.gradient
    masked_gradient = np.isclose(gradient.at((-0.1, 0.1)), 0.0)
    unmasked_gradient = np.abs(gradient.at((-1.0, 1.0))) > 1e-5
    last_functional_small = fwi_obj.functional < last_functional_max
    reduced_functional = (
        fwi_obj.functional_history[-1] / fwi_obj.functional_history[0] < reduction_max
    )

    print(f"PML looks masked: {masked_gradient}", flush=True)
    print(f"Center looks unmasked: {unmasked_gradient}", flush=True)
    print(f"Last functional small: {last_functional_small}", flush=True)
    print(
        "Considerable functional reduction during test: "
        f"{reduced_functional}",
        flush=True,
    )

    assert all(
        [
            masked_gradient,
            unmasked_gradient,
            last_functional_small,
            reduced_functional,
        ]
    )


@pytest.mark.slow
@pytest.mark.skip(reason="ROL is not working for spyro")
def test_serialshot_fwi_with_rol():
    fwi_obj = spyro.FullWaveformInversion(dictionary=build_dictionary(False))
    try:
        fwi_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.1})
        cond = fire.conditional(
            (fwi_obj.mesh_z + 1.0) ** 2 + (fwi_obj.mesh_x - 1.0) ** 2 < 0.2 ** 2,
            3.0,
            2.5,
        )
        fwi_obj.set_real_velocity_model(
            conditional=cond,
            output=True,
            dg_velocity_model=False,
        )
        fwi_obj.generate_real_shot_record()
        fwi_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
        fwi_obj.set_guess_velocity_model(constant=2.5)
        fwi_obj.run_fwi_rol(vmin=2.5, vmax=3.0, maxiter=2)
    finally:
        spyro.io.delete_tmp_files(fwi_obj)
