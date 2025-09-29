import numpy as np
import firedrake as fire
import spyro
import pytest


def is_rol_installed():
    try:
        import ROL
        return True
    except ImportError:
        return False


final_time = 0.9

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "length_z": 2.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "length_x": 2.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 6),
    # "source_locations": [(-1.1, 1.5)],
    "frequency": 5.0,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.45, 0.7), (-1.45, 1.3), 200),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
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
}


@pytest.mark.parallel(6)
def test_fwi(load_real_shot=False, use_rol=False):
    """
    Run the Full Waveform Inversion (FWI) test.

    Parameters
    ----------
        load_real_shot (bool, optional): Whether to load a real shot record or not. Defaults to False.
    """

    # Setting up to run synthetic real problem
    if load_real_shot is False:
        FWI_obj = spyro.FullWaveformInversion(dictionary=dictionary)

        FWI_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.1})
        center_z = -1.0
        center_x = 1.0
        mesh_z = FWI_obj.mesh_z
        mesh_x = FWI_obj.mesh_x
        cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .2**2, 3.0, 2.5)

        FWI_obj.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
        FWI_obj.generate_real_shot_record(
            plot_model=True,
            model_filename="True_experiment.png",
            abc_points=[(-0.5, 0.5), (-1.5, 0.5), (-1.5, 1.5), (-0.5, 1.5)]
        )
        np.save("real_shot_record", FWI_obj.real_shot_record)

    else:
        dictionary["inversion"]["shot_record_file"] = "real_shot_record.npy"
        FWI_obj = spyro.FullWaveformInversion(dictionary=dictionary)

    # Setting up initial guess problem
    FWI_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
    FWI_obj.set_guess_velocity_model(constant=2.5)
    mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
    FWI_obj.set_gradient_mask(boundaries=mask_boundaries)
    if use_rol:
        FWI_obj.run_fwi_rol(vmin=2.5, vmax=3.0, maxiter=2)
    else:
        FWI_obj.run_fwi(vmin=2.5, vmax=3.0, maxiter=5)

    # simple mask test
    grad_test = FWI_obj.gradient
    test0 = np.isclose(grad_test.at((-0.1, 0.1)), 0.0)
    print(f"PML looks masked: {test0}", flush=True)
    test1 = np.abs(grad_test.at((-1.0, 1.0))) > 1e-5
    print(f"Center looks unmasked: {test1}", flush=True)

    # quick look at functional and if it reduced
    test2 = FWI_obj.functional < 1e-3
    print(f"Last functional small: {test2}", flush=True)
    test3 = FWI_obj.functional_history[-1]/FWI_obj.functional_history[0] < 1e-2
    print(f"Considerable functional reduction during test: {test3}", flush=True)

    print("END", flush=True)
    assert all([test0, test1, test2, test3])


@pytest.mark.skip()
@pytest.mark.parallel(6)
def test_fwi_with_rol(load_real_shot=False, use_rol=True):
    test_fwi(load_real_shot=load_real_shot, use_rol=use_rol)


if __name__ == "__main__":
    test_fwi(load_real_shot=False)
    test_fwi_with_rol()
