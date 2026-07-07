"""Demo script for running a small full waveform inversion example.

The script builds a synthetic "true" model, generates a shot record, and then
runs a simple inversion loop against that record.
"""

import numpy as np
import firedrake as fire
import spyro
import pytest


def run_forward_real_model(input_dictionary, case="camembert", output_filename="real_shot_record"):
    """Generate and save a synthetic shot record for the chosen demo case.

    Parameters
    ----------
    input_dictionary : dict
        Configuration dictionary used to build the forward-modeling object.
    case : str, optional
        Demo model to generate. Currently only ``"camembert"`` is supported.
    output_filename : str, optional
        Base filename used when saving the generated shot record with NumPy.

    Returns
    -------
    None
        The generated shot record is written to disk.
    """

    fwi_obj = spyro.FullWaveformInversion(dictionary=input_dictionary)

    fwi_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.05})

    supported_cases = ["camembert"]
    if case == "camembert":
        # Builds the true velocity model based on a conditional
    
        center_z = -1.0
        center_x = 1.0
        mesh_z = fwi_obj.wave.mesh_z
        mesh_x = fwi_obj.wave.mesh_x
        cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .2**2, 3.0, 2.5)
    elif case not in supported_cases:
        return ValueError(f"Case of {case} not part of supported cases: {supported_cases}")
    else:
        return ValueError(f"Case of {case} only partially implemented.")

    fwi_obj.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
    fwi_obj.generate_real_shot_record(
        plot_model=True,
        model_filename="True_experiment.png",
        abc_points=[(-0.5, 0.5), (-1.5, 0.5), (-1.5, 1.5), (-0.5, 1.5)]
    )
    np.save(output_filename, fwi_obj.real_shot_record)


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
    "length_z": 2.0,  # depth in km - always positive
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
}


def run_fwi(load_real_shot=True):
    """Run the demo inversion workflow.

    Parameters
    ----------
    load_real_shot : bool, optional
        If ``True``, load the saved shot record from disk. If ``False``,
        generate a fresh synthetic shot record first.

    Returns
    -------
    None
        The inversion is run for its side effects.
    """

    # Setting up to run synthetic real problem
    if load_real_shot is False:
        run_forward_real_model(
            dictionary,
            case="camembert",
            output_filename="real_shot_record",
        )
        dictionary["inversion"]["shot_record_file"] = "real_shot_record.npy"

    else:
        dictionary["inversion"]["shot_record_file"] = "real_shot_record.npy"

    fwi_obj = spyro.FullWaveformInversion(dictionary=dictionary)

    # Checking if shot recorded loaded:
    if fwi_obj.real_shot_record is None:
        raise ValueError("True shot record not loaded. Either create one or load correctly.")

    # Setting up initial guess problem
    fwi_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
    fwi_obj.set_guess_velocity_model(constant=2.5)

    # This is deprecated, in more complex cases we mark zero gradient boundaries directly on the mesh
    # which is a lot more efficient
    mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
    fwi_obj.set_gradient_mask(boundaries=mask_boundaries)

    fwi_obj.run_fwi(vmin=2.5, vmax=3.0, maxiter=5)

    print("END", flush=True)


if __name__ == "__main__":
    run_fwi(load_real_shot=False)
