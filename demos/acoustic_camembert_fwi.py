"""Demo script for running a small full waveform inversion example.

This demo has automatic parallelism set up and 6 shots, therefore we need
a 6n (n positive integer) number of cores in mpiexec -n N_CORES to run. 
You can experiment with a different number of cores if desired, but you
would need to change the parallelism dicionary setting.

The script builds a synthetic "true" model, generates a shot record, and then
runs a simple inversion loop against that record. It uses a simple Camembert
circular cheese model.
"""

from copy import deepcopy
import numpy as np
import firedrake as fire
import spyro
import pytest


def run_forward_real_model(input_dictionary, case="camembert", shot_filename="shots/shot_record_", dt=None):
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
    if dt is not None:
        original_dt = deepcopy(dictionary["time_axis"]["dt"])
        dictionary["time_axis"]["dt"] = dt

    fwi_obj = spyro.FullWaveformInversion(dictionary=input_dictionary)

    fwi_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.05})

    # Builds the true velocity model based on a conditional

    center_z = -1.0
    center_x = 1.0
    mesh_z = fwi_obj.wave.mesh_z
    mesh_x = fwi_obj.wave.mesh_x
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .2**2, 3.0, 2.5)

    fwi_obj.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
    fwi_obj.generate_real_shot_record(
        plot_model=True,
        model_filename="True_experiment.png",
        shot_filename=shot_filename,
        abc_points=[(-0.5, 0.5), (-1.5, 0.5), (-1.5, 1.5), (-0.5, 1.5)]
    )
    if dt is not None:
        fwi_obj.wave.dt = original_dt

    return fwi_obj


final_time = 0.9
real_shot_record_dt = 0.0005
simulation_dt = 0.001

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for every propagation)
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
    "frequency": 5.0,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.45, 0.7), (-1.45, 1.3), 200),
    "use_vertex_only_mesh": True,
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": simulation_dt,  # timestep size
    "amplitude": 1,  # the Ricker wave has an amplitude of 1.
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
    shots_filenames="shots/shot_record_"

    # Setting up to run synthetic real problem
    if load_real_shot is False:
        fwi_obj = run_forward_real_model(
            dictionary,
            dt=real_shot_record_dt,
            shot_filename=shots_filenames,
        )

    else:
        dictionary["time_axis"]["dt"] = simulation_dt
        dictionary["inversion"]["real_shot_record_file"] = shots_filenames
        fwi_obj = spyro.FullWaveformInversion(dictionary=dictionary)

    # Since the shot record is using a different timestep than our guess model we have to interpolate it
    fwi_obj.real_shot_record = spyro.io.time_io.interpolate_time_series(
        fwi_obj.real_shot_record,
        simulation_dt,
        0.0,
        final_time,
    )

    # Setting up initial guess problem
    fwi_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.2})
    fwi_obj.set_guess_velocity_model(constant=2.5)

    # This is deprecated, in more complex cases we mark zero gradient boundaries directly on the mesh
    # which is a lot more efficient.
    mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
    fwi_obj.set_gradient_mask(boundaries=mask_boundaries)

    fwi_obj.run_fwi(vmin=2.5, vmax=3.0, maxiter=20)
    export_grid_spacing = 0.01

    # Let us have a look at our solution
    spyro.io.export_scalar_field(fwi_obj.wave.c, export_grid_spacing, "camembert.png", comm=fwi_obj.wave.comm)

    print("END", flush=True)


if __name__ == "__main__":
    run_fwi(load_real_shot=False)
