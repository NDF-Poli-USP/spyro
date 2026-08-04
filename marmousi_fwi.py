"""Demo script for running a 2D marmousi full waveform inversion example.

This demo has automatic parallelism set up and 40 shots, therefore we need
a 40n (n positive integer) number of cores in mpiexec -n N_CORES to run. 
You can experiment with a different number of cores if desired, but you
would need to change the parallelism dicionary setting. Please use mintrop
to run this script.

The script builds a synthetic "true" model, generates a shot record, and then
runs a simple inversion loop against that record.
"""

from copy import deepcopy
import numpy as np
import firedrake as fire
import spyro
import pytest


def run_forward_real_model(default_dictionary, shot_filename="shots/shot_record_", dt=None, save_vp_as_segy=False, segy_filename="velocity_models/vp_marmousi-ii.segy"):
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
    input_dictionary = deepcopy(default_dictionary)
    if dt is not None:
        original_dt = deepcopy(default_dictionary["time_axis"]["dt"])
        input_dictionary["time_axis"]["dt"] = dt

    input_dictionary["mesh"]["cells_per_wavelength"] = 3.5

    fwi_obj = spyro.FullWaveformInversion(dictionary=input_dictionary)

    fwi_obj.set_real_velocity_model(new_file="velocity_models/vp_marmousi-ii.segy", fast_interpolate=True)
    fwi_obj.generate_real_shot_record(
        plot_model=True,
        model_filename="True_experiment.png",
        shot_filename=shot_filename,
    )
    if dt is not None:
        fwi_obj.wave.dt = original_dt

    if save_vp_as_segy:
        export_grid_spacing = 0.01
        spyro.io.export_scalar_field(
            fwi_obj.wave.initial_velocity_model,
            export_grid_spacing, segy_filename,
            comm=fwi_obj.wave.comm,
        )

    return fwi_obj


real_shot_record_dt = 0.001
simulation_dt = 0.001
degree = 4
frequency = 5.0
final_time = 4.0
length_x = 17.0
length_z = 3.5


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": degree,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "length_z": length_z,
    "length_x": length_x,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "output_filename": "trial01.msh",
    "cells_per_wavelength": 3.0,
    "frequency": frequency,
    "segy_velocity_model": "velocity_models/vp_marmousi-ii.segy",
    "mesh_type": "gmsh_mesh",
    "grade": 0.05,
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.01, 4.0), (-0.01, 12.0), 40),
    "frequency": frequency,
    "delay": 1.0/frequency,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-0.1, 4.0), (-0.1, 12.0), 100),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}
dictionary["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_output.pvd",
}


def setting_up_fwi():

    shots_filenames="shots/shot_record_"

    # Setting up to run synthetic real problem
    wave_obj = run_forward_real_model(
        dictionary,
        dt=real_shot_record_dt,
        shot_filename=shots_filenames,
        save_vp_as_segy=True,
        segy_filename="velocity_models/vp_marmousi-ii.segy",
    )

    # Let us create a smoothed out initial guess based on a gaussian
    # filter of our true model
    sigma = 100  # standart deviation for the gaussian filter
    spyro.tools.smooth_velocity_field_file(
        "velocity_models/vp_marmousi-ii.segy",
        "initial_guess.segy",
        sigma,
        save_fig=True,
        vp_limit=0.0,
        i_limit=45,
        comm=wave_obj.comm,
    )


def run_fwi():
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
    dictionary["time_axis"]["dt"] = simulation_dt
    shots_filenames="shots/shot_record_"
    dictionary["inversion"]["real_shot_record_file"] = shots_filenames
    fwi_obj = spyro.FullWaveformInversion(dictionary=dictionary)

    # Since the shot record is using a different timestep than our guess model we have to interpolate the time series into our desired timestep
    fwi_obj.real_shot_record = spyro.io.time_io.interpolate_time_series(
        fwi_obj.real_shot_record,
        simulation_dt,
        0.0,
        final_time,
    )

    # Setting up initial guess problem. Since our mesh is adapted to our segy file
    # we cannot yet load the segy. Thi is because loading requires interpolation or
    # projection into a Function Space where the velocity will live for the
    # simulation. This Function Space cannot be created without a mesh.

    # Therefore, we need to pass the vp as a grid to the meshing utilities.
    grid_vp_data = spyro.io.segy_io.create_grid_dictionary_from_segy(
        "initial_guess.segy",
        length_z,
        length_x,
    )

    # let us create our mesh adapted to material properties.
    # inside the mesh we can also already tag the gradient mask.
    meshing_dictionary = {
        "length_z": length_z,
        "length_x": length_x,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "output_filename": "trial01.msh",
        "cells_per_wavelength": 2.7,
        "frequency": frequency,
        "segy_velocity_model": "initial_guess.segy",
        "mesh_type": "gmsh_mesh",
        "grade": 0.05,
    }
    fwi_obj.set_guess_mesh(input_mesh_parameters=meshing_dictionary)

    # Let us set the initial guess velocity
    fwi_obj.set_guess_velocity_model(new_file="initial_guess.hdf5", fast_interpolate=True)
    fwi_obj.run_fwi(vmin=1.4, vmax=4.7, maxiter=20, )

    # Let us have a look at our solution
    export_grid_spacing = 0.01
    spyro.io.export_scalar_field(fwi_obj.wave.c, export_grid_spacing, "layers.png", comm=fwi_obj.wave.comm)

    print("END", flush=True)


if __name__ == "__main__":
    setting_up_fwi()
    run_fwi()
