"""Demo script for running a small full waveform inversion example.

This demo has automatic parallelism set up and 8 shots, therefore we need
a 6n (n positive integer) number of cores in mpiexec -n N_CORES to run. 
You can experiment with a different number of cores if desired, but you
would need to change the parallelism dicionary setting.

The script builds a synthetic "true" model, generates a shot record, and then
runs a simple inversion loop against that record. It uses a simple layer model.
"""
# from mpi4py.MPI import COMM_WORLD
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()

from copy import deepcopy
import numpy as np
import firedrake as fire
import spyro
import pytest


def run_forward_real_model(input_dictionary, shot_filename="shots/shot_record_", dt=None, save_vp_as_segy=False, segy_filename="real_vp.segy"):
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

    z_switch = [-1.0]  # List of floats representing z value where vp changes
    layer_vps = [2.5, 3.0]  # List of vp values
    cond = multiple_layer_velocity_model(fwi_obj, z_switch, layer_vps)

    fwi_obj.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
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


def multiple_layer_velocity_model(fwi_obj, z_switch, layers):
    """
    Sets the heterogeneous velocity model to be split into horizontal layers.
    Each layer's velocity value is defined by the corresponding value in the
    layers list. The layers are separated by the values in the z_switch list.

    Parameters
    ----------
    z_switch : list of floats
        List of z values that separate the layers.
    layers : list of floats
        List of velocity values for each layer.
    """
    if len(z_switch) != (len(layers) - 1):
        raise ValueError(
            "Float list of z_switch has to have length exactly one less \
                            than list of layer values"
        )
    if len(z_switch) == 0:
        raise ValueError("Float list of z_switch cannot be empty")
    for i in range(len(z_switch)):
        if i == 0:
            cond = fire.conditional(
                fwi_obj.wave.mesh_z > z_switch[i], layers[i], layers[i + 1]
            )
        else:
            cond = fire.conditional(
                fwi_obj.wave.mesh_z > z_switch[i], cond, layers[i + 1]
            )

    return cond


final_time = 1.2
real_shot_record_dt = 0.0005
simulation_dt = 0.001

length_z = 2.0
length_x = 2.0
frequency = 5.0

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
    "length_z": length_z,  # depth in km - always positive
    "length_x": length_x,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.05, 0.5), (-0.05, 1.5), 6),
    "frequency": frequency,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-.05, 0.5), (-0.05, 1.5), 200),
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


def setting_up_fwi():

    shots_filenames="shots/shot_record_"

    # Setting up to run synthetic real problem
    wave_obj = run_forward_real_model(
        dictionary,
        dt=real_shot_record_dt,
        shot_filename=shots_filenames,
        save_vp_as_segy=True,
        segy_filename="real_layers.segy",
    )

    # Let us create a smoothed out initial guess based on a gaussian
    # filter of our true model
    sigma = 20  # standart deviation for the gaussian filter
    spyro.tools.smooth_velocity_field_file(
        "real_layers.segy",
        "initial_guess.segy",
        sigma,
        save_fig=True,
        vp_limit=0.0,
        i_limit=25,
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
        "length_x": length_x,
        "mesh_type": "spyro_mesh",
        "dimension": 2,
        "output_filename": "trial01.msh",
        # Values greater than this will have a null gradient
        "gradient_mask": {
            "z_max": -0.5,
            "z_min": -length_z,
            "x_min": 0.0,
            "x_max": length_x,
        },
        "cells_per_wavelength": 3.0,
        "frequency": frequency,
        "grid_velocity_data": grid_vp_data,

    }
    fwi_obj.set_guess_mesh(input_mesh_parameters=meshing_dictionary)

    # Let us set the initial guess velocity
    # fwi_obj.set_guess_velocity_model(new_file="initial_guess.hdf5")
    fwi_obj.set_guess_velocity_model(constant=2.5)
    fwi_obj.run_fwi(vmin=2.0, vmax=3.5, maxiter=20, )

    # Let us have a look at our solution
    export_grid_spacing = 0.01
    spyro.io.export_scalar_field(fwi_obj.wave.c, export_grid_spacing, "layers.png", comm=fwi_obj.wave.comm)

    print("END", flush=True)


if __name__ == "__main__":
    setting_up_fwi()
    run_fwi()
