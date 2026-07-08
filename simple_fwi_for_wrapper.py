"""Demo script for running a small full waveform inversion example.

The script builds a synthetic "true" model, generates a shot record, and then
runs a simple inversion loop against that record.
"""

import numpy as np
import firedrake as fire
from scipy.optimize import minimize as scipy_minimize
import spyro
import pytest


class MyFWI(spyro.FullWaveformInversion):
    def run_fwi(self, **kwargs):
        """
        Run full waveform inversion using scipy L-BFGS-B optimizer.

        Performs the complete FWI optimization using scipy.optimize.minimize
        with the L-BFGS-B method. The optimization minimizes the misfit between
        observed and simulated data by updating the configured control
        parameter.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for customizing the optimization:

            vmin : float, optional
                Lower bound for the control parameter. Default is 1.429.
            vmax : float, optional
                Upper bound for the control parameter. Default is 6.0.
            maxiter : int, optional
                Maximum number of iterations. Default is 20.
            scipy_options : dict, optional
                Additional options passed to scipy.optimize.minimize.
                Default includes disp=True, eps=1e-15, ftol=1e-11.

        Notes
        -----
        The final control parameter is stored in ``control_parameter_result``
        and saved to ``control_end.pvd``. The raw optimizer vector is also
        saved to ``result.npy``.

        This method uses the L-BFGS-B algorithm which is well-suited for
        large-scale bound-constrained optimization problems.

        Examples
        --------
        >>> fwi.run_fwi(maxiter=100, vmin=1.5, vmax=5.0)
        """
        parameters = {
            "vmin": kwargs.pop("vmin", 1.429),
            "vmax": kwargs.pop("vmax", 6.0),
            "scipy_options": {
                "disp": True,
                "eps": kwargs.pop("eps", 1e-15),
                "ftol": kwargs.pop("ftol", 1e-11),
                "maxiter": kwargs.pop("maxiter", 20),
            },
        }
        parameters.update(kwargs)

        control_reference = self._guess_control_reference()
        lower = self._expand_bound(parameters["vmin"], control_reference)
        upper = self._expand_bound(parameters["vmax"], control_reference)
        bounds = list(zip(lower, upper))
        control_0 = self._flatten_control(control_reference)
        options = parameters["scipy_options"]

        result = scipy_minimize(
            self.return_functional_and_gradient,
            control_0,
            method="L-BFGS-B",
            jac=True,
            tol=1e-15,
            bounds=bounds,
            options=options,
        )

        self.control_result = self._rebuild_control_from_vector(
            control_reference,
            result.x,
        )
        self.set_guess_control(self.control_result)

        self.control_parameter_result = self._copy_control_structure(
            self.control_result,
        )
        fire.VTKFile("control_end.pvd").write(self.control_parameter_result)

        np.save("result", result.x)
        return result


def run_forward_real_model(input_dictionary, case="camembert", shot_filename="shots/shot_record_"):
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

    fwi_obj = MyFWI(dictionary=input_dictionary)

    fwi_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.05})

    supported_cases = ["camembert"]
    if case == "camembert":
        # Builds the true velocity model based on a conditional
    
        center_z = -1.0
        center_x = 1.0
        mesh_z = fwi_obj.wave.mesh_z
        mesh_x = fwi_obj.wave.mesh_x
        cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .2**2, 3.0, 2.5)
    elif case == "layers":
        # not yet done here
        # Works for any number of horizontal layers and velocity values
        z_switch = [-1.0]  # List of floats representing z value where vp changes
        layer_vps = [2.5, 3.0]  # List of vp values
        cond = multiple_layer_velocity_model(fwi_obj, z_switch, layer_vps)
    elif case not in supported_cases:
        return ValueError(f"Case of {case} not part of supported cases: {supported_cases}")
    else:
        return ValueError(f"Case of {case} only partially implemented.")

    fwi_obj.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
    fwi_obj.generate_real_shot_record(
        plot_model=True,
        model_filename="True_experiment.png",
        shot_filename=shot_filename,
        abc_points=[(-0.5, 0.5), (-1.5, 0.5), (-1.5, 1.5), (-0.5, 1.5)]
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
    "dt": 0.0005,  # timestep size
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
    shots_filenames="shots/shot_record_"

    # Setting up to run synthetic real problem
    if load_real_shot is False:
        fwi_obj = run_forward_real_model(
            dictionary,
            case="layers",
            shot_filename=shots_filenames,
        )

    else:
        dictionary["inversion"]["real_shot_record_file"] = shots_filenames
        fwi_obj = MyFWI(dictionary=dictionary)

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

    fwi_obj.run_fwi(vmin=2.5, vmax=3.0, maxiter=15)

    print("END", flush=True)


if __name__ == "__main__":
    run_fwi(load_real_shot=False)
