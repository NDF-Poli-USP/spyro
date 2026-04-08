"""Camembert example models for acoustic forward and FWI workflows.

This module defines reusable dictionary configurations and helper classes for
setting up a 2D acoustic Camembert velocity model in spyro.
"""

from spyro import create_transect
from spyro.examples.rectangle import Rectangle_acoustic, Rectangle_acoustic_FWI
import firedrake as fire
import copy

camembert_optimization_parameters = {
    "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
    "Step": {
        "Type": "Augmented Lagrangian",
        "Augmented Lagrangian": {
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 5.0,
        },
        "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
    },
    "Status Test": {
        "Gradient Tolerance": 1e-16,
        "Iteration Limit": None,
        "Step Tolerance": 1.0e-16,
    },
}

camembert_dictionary = {}
camembert_dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T)
    # or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "method": "MLT",  # (MLT/spectral_quadrilateral/DG_triangle/
    # DG_quadrilateral)
    # You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    "automatic_adjoint": False,
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
camembert_dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for
    # every processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML)
# to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
camembert_dictionary["mesh"] = {
    "length_z": 1.0,  # depth in km - always positive
    "length_x": 1.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "h": 0.05,  # mesh size in km
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
}
# For use only if you are using a synthetic test model
# or a forward only simulation
camembert_dictionary["synthetic_data"] = {
    "real_mesh_file": None,
    "real_velocity_file": None,
}
camembert_dictionary["inversion"] = {
    "perform_fwi": False,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": camembert_optimization_parameters,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
camembert_dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center
# of the mesh.
# We also specify to record the solution at 101 microphones near the top
# of the domain.
# This transect of receivers is created with the helper function
#  `create_transect`.
camembert_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": create_transect((-0.90, 0.1), (-0.90, 0.9), 30),
}

# Simulate for 2.0 seconds.
camembert_dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.0,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    # how frequently to save solution to RAM
    "gradient_sampling_frequency": 100,
}

camembert_dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/camembert_forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}
camembert_dictionary["camembert_options"] = {
    "radius": 0.2,
    "circle_center": (-0.5, 0.5),
    "outside_velocity": 1.6,
    "inside_circle_velocity": 4.6,
}
camembert_dictionary_fwi = copy.deepcopy(camembert_dictionary)
camembert_dictionary_fwi["inversion"]["perform_fwi"] = True


class CamembertVelocity:
    """Mixin that builds the Camembert circular velocity distribution."""

    def _camembert_velocity_model(self):
        """Set the initial velocity model using a circular inclusion.

        The velocity is defined from ``camembert_options`` in
        ``self.input_dictionary``. Points inside the circle use
        ``inside_circle_velocity`` and points outside use ``outside_velocity``.

        Returns
        -------
        None
            The model is assigned in-place through
            :meth:`set_initial_velocity_model`.
        """
        camembert_dict = self.input_dictionary["camembert_options"]
        z = self.mesh_z
        x = self.mesh_x
        zc, xc = camembert_dict["circle_center"]
        rc = camembert_dict["radius"]
        c_salt = camembert_dict["inside_circle_velocity"]
        c_not_salt = camembert_dict["outside_velocity"]
        cond = fire.conditional(
            (z - zc) ** 2 + (x - xc) ** 2 < rc**2, c_salt, c_not_salt
        )
        self.set_initial_velocity_model(conditional=cond, dg_velocity_model=False)
        return None


class Camembert_acoustic(CamembertVelocity, Rectangle_acoustic):
    """Camembert acoustic example model.

    This class provides a reusable setup for a Camembert velocity model in
    forward acoustic simulations.

    Notes
    -----
    This example is intended as a reusable model configuration for
    development and testing of numerical methods. It does not represent a
    targeted validation case with a single expected output.

    By isolating common model setup logic, this class reduces boilerplate and
    encourages consistency across experiments.

    Parameters
    ----------
    dictionary: dict, optional
        Dictionary containing values that override entries in the default
        Camembert example configuration.
    example_dictionary: dict, optional
        Base dictionary used to configure the simulation. Defaults to
        ``camembert_dictionary``.
    comm: mpi4py.MPI.Comm, optional
        MPI communicator passed to the parent model class.
    periodic: bool, optional
        Whether periodic boundaries are requested. This example currently
        forwards ``False`` to the parent implementation.
    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=camembert_dictionary,
        comm=None,
        periodic=False,
    ):
        super().__init__(
            dictionary=dictionary,
            example_dictionary=example_dictionary,
            comm=comm,
            periodic=False,
        )
        self._camembert_velocity_model()


class Camembert_acoustic_FWI(CamembertVelocity, Rectangle_acoustic_FWI):
    """Camembert acoustic example model configured for FWI.

    This class provides a reusable Camembert setup for full-waveform
    inversion workflows.

    Notes
    -----
    This example is intended as a reusable model configuration for
    development and testing of inversion methods. It does not represent a
    targeted validation case with a single expected output.

    By isolating common model setup logic, this class reduces boilerplate and
    encourages consistency across experiments.

    Parameters
    ----------
    dictionary : dict, optional
        Dictionary containing values that override entries in the default
        Camembert FWI example configuration.
    example_dictionary : dict, optional
        Base dictionary used to configure the simulation. Defaults to
        ``camembert_dictionary_fwi``.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator passed to the parent model class.
    periodic : bool, optional
        Whether periodic boundaries are requested. This example currently
        forwards ``False`` to the parent implementation.
    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=camembert_dictionary_fwi,
        comm=None,
        periodic=False,
    ):
        super().__init__(
            dictionary=dictionary,
            example_dictionary=example_dictionary,
            comm=comm,
            periodic=False,
        )
        self._camembert_velocity_model()
        self.real_velocity_model = self.initial_velocity_model
        self.real_mesh = self.mesh
