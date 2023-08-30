from spyro import create_transect
from spyro.examples.rectangle import Rectangle_acoustic
import firedrake as fire

camembert_optimization_parameters = {
    "General": {
        "Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}
    },
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
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "h": 0.05,  # mesh size in km
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
    "h": 0.05,
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


class Camembert_acoustic(Rectangle_acoustic):
    """Camembert model.
    This class is a child of the Example_model class.
    It is used to create a dictionary with the parameters of the
    Camembert model.

    Parameters
    ----------
    dictionary : dict, optional
        Dictionary with the parameters of the model that are different from
        the default Camembert model. The default is None.

    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=camembert_dictionary,
        comm=None,
    ):
        super().__init__(
            dictionary=dictionary,
            example_dictionary=example_dictionary,
            comm=comm,
        )
        self._camembert_velocity_model()

    def _camembert_velocity_model(self):
        z = self.mesh_z
        x = self.mesh_x
        zc = -0.5
        xc = 0.5
        rc = 0.2
        c_salt = 4.6
        c_not_salt = 1.6
        cond = fire.conditional(
            (z - zc) ** 2 + (x - xc) ** 2 < rc**2, c_salt, c_not_salt
        )
        self.set_initial_velocity_model(conditional=cond)
        return None
