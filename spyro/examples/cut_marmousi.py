from spyro import create_transect
from spyro.examples.example_model import Example_model_acoustic

cut_marmousi_optimization_parameters = {
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

cut_marmousi_dictionary = {}
cut_marmousi_dictionary["options"] = {
    # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "cell_type": "T",
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    "automatic_adjoint": False,
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
cut_marmousi_dictionary["parallelism"] = {
    # options: automatic (same number of cores for evey processor) or spatial
    "type": "automatic",
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML)
# to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
cut_marmousi_dictionary["mesh"] = {
    "Lz": 2.0,  # depth in km - always positive
    "Lx": 4.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": "meshes/cut_marmousi_small_p=2_M=7.02.msh",
}
cut_marmousi_dictionary[
    "synthetic_data"
    # For use only if you are using a synthetic test model or
    #  a forward only simulation -adicionar discrição para modelo direto
] = {
    "real_velocity_file": "velocity_models/MODEL_P-WAVE_VELOCITY_1.25m_small_domain.hdf5",  # noqa: E501
}
cut_marmousi_dictionary["inversion"] = {
    "perform_fwi": False,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": cut_marmousi_optimization_parameters,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
cut_marmousi_dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
    "outer_bc": False,  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    # thickness of the PML in the z-direction (km) - always positive
    "lz": 0.25,
    # thickness of the PML in the x-direction (km) - always positive
    "lx": 0.25,
    # thickness of the PML in the y-direction (km) - always positive
    "ly": 0.0,
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center
# of the mesh.
# We also specify to record the solution at 101 microphones near the top
# of the domain.
# This transect of receivers is created with the helper function
# `create_transect`.
cut_marmousi_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(2.0, -0.01)],
    "frequency": 3.0,
    "amplitude": 1.0,
    "delay": 1.0,
    "receiver_locations": create_transect((0.1, -0.10), (3.9, -0.10), 100),
}

# Simulate for 2.0 seconds.
cut_marmousi_dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 2.5,  # Final time for event
    "dt": 0.00025,  # timestep size
    "output_frequency": 20,  # how frequently to output solution to pvds
    # how frequently to save solution to RAM
    "gradient_sampling_frequency": 10,
}


class Cut_marmousi_acoustic(Example_model_acoustic):
    """
    Class for the cut marmousi model.

    Example Setup

    These examples are intended as reusable velocity model configurations to assist in the development and testing of new methods, such as optimization algorithms, time-marching schemes, or inversion techniques.

    Unlike targeted test cases, these examples do not have a specific objective or expected result. Instead, they provide standardized setups, such as Camembert, rectangular, and Marmousi velocity models, that can be quickly reused when prototyping, testing, or validating new functionality.

    By isolating the setup of common velocity models, we aim to reduce boilerplate and encourage consistency across experiments.

    Feel free to adapt these templates to your needs.

    Parameters
    ----------
    dictionary : dict, optional
        Dictionary with the parameters of the model that are different from
        the default model. The default is None.
    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=cut_marmousi_dictionary,
        comm=None,
    ):
        super().__init__(
            dictionary=dictionary,
            default_dictionary=example_dictionary,
            comm=comm,
        )
