from spyro import create_transect
from spyro.examples.example_model import Example_model_acoustic
import firedrake as fire

rectangle_optimization_parameters = {
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

rectangle_dictionary = {}
rectangle_dictionary["options"] = {
    # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "cell_type": "Q",
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    "automatic_adjoint": False,
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
rectangle_dictionary["parallelism"] = {
    # options: automatic (same number of cores for evey processor) or spatial
    "type": "automatic",
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML)
# to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
rectangle_dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
}
rectangle_dictionary[
    "synthetic_data"
    # For use only if you are using a synthetic test model or a forward only
    # simulation -adicionar discrição para modelo direto
] = {
    "real_mesh_file": None,
    "real_velocity_file": None,
    "velocity_conditional": None,
}
rectangle_dictionary["inversion"] = {
    "perform_fwi": False,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": rectangle_optimization_parameters,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
rectangle_dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "PML",
    "exponent": 2,
    "cmax": 4.5,
    "R": 1e-6,
    "pad_length": 0.25,
}

rectangle_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.3)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": create_transect((-0.10, 0.1), (-0.10, 0.9), 20),
}

# Simulate for 2.0 seconds.
rectangle_dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    # how frequently to save solution to RAM
    "gradient_sampling_frequency": 100,
}

rectangle_dictionary["visualization"] = {
    "forward_output": True,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}


class Rectangle_acoustic(Example_model_acoustic):
    def __init__(
        self,
        dictionary=None,
        example_dictionary=rectangle_dictionary,
        comm=None,
    ):
        super().__init__(
            dictionary=dictionary,
            default_dictionary=example_dictionary,
            comm=comm,
        )
        self._rectangle_mesh()

    def _rectangle_mesh(self):
        dictionary = self.input_dictionary["mesh"]
        h = dictionary["h"]
        super().set_mesh(dx=h)


# class Rectangle(AcousticWave):
#     def __init__(self, model_dictionary=None, comm=None):
#         model_parameters = Rectangle_parameters(
#             dictionary=model_dictionary, comm=comm
#         )
#         super().__init__(
#             model_parameters=model_parameters, comm=model_parameters.comm
#         )
#         comm = self.comm
#         num_sources = self.number_of_sources
#         if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
#             print(
#                 "INFO: Distributing %d shot(s) across %d core(s). \
#                     Each shot is using %d cores"
#                 % (
#                     num_sources,
#                     fire.COMM_WORLD.size,
#                     fire.COMM_WORLD.size / comm.ensemble_comm.size,
#                 ),
#                 flush=True,
#             )
