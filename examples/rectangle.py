from spyro import create_transect
from spyro.examples.example_model import Example_model_acoustic
from spyro.examples.example_model import Example_model_acoustic_FWI
import firedrake as fire
import copy

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
    "cell_type": "T",
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
    "h": 0.05,  # mesh size in km
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
    "final_time": 1.0,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    # how frequently to save solution to RAM
    "gradient_sampling_frequency": 100,
}

rectangle_dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/rectangle_forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

rectangle_dictionary_fwi = copy.deepcopy(rectangle_dictionary)
rectangle_dictionary_fwi["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
}


class Rectangle_mesh_and_velocity:
    def _rectangle_mesh(self):
        mesh_dict = self.input_dictionary["mesh"]
        mesh_parameters = {
            "length_z": mesh_dict["Lz"],
            "length_x": mesh_dict["Lx"],
            "length_y": mesh_dict["Ly"],
            "edge_length": mesh_dict["h"],
            "mesh_file": mesh_dict["mesh_file"],
            "mesh_type": mesh_dict["mesh_type"],
            "periodic": self.periodic,
        }
        super().set_mesh(input_mesh_parameters=mesh_parameters)

    def multiple_layer_velocity_model(self, z_switch, layers):
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
                    self.mesh_z > z_switch[i], layers[i], layers[i + 1]
                )
            else:
                cond = fire.conditional(
                    self.mesh_z > z_switch[i], cond, layers[i + 1]
                )
        # cond = fire.conditional(self.mesh_z > z_switch, layer1, layer2)
        self.set_initial_velocity_model(conditional=cond)


class Rectangle_acoustic(Rectangle_mesh_and_velocity, Example_model_acoustic):
    """
    Rectangle model.
    This class is a child of the Example_model class.
    It is used to create a dictionary with the parameters of the
    Rectangle model.

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
    comm : firedrake.mpi_comm.MPI.Intracomm, optional
    periodic : bool, optional
        If True, the mesh will be periodic in all directions. The default is
        False.
    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=rectangle_dictionary,
        comm=None,
        periodic=False,
    ):
        super().__init__(
            dictionary=dictionary,
            default_dictionary=example_dictionary,
            comm=comm,
        )
        self.periodic = periodic

        self._rectangle_mesh()


class Rectangle_acoustic_FWI(Rectangle_mesh_and_velocity, Example_model_acoustic_FWI):
    """
    Rectangle model.
    This class is a child of the Example_model class.
    It is used to create a dictionary with the parameters of the
    Rectangle model.

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
    comm : firedrake.mpi_comm.MPI.Intracomm, optional
    periodic : bool, optional
        If True, the mesh will be periodic in all directions. The default is
        False.
    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=rectangle_dictionary_fwi,
        comm=None,
        periodic=False,
    ):
        super().__init__(
            dictionary=dictionary,
            default_dictionary=example_dictionary,
            comm=comm,
        )
        self.periodic = periodic

        self._rectangle_mesh()
