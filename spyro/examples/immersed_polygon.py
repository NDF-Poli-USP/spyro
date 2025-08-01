from spyro import create_transect
from spyro.examples.rectangle import Rectangle_acoustic, Rectangle_acoustic_FWI
import firedrake as fire
import copy
# Adapted from Velocity model-based adapted meshes using optimal transport
# TODO: add correct citation as soon as Thiago's paper is published

polygon_optimization_parameters = {
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

polygon_dictionary = {}
polygon_dictionary["options"] = {
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
polygon_dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for
    # every processor) or spatial
}
polygon_dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "h": 0.05,  # mesh size in km
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
}
polygon_dictionary["synthetic_data"] = {
    "real_mesh_file": None,
    "real_velocity_file": None,
}
polygon_dictionary["inversion"] = {
    "perform_fwi": False,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": polygon_optimization_parameters,
}
polygon_dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
}
polygon_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": create_transect((-0.90, 0.1), (-0.90, 0.9), 30),
}
polygon_dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.0,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    # how frequently to save solution to RAM
    "gradient_sampling_frequency": 100,
}
polygon_dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/polygon_forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}
polygon_dictionary["polygon_options"] = {
    "water_layer_is_present": True,
    "water_layer_depth": 0.2,
    "upper_layer": 2.0,
    "middle_layer": 2.5,
    "lower_layer": 3.0,
    "polygon_layer_perturbation": 0.3,
}
polygon_dictionary_fwi = copy.deepcopy(polygon_dictionary)
polygon_dictionary_fwi["inversion"]["perform_fwi"] = True


class Polygon_velocity:
    def _polygon_velocity_model(self):
        polygon_dict = self.input_dictionary["polygon_options"]
        z = self.mesh_z
        x = self.mesh_x

        v0 = 1.5  # water layer
        water_layer_depth = polygon_dict.get("water_layer_depth", 0.0)
        water_layer_present = polygon_dict.get("water_layer_is_present", False)
        v1 = polygon_dict["upper_layer"]
        v2 = polygon_dict["middle_layer"]  # background vp (km/s)
        vl = polygon_dict["lower_layer"]  # lower layer (km/s)
        dv = polygon_dict["polygon_layer_perturbation"]*v2  # 30% of perturbation
        d0 = -water_layer_depth
        d1 = d0 - 0.14
        d2 = d1 - 0.2

        if water_layer_present:
            cond = fire.conditional(z >= d0, v0, v1)
            cond = fire.conditional(z <= d1, v2, cond)
        else:
            cond = fire.conditional(z <= d1, v2, v1)
        cond = fire.conditional(z <= d2 - 0.2*x, vl, cond)

        cond = fire.conditional(300*((x-0.5)*(-z-0.5))**2 + ((x-0.5)+(-z-0.5))**2 <= 0.300**2, v2+dv, cond)

        if self.abc_pad_length > 0.0:
            middle_of_pad = -self.length_z - self.abc_pad_length*0.5
            cond = fire.conditional(z <= middle_of_pad, v0, cond)

        self.set_initial_velocity_model(conditional=cond, dg_velocity_model=False)
        return None


class Polygon_acoustic(Polygon_velocity, Rectangle_acoustic):
    """polygon model.
    This class is a child of the Example_model class.
    It is used to create a dictionary with the parameters of the
    polygon model.

    Example Setup

    These examples are intended as reusable velocity model configurations to assist in the development and testing of new methods, such as optimization algorithms, time-marching schemes, or inversion techniques.

    Unlike targeted test cases, these examples do not have a specific objective or expected result. Instead, they provide standardized setups, such as Camembert, rectangular, and Marmousi velocity models, that can be quickly reused when prototyping, testing, or validating new functionality.

    By isolating the setup of common velocity models, we aim to reduce boilerplate and encourage consistency across experiments.

    Feel free to adapt these templates to your needs.

    Parameters
    ----------
    dictionary : dict, optional
        Dictionary with the parameters of the model that are different from
        the default polygon model. The default is None.

    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=polygon_dictionary,
        comm=None,
        periodic=False,
    ):
        super().__init__(
            dictionary=dictionary,
            example_dictionary=example_dictionary,
            comm=comm,
            periodic=periodic,
        )
        self._polygon_velocity_model()


class Polygon_acoustic_FWI(Polygon_velocity, Rectangle_acoustic_FWI):
    """polygon model.
    This class is a child of the Example_model class.
    It is used to create a dictionary with the parameters of the
    polygon model.

    Example Setup

    These examples are intended as reusable velocity model configurations to assist in the development and testing of new methods, such as optimization algorithms, time-marching schemes, or inversion techniques.

    Unlike targeted test cases, these examples do not have a specific objective or expected result. Instead, they provide standardized setups, such as Camembert, rectangular, and Marmousi velocity models, that can be quickly reused when prototyping, testing, or validating new functionality.

    By isolating the setup of common velocity models, we aim to reduce boilerplate and encourage consistency across experiments.

    Feel free to adapt these templates to your needs.

    Parameters
    ----------
    dictionary : dict, optional
        Dictionary with the parameters of the model that are different from
        the default polygon model. The default is None.

    """

    def __init__(
        self,
        dictionary=None,
        example_dictionary=polygon_dictionary,
        comm=None,
        periodic=False,
    ):
        super().__init__(
            dictionary=dictionary,
            example_dictionary=example_dictionary,
            comm=comm,
            periodic=periodic,
        )
        self._polygon_velocity_model()
        self.real_velocity_model = self.initial_velocity_model
        self.real_mesh = self.mesh
