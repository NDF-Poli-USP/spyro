import numpy as np
import spyro
import warnings


def build_on_top_of_base_dictionary(variables):
    """
    Builds a model dictionary on top of the base dictionary.

    Parameters
    ----------
    variables : dict
        Dictionary containing the variables to be used in the model dictionary. It should include:
            - method: string
                The finite element method to be used. Either "mass_lumped_triangle" or "spectral_quadrilateral".
            - degree: int
                The spatial polynomial degree of the finite element method
            - dimension: int
                The dimension of the problem. Either 2 or 3.
            - Lz: float
                The length of the domain in the z direction.
            - Lx: float
                The length of the domain in the x direction.
            - Ly: float
                The length of the domain in the y direction.
            - cells_per_wavelength: float
                The number of cells per wavelength.
            - pad: float
                The padding to be used in the domain.
            - source_locations: list
                A list containing the source locations.
            - frequency: float
                The frequency of the source.
            - receiver_locations: list
                A list containing the receiver locations.
            - final_time: float
                The final time of the simulation.
            - dt: float
                The time step size of the simulation.

    Returns
    -------
    model_dictionary : dict
        Dictionary containing the model dictionary.
    """
    mesh_type = set_mesh_type(variables["method"])
    model_dictionary = {}
    model_dictionary["options"] = {
        "method": variables["method"],
        "degree": variables["degree"],
        "dimension": variables["dimension"],
        "automatic_adjoint": False,
    }
    model_dictionary["parallelism"] = {"type": "automatic",}
    model_dictionary["mesh"] = {
        "Lz": variables["Lz"],
        "Lx": variables["Lx"],
        "Ly": variables["Ly"],
        "cells_per_wavelength": variables["cells_per_wavelength"],
        "mesh_type": mesh_type,
    }
    model_dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": variables["pad"],
    }
    model_dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": variables["source_locations"],
        "frequency": variables["frequency"],
        "receiver_locations": variables["receiver_locations"],
    }
    model_dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": variables["final_time"],  # Final time for event
        "dt": variables["dt"],  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 1000,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }
    model_dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/reference_forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }
    return model_dictionary


def set_mesh_type(method):
    """
    Sets the mesh type based on the method.

    Parameters
    ----------
    method : string
        The finite element method to be used. Either "mass_lumped_triangle" or "spectral_quadrilateral".

    Returns
    -------
    mesh_type : string
        The mesh type to be used.
    """
    if method == "mass_lumped_triangle":
        mesh_type = "SeismicMesh"
    elif method == "spectral_quadrilateral":
        mesh_type = "firedrake_mesh"
    else:
        raise ValueError("Method is not mass_lumped_triangle or spectral_quadrilateral")
    return mesh_type


def create_initial_model_for_meshing_parameter(Meshing_calc_obj):
    """
    Creates an initial model dictionary for the meshing parameter calculation.

    Parameters
    ----------
    Meshing_calc_obj : spyro.Meshing_parameter_calculator
        The meshing calculation object.

    Returns
    -------
    model_dictionary : dict
        Dictionary containing the initial model dictionary to be later incremented.
    """
    dimension = Meshing_calc_obj.dimension
    if dimension == 2:
        model_dictionary = create_initial_model_for_meshing_parameter_2D(Meshing_calc_obj)
    elif dimension == 3:
        model_dictionary = create_initial_model_for_meshing_parameter_3D(Meshing_calc_obj)
    else:
        raise ValueError("Dimension is not 2 or 3")
    
    if Meshing_calc_obj.equation_type == "isotropic_elastic" and dimension == 2:
        model_dictionary = add_elastic_to_dictionary(model_dictionary)

    return model_dictionary


def add_elastic_to_dictionary(dictionary):
    model_dictionary = dictionary
    dictionary["absorving_boundary_conditions"]["damping_type"] = "local"
    dictionary["absorving_boundary_conditions"]["pad"] = 0.0

    return model_dictionary


def create_initial_model_for_meshing_parameter_2D(Meshing_calc_obj):
    """
    Creates an initial model dictionary for the meshing parameter calculation in 2D.

    Parameters
    ----------
    Meshing_calc_obj : spyro.Meshing_parameter_calculator
        The meshing calculation object.

    Returns
    -------
    model_dictionary : dict
        Dictionary containing the initial model dictionary to be later incremented.
    """
    velocity_profile_type = Meshing_calc_obj.velocity_profile_type
    if velocity_profile_type == "homogeneous":
        return create_initial_model_for_meshing_parameter_2D_homogeneous(
            Meshing_calc_obj
        )
    elif velocity_profile_type == "heterogeneous":
        return create_initial_model_for_meshing_parameter_2D_heterogeneous(Meshing_calc_obj)
    else:
        raise ValueError(
            "Velocity profile type is not homogeneous or heterogeneous"
        )


def create_initial_model_for_meshing_parameter_2D_heterogeneous(Meshing_calc_obj):
    """
    Creates an initial model dictionary for the meshing parameter calculation in 2D with a heterogeneous velocity model.

    Parameters
    ----------
    Meshing_calc_obj : spyro.Meshing_parameter_calculator
        The meshing calculation object.
    
    Returns
    -------
    model_dictionary : dict
        Dictionary containing the initial model dictionary.
    """
    dimension = 2
    c_value = Meshing_calc_obj.minimum_velocity
    frequency = Meshing_calc_obj.source_frequency
    cells_per_wavelength = Meshing_calc_obj.cpw_initial

    method = Meshing_calc_obj.FEM_method_to_evaluate
    degree = Meshing_calc_obj.desired_degree
    reduced = Meshing_calc_obj.reduced_obj_for_testing

    # Domain calculations
    lbda = c_value / frequency
    pad = lbda

    parameters = Meshing_calc_obj.parameters_dictionary
    length_z = parameters["length_z"]
    length_x = parameters["length_x"]

    # Source and receiver calculations
    source_z = -0.3
    source_x = 3.0
    source_locations = [(source_z, source_x)]

    # Receiver calculations
    receiver_bin_center1 = 2000.0/1000
    receiver_bin_center2 = 10000.0/1000
    receiver_quantity = 500

    bin1_startZ = source_z
    bin1_endZ = source_z
    bin1_startX = source_x + receiver_bin_center1
    bin1_endX = source_x + receiver_bin_center2

    receiver_locations = spyro.create_transect(
        (bin1_startZ, bin1_startX),
        (bin1_endZ, bin1_endX),
        receiver_quantity
    )

    # Time axis calculations
    tmin = 1.0 / frequency
    final_time = 7.5

    variables = {
        "method": method,
        "degree": degree,
        "dimension": dimension,
        "Lz": length_z,
        "Lx": length_x,
        "Ly": 0.0,
        "cells_per_wavelength": cells_per_wavelength,
        "pad": pad,
        "source_locations": source_locations,
        "frequency": frequency,
        "receiver_locations": receiver_locations,
        "final_time": final_time,
        "dt": 0.0001,
    }

    model_dictionary = build_on_top_of_base_dictionary(variables)

    model_dictionary["synthetic_data"] = {
        "real_velocity_file": Meshing_calc_obj.velocity_model_file_name,
    }

    return model_dictionary


def create_initial_model_for_meshing_parameter_3D(Meshing_calc_obj):
    """
    Creates an initial model dictionary for the meshing parameter calculation in 3D.

    Parameters
    ----------
    Meshing_calc_obj : spyro.Meshing_parameter_calculator
        The meshing calculation object.

    Returns
    -------
    model_dictionary : dict
        Dictionary containing the initial model dictionary.
    """
    velocity_profile_type = Meshing_calc_obj.velocity_profile_type
    if velocity_profile_type == "homogeneous":
        raise NotImplementedError("Not yet implemented")
        # return create_initial_model_for_meshing_parameter_3D_homogeneous(Meshing_calc_obj)
    elif velocity_profile_type == "heterogeneous":
        raise NotImplementedError("Not yet implemented")
        # return create_initial_model_for_meshing_parameter_3D_heterogeneous(Meshing_calc_obj)
    else:
        raise ValueError(
            "Velocity profile type is not homogeneous or heterogeneous"
        )


def create_initial_model_for_meshing_parameter_2D_homogeneous(Meshing_calc_obj):
    """
    Creates an initial model dictionary for the meshing parameter calculation in 2D with a homogeneous velocity model.

    Parameters
    ----------
    Meshing_calc_obj : spyro.Meshing_parameter_calculator
        The meshing calculation object.

    Returns
    -------
    model_dictionary : dict
        Dictionary containing the initial model dictionary.
    """
    dimension = 2
    c_value = Meshing_calc_obj.minimum_velocity
    frequency = Meshing_calc_obj.source_frequency
    cells_per_wavelength = Meshing_calc_obj.cpw_initial

    method = Meshing_calc_obj.FEM_method_to_evaluate
    degree = Meshing_calc_obj.desired_degree
    reduced = Meshing_calc_obj.reduced_obj_for_testing

    if c_value > 500:
        warnings.warn("Velocity in meters per second")

    # Domain calculations
    lbda = c_value / frequency
    Lz = 40 * lbda
    Lx = 30 * lbda
    Ly = 0.0
    pad = lbda

    # Source and receiver calculations
    source_z = -Lz / 2.0
    source_x = Lx / 2.0
    source_locations = [(source_z, source_x)]

    receiver_bin_center1 = 10 * lbda  # 20*lbda
    receiver_bin_width = 5 * lbda  # 15*lbda
    if reduced is True:
        receiver_quantity = 4
    else:
        receiver_quantity = 36  # 2500 # 50 squared

    bin1_startZ = source_z + receiver_bin_center1 - receiver_bin_width / 2.0
    bin1_endZ = source_z + receiver_bin_center1 + receiver_bin_width / 2.0
    bin1_startX = source_x - receiver_bin_width / 2.0
    bin1_endX = source_x + receiver_bin_width / 2.0

    receiver_locations = spyro.create_2d_grid(
        bin1_startZ,
        bin1_endZ,
        bin1_startX,
        bin1_endX,
        int(np.sqrt(receiver_quantity)),
    )

    # Time axis calculations
    tmin = 1.0 / frequency
    final_time = 20 * tmin  # Should be 35

    variables = {
        "method": method,
        "degree": degree,
        "dimension": dimension,
        "Lz": Lz,
        "Lx": Lx,
        "Ly": Ly,
        "cells_per_wavelength": cells_per_wavelength,
        "pad": pad,
        "source_locations": source_locations,
        "frequency": frequency,
        "receiver_locations": receiver_locations,
        "final_time": final_time,
        "dt": 0.0005,
    }

    model_dictionary = build_on_top_of_base_dictionary(variables)

    return model_dictionary
