import numpy as np
import spyro
import warnings


def build_on_top_of_base_dictionary(variables):
    if variables["method"] == "mass_lumped_triangle":
        mesh_type = "SeismicMesh"
    elif variables["method"] == "spectral_quadrilateral":
        mesh_type = "firedrake_mesh"
    model_dictionary = {}
    model_dictionary["options"] = {
        "method": variables["method"],
        "degree": variables["degree"],
        "dimension": variables["dimension"],
        "automatic_adjoint": False,
    }
    model_dictionary["parallelism"] = {
        "type": "automatic",
    }
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
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }
    model_dictionary["visualization"] = {
        "forward_output": True,
        "output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    return model_dictionary


def create_initial_model_for_meshing_parameter(Meshing_calc_obj):
    dimension = Meshing_calc_obj.dimension
    if dimension == 2:
        return create_initial_model_for_meshing_parameter_2D(Meshing_calc_obj)
    elif dimension == 3:
        return create_initial_model_for_meshing_parameter_3D(Meshing_calc_obj)
    else:
        raise ValueError("Dimension is not 2 or 3")


def create_initial_model_for_meshing_parameter_2D(Meshing_calc_obj):
    velocity_profile_type = Meshing_calc_obj.velocity_profile_type
    if velocity_profile_type == "homogeneous":
        return create_initial_model_for_meshing_parameter_2D_homogeneous(Meshing_calc_obj)  
    elif velocity_profile_type == "heterogeneous":
        raise NotImplementedError("Not yet implemented")
        # return create_initial_model_for_meshing_parameter_2D_heterogeneous(Meshing_calc_obj)
    else:
        raise ValueError("Velocity profile type is not homogeneous or heterogeneous")


def create_initial_model_for_meshing_parameter_3D(Meshing_calc_obj):
    velocity_profile_type = Meshing_calc_obj.velocity_profile_type
    if velocity_profile_type == "homogeneous":
        raise NotImplementedError("Not yet implemented")
        # return create_initial_model_for_meshing_parameter_3D_homogeneous(Meshing_calc_obj)  
    elif velocity_profile_type == "heterogeneous":
        raise NotImplementedError("Not yet implemented")
        # return create_initial_model_for_meshing_parameter_3D_heterogeneous(Meshing_calc_obj)
    else:
        raise ValueError("Velocity profile type is not homogeneous or heterogeneous")


def create_initial_model_for_meshing_parameter_2D_homogeneous(Meshing_calc_obj):
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
    lbda = c_value/frequency
    Lz = 40*lbda
    Lx = 30*lbda
    Ly = 0.0
    pad = lbda

    # Source and receiver calculations
    source_z = -Lz/2.0
    source_x = Lx/2.0
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
        bin1_startZ, bin1_endZ, bin1_startX, bin1_endX, int(np.sqrt(receiver_quantity))
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
