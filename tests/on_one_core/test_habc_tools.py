import pytest
import warnings
import numpy as np
import firedrake as fire
import spyro.habc.habc as habc
from spyro.domains.space import create_function_space
from spyro.tools.habc_tools import layer_mask_field, point_cloud_field
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def wave_dict(element_type, dimension, degree_layer):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    element_type : `str`
        Type of finite element. 'T' for triangles or 'Q' for quadrilaterals
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": element_type,
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        # Default is lumped "method":"MLT"
        # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
        # You can either specify a cell_type+variant or a method
        # accepted_variants = ["lumped", "equispaced", "DG"]
        "degree": 4 if dimension == 2 else 3,  # p<=4 for 2D and p<=3 for 3D
        "dimension": dimension,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "automatic",
    }

    # Define the domain size without the PML or AL.
    if dimension == 2:
        Lz, Lx, Ly = [1., 1., 0.]
    elif dimension == 3:
        Lz, Lx, Ly = [1., 1., 1.]  # in km
    dictionary["mesh"] = {
        "length_z": Lz,  # depth in km - always positive
        "length_x": Lx,  # width in km - always positive
        "length_y": Ly,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": ([(-0.5, 0.25)] if dimension == 2
                             else [(-0.5, 0.25, 0.5)]),
        "frequency": 5.,  # in Hz
        "delay": 1.5,
        "receiver_locations": ([(-Lz, 0.), (-Lz, Lx), (0., 0.), (0., Lx)]
                               if dimension == 2
                               else [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                                     (0., 0., 0), (0., Lx, 0.),
                                     (-Lz, 0., Ly), (-Lz, Lx, Ly),
                                     (0., 0., Ly), (0., Lx, Ly)])
    }

    # Simulate for 1. seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": 0.001,  # timestep size in seconds
        "amplitude": 1.,  # The Ricker has an amplitude of 1.
        "output_frequency": 100,  # How frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # How frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": "rectangular",  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
        "degree_type": "real",  # Options: real or integer
        "habc_reference_freq": "source",  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": False,  # If True, the infinite model is created
    }

    return dictionary


def preamble_tools(dictionary, edge_length, f_est, dimension):
    '''
    Run the infinite model and the Eikonal analysis

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km
    f_est : `float`, optional
        Factor for the stabilizing term in Eikonal Eq.
    dimension : `int`
        Dimension of the model (2 or 3)

    Returns
    -------
    wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    '''

    # ============ MESH FEATURES ============

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    wave_obj = habc.HABC_Wave(dictionary=dictionary,
                              output_folder=f"output/test_habc_tools{dimension}d")

    # Mesh
    wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = fire.conditional(wave_obj.mesh_x < 0.5, 3.0, 1.5)
    wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    wave_obj.preamble_mesh_operations(f_est=f_est)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=wave_obj.path_save + "preamble/MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    wave_obj.critical_boundary_points()

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=wave_obj.path_save + "preamble/EIK_")

    return wave_obj


def run_tools(wave_obj, n_root=1):
    '''
    Apply the HABC to the model in Fig. 8 of Salas et al. (2022).

    Parameters
    ----------
    wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    n_root : `int`, optional
        n-th Root selected as the size of the absorbing layer. Default is 1

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    wave_obj.identify_habc_case(output_folder=f"output/modal_test{wave_obj.dimension}d")

    # Determining layer size
    wave_obj.size_habc_criterion(n_root=n_root)

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Creating mesh with absorbing layer
    wave_obj.create_mesh_habc()

    # Updating velocity model
    wave_obj.velocity_habc()

    # Estimating computational resource usage
    ele_str = "Q" if wave_obj.quadrilateral else "T"
    name_cost = wave_obj.path_case_habc + ele_str + "_"
    comp_cost("tfin", tRef=tRef, user_name=name_cost)

    # Expected values
    expect_xmhalf = 3.
    expect_xphalf = 1.5
    tolerance = 0.03  # 3% tolerance

    # Create layer mask
    method_element = "DQ" if wave_obj.quadrilateral else "DG"
    V = create_function_space(wave_obj.mesh, method_element, 0)
    layer_mask = layer_mask_field(wave_obj.mesh_ops.domain_dim, wave_obj.mesh,
                                  wave_obj.dimension,
                                  wave_obj.get_spatial_coordinates_habc(), V,
                                  type_marker='mask', name_mask='test_mask')

    # Extracting nodes from the layer field
    mask_nodes = wave_obj.mesh_ops.extract_node_positions(wave_obj.mesh, V,
                                                          output_type="array")
    indlay_nodes = np.where(layer_mask.dat.data_with_halos == 1.)[0]
    pts_layer = mask_nodes[indlay_nodes]  # Inside layer
    pts_layer_xlt = pts_layer[pts_layer[:, 1] < 0.5]
    pts_layer_xge = pts_layer[pts_layer[:, 1] >= 0.5]
    original_nodes = np.where(layer_mask.dat.data_with_halos == 0.)[0]
    pts_original = mask_nodes[original_nodes]  # Inside original domain
    pts_original_xlt = pts_original[pts_original[:, 1] < 0.5]
    pts_original_xge = pts_original[pts_original[:, 1] >= 0.5]

    # Cloud fields
    layer_cloud_xlt = point_cloud_field(wave_obj.mesh, pts_layer_xlt, wave_obj.c,
                                        wave_obj.mesh_parameters.tol)

    layer_cloud_xge = point_cloud_field(wave_obj.mesh, pts_layer_xge, wave_obj.c,
                                        wave_obj.mesh_parameters.tol)
    original_cloud_xlt = point_cloud_field(wave_obj.mesh, pts_original_xlt, wave_obj.c,
                                           wave_obj.mesh_parameters.tol)
    original_cloud_xge = point_cloud_field(wave_obj.mesh, pts_original_xge, wave_obj.c,
                                           wave_obj.mesh_parameters.tol)
    # Verify cloud values
    met_str = f"HABC Tools {ele_str} {wave_obj.case_habc[:-4]} {wave_obj.dimension}D. "
    expected_values = [expect_xmhalf, expect_xphalf, expect_xmhalf, expect_xphalf]
    mean_val = [layer_cloud_xlt.dat.data_with_halos.mean(),
                layer_cloud_xge.dat.data_with_halos.mean(),
                original_cloud_xlt.dat.data_with_halos.mean(),
                original_cloud_xge.dat.data_with_halos.mean()]
    region_names = ["Layer x<0.5", "Layer x>=0.5", "Original x<0.5", "Original x>=0.5"]

    for region, exp_value, mean_val in zip(region_names, expected_values, mean_val):
        cmp_str = f"{region}: Expected {exp_value:.5f}, got = {mean_val:.5f}"
        assert np.isclose(mean_val / exp_value, 1., atol=tolerance), \
            "✗ " + met_str + "  → " + cmp_str
        print("✓ " + met_str + "Verified: " + cmp_str, flush=True)


@pytest.mark.parametrize("element_type", ["T", "Q"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_habc_tools(element_type, dimension):
    """Test of HABC tools for 2D case.

    Parameters
    ----------
    element_type : `str`
        Type of finite element. 'T' for triangles or 'Q' for quadrilaterals

    Returns
    -------
    None
    """

    print("\n" + 85 * "=" + f"\nTesting HABC Tools with {element_type} "
          + f"elements for {dimension}D case\n" + 85 * "=", flush=True)

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1 if dimension == 2 else 0.15

    # Factor for the stabilizing term in Eikonal equation
    f_est = 0.06 if dimension == 2 else 0.08

    # Get simulation parameters
    print(f"\nMesh Size: {1e3 * edge_length:.3f} m", flush=True)
    print(f"Eikonal Stabilizing Factor: {f_est:.2f}", flush=True)

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    degree_layer = 3. if element_type == "T" else None

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict(element_type, dimension, degree_layer)

    # ============ MESH AND EIKONAL ============

    # Creating mesh and performing eikonal analysis
    wave_obj = preamble_tools(dictionary, edge_length, f_est, dimension)

    # ============ HABC TOOLS ============

    try:
        # Testing tools for the HABC implementation
        run_tools(wave_obj)

        # Renaming the folder if degree_layer is modified
        wave_obj.rename_folder_habc()

    except fire.ConvergenceError as e:
        pytest.fail(f"Checking HABC tools {dimension}D raised an exception: {str(e)}")


'''
=================================================================
DATA FOR 2D MODEL Δx = 100m
---------------------------

*EIKONAL
eik_min = 83.333 ms
f_est  eik[ms]
 0.01  66.836
 0.02  73.308
 0.03  77.178
 0.04  79.680
 0.05  81.498
 0.06  82.942*
 0.07  84.160
 0.08  85.233

=================================================================
DATA FOR 3D MODEL Δx = 150m
---------------------------

*EIKONAL
eik_min = 83.333 ms
f_est  eik[ms]
 0.02  69.442
 0.03  70.974
 0.04  73.179
 0.05  75.766
 0.06  78.548
 0.07  81.431
 0.08  84.377*
 0.09  87.376
'''
