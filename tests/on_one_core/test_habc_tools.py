"""Unit tests for the tools used in the HABC scheme implemented in spyro.tools.habc_tools.

This test verifies the correct functioning of the tools used in the implementation
of the HABC in spyro. It checks that the absorbing layer is correctly created and
that the velocity profile is correctly extended in the layer using both the point
cloud and nearest point methods. The test is performed for both 2D and 3D cases,
and for both triangular and quadrilateral elements. The expected values are based on
the theoretical velocity model and the geometry of the absorbing layer. The test also
measures the computational cost of the operations.
"""
from pytest import fail, mark, param
from firedrake import conditional, ConvergenceError
from numpy import isclose, where
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.domains.space import create_function_space
from spyro.tools.habc_tools import layer_mask_field, point_cloud_field
from spyro.utils.cost import comp_cost


def wave_dict(element_geometry, dimension, layer_shape, degree_layer):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.
    layer_shape : `str`.
        Shape of the absorbing layer, either "rectangular or "hypershape".
    degree_layer : `int` or `float` or `None`
        Degree of the hypershape layer, if applicable. If `None`, it is not used.
    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    """

    dictionary = {}
    # Define options for the model. We specify the cell type, variant,
    # degree, dimension and analysis type.
    dictionary["options"] = {
        "cell_type": element_geometry,  # Options: tri/tetra(T) or quad/hexa(Q)
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        "degree": 4 if dimension == 2 else 3,  # p <= 4 for 2D and p <= 3 for 3D
        "dimension": dimension,  # Model dimension
        "analysis": "eikonal",  # Options: transient, modal or eikonal
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "automatic",
    }

    # Define the domain size without the PML or AL. Here we'll assume a domain
    # with a width and depth of 1 km, and a thickness of 1 km for the 3D case.
    if dimension == 2:
        length_z, length_x, length_y = [1., 1., 0.]
    elif dimension == 3:
        length_z, length_x, length_y = [1., 1., 1.]  # in km
    dictionary["mesh"] = {
        "length_z": length_z,  # depth in km - always positive
        "length_x": length_x,  # width in km - always positive
        "length_y": length_y,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_locations": ([(-length_z / 2., length_x / 4.)] if dimension == 2
                             else [(-length_z / 2., length_x / 4., length_y / 2.)]),
        "frequency": 5.,  # in Hz
        "receiver_locations": ([(-length_z, 0.),
                                (-length_z, length_x),
                                (0., 0.), (0., length_x)]
                               if dimension == 2
                               else [(-length_z, 0., 0.),
                                     (-length_z, length_x, 0.),
                                     (0., 0., 0),
                                     (0., length_x, 0.),
                                     (-length_z, 0., length_y),
                                     (-length_z, length_x, length_y),
                                     (0., 0., length_y),
                                     (0., length_x, length_y)])
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "output_folder": f"output/habc_tools_test{dimension}d",  # Output folder
    }

    return dictionary


def preamble_tools(dictionary, edge_length, f_est, dimension):
    """Build the mesh and run the Eikonal analysis.

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    edge_length : `float`
        Mesh size in km.
    f_est : `float`, optional
        Factor for the stabilizing term in Eikonal Eq.
    dimension : `int`
        Dimension of the model (2 or 3).

    Returns
    -------
    Wave_obj : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """

    # ============ MESH FEATURES ============

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    Wave_obj = AcousticWave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.mesh_ops.preamble_mesh_operations(Wave_obj, f_est=f_est)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=Wave_obj.path_save + "preamble/MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    Wave_obj.layer_ops.critical_boundary_points(Wave_obj)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=Wave_obj.path_save + "preamble/EIK_")

    return Wave_obj


def run_tools(Wave_obj, method_extend, n_root=1):
    """Test the HABC tools in a bi-material model.

    Parameters
    ----------
    Wave_obj : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    method_extend : `str`
        Method to extend the velocity profile. Options: "point_cloud" or "nearest_point".
    n_root : `int`, optional
        n-th Root selected as the size of the absorbing layer. Default is 1.

    Returns
    -------
    None
    """

    # Determining layer size
    Wave_obj.layer_ops.layer_size_criterion(Wave_obj.mesh_parameters.lmin, n_root=n_root)

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Creating mesh with absorbing layer
    Wave_obj.layer_ops.create_mesh_with_layer(Wave_obj)

    # Updating velocity model
    Wave_obj.layer_ops.velocity_abc(Wave_obj, method=method_extend)

    # Estimating computational resource usage
    ele_str = "Q" if Wave_obj.mesh_ops.quadrilateral else "T"
    ext_str = "CLOUD" if method_extend == "point_cloud" else "NEARP"
    name_cost = Wave_obj.layer_ops.path_case_abc + ele_str + "_" + ext_str + "_"
    comp_cost("tfin", tRef=tRef, user_name=name_cost)

    # Expected values
    expect_xmhalf = 3.
    expect_xphalf = 1.5
    tolerance = 0.03  # 3% tolerance

    # Create layer mask
    method_element = "DQ" if Wave_obj.mesh_ops.quadrilateral else "DG"
    V = create_function_space(Wave_obj.mesh, method_element, 0)

    # Clipping coordinates to the layer domain
    domain_layer = Wave_obj.layer_ops.abc_domain_dimensions(full_hyp=False)
    layer_mask = layer_mask_field(
        Wave_obj.mesh_ops.domain_dim, Wave_obj.mesh, Wave_obj.dimension,
        Wave_obj.mesh_ops.get_spatial_coordinates_abc(Wave_obj.mesh, domain_layer),
        V, type_marker='mask', name_mask='test_mask')

    # Extracting nodes from the layer field
    mask_nodes = Wave_obj.mesh_ops.extract_node_positions(Wave_obj.mesh, V,
                                                          output_type="array")
    indlay_nodes = where(layer_mask.dat.data_with_halos == 1.)[0]
    pts_layer = mask_nodes[indlay_nodes]  # Inside layer
    pts_layer_xlt = pts_layer[pts_layer[:, 1] < 0.5]
    pts_layer_xge = pts_layer[pts_layer[:, 1] >= 0.5]
    original_nodes = where(layer_mask.dat.data_with_halos == 0.)[0]
    pts_original = mask_nodes[original_nodes]  # Inside original domain
    pts_original_xlt = pts_original[pts_original[:, 1] < 0.5]
    pts_original_xge = pts_original[pts_original[:, 1] >= 0.5]

    # Cloud fields
    layer_cloud_xlt = point_cloud_field(Wave_obj.mesh, pts_layer_xlt, Wave_obj.c,
                                        Wave_obj.mesh_parameters.tol)

    layer_cloud_xge = point_cloud_field(Wave_obj.mesh, pts_layer_xge, Wave_obj.c,
                                        Wave_obj.mesh_parameters.tol)
    original_cloud_xlt = point_cloud_field(Wave_obj.mesh, pts_original_xlt, Wave_obj.c,
                                           Wave_obj.mesh_parameters.tol)
    original_cloud_xge = point_cloud_field(Wave_obj.mesh, pts_original_xge, Wave_obj.c,
                                           Wave_obj.mesh_parameters.tol)
    # Verify cloud values
    met_str = f"HABC Tools {ele_str}-{ext_str}" + \
        f" {Wave_obj.layer_ops.case_abc[:-4]} {Wave_obj.dimension}D. "
    expected_values = [expect_xmhalf, expect_xphalf, expect_xmhalf, expect_xphalf]
    mean_val = [layer_cloud_xlt.dat.data_with_halos.mean(),
                layer_cloud_xge.dat.data_with_halos.mean(),
                original_cloud_xlt.dat.data_with_halos.mean(),
                original_cloud_xge.dat.data_with_halos.mean()]
    region_names = ["Layer x<0.5", "Layer x>=0.5", "Original x<0.5", "Original x>=0.5"]

    for region, exp_value, mean_val in zip(region_names, expected_values, mean_val):
        cmp_str = f"{region}: Expected {exp_value:.5f}, got = {mean_val:.5f}"
        assert isclose(mean_val / exp_value, 1., atol=tolerance), \
            "✗ " + met_str + "  → " + cmp_str
        print("✓ " + met_str + "Verified: " + cmp_str, flush=True)

    # Renaming the folder if degree_layer is modified
    Wave_obj.layer_ops.rename_folder_habc()


@pytest.mark.older_firedrake
@pytest.mark.parametrize("element_type, dimension, method_extend", [
    ("T", 2, "point_cloud"),
    ("T", 2, "nearest_point"),
    ("Q", 2, "point_cloud"),
    ("Q", 2, "nearest_point"),
    pytest.param("T", 3, "point_cloud", marks=pytest.mark.slow),
    pytest.param("T", 3, "nearest_point", marks=pytest.mark.slow),
    pytest.param("Q", 3, "point_cloud", marks=pytest.mark.slow),
    pytest.param("Q", 3, "nearest_point", marks=pytest.mark.slow)])
def test_habc_tools(element_type, dimension, method_extend):
    """Test of HABC tools for 2D and 3D case.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. Options: "T" for triangles/tetrahedra or
        "Q" for quadrilaterals/hexahedra.
    dimension : `int`
        Dimension of the model (2 or 3).

    Returns
    -------
    None

    Eikonal for 2D model Δx = 100m
    ------------------------------
    eik_min = 83.333 ms (Theoretical value)
    f_est  T-ele   Q-ele
     0.01 66.836  65.002
     0.02 73.308  75.811
     0.03 77.178  79.845
     0.04 79.680  82.101
     0.05 81.498  83.744*
     0.06 82.942* 85.118
     0.07 84.160  86.345
     0.08 85.233  87.480

    Eikonal for 3D model Δx = 150m
    ------------------------------
    eik_min = 83.333 ms (Theoretical value)
    f_est  T-ele   Q-ele
     0.02  --/--  69.442
     0.03 76.777  70.974
     0.04 79.409  73.179
     0.05 82.273* 75.766
     0.06 85.347  78.548
     0.07 88.562  81.431*
     0.08 91.876  84.377
    """

    print("\n" + 50 * "=" + f"\nTesting HABC Tools with {element_geometry} "
          + f"elements for {dimension}D case\n" + 50 * "=", flush=True)

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1 if dimension == 2 else 0.15

    # Factor for the stabilizing term in Eikonal equation
    if element_geometry == "T":
        f_est = 0.06 if dimension == 2 else 0.05
    else:
        f_est = 0.05 if dimension == 2 else 0.07

    # Get simulation parameters
    print(f"\nMesh Size: {1e3 * edge_length:.3f} m", flush=True)
    print(f"Eikonal Stabilizing Factor: {f_est:.2f}", flush=True)

    # ============ HABC PARAMETERS ============

    # Hyperellipse degrees
    if element_geometry == "T":
        layer_shape = "hypershape"
        degree_layer = 2.
    else:
        layer_shape = "rectangular"
        degree_layer = None

    # ============ INPUT DATA ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict(element_geometry, dimension, layer_shape, degree_layer)

    # ============ MESH AND EIKONAL ============

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_tools(dictionary, edge_length, f_est, dimension)

    # ============ HABC TOOLS ============

    try:
        # Testing tools for the HABC implementation
        for method_extend in ["point_cloud", "nearest_point"]:
            # Method to extend the velocity profile in the absorbing layer
            print("\n" + 30 * "=" + f"\nTesting Method: {method_extend}\n"
                  + 30 * "=", flush=True)

            # Determining the case for the folder name
            str_id = element_geometry + ("CL" if method_extend == "point_cloud" else "NP")
            Wave_obj.layer_ops.path_to_save_abc_layer_case(
                output_folder=Wave_obj.output_folder+f"/ht_test{dimension}d{str_id}")
            Wave_obj.case_abc = Wave_obj.layer_ops.case_abc
            Wave_obj.path_save = Wave_obj.layer_ops.path_save
            Wave_obj.path_case_abc = Wave_obj.layer_ops.path_case_abc

            # Running the HABC tools
            run_tools(Wave_obj, method_extend)

    except ConvergenceError as e:
        fail(f"Checking HABC tools {dimension}D raised an exception: {str(e)}")
