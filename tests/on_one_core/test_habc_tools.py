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
        Geometry of the finite element. Options: 'T' for triangles/tetrahedra or
        'Q' for quadrilaterals/hexahedra
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D
    layer_shape : `str`
        Shape of the absorbing layer, either 'rectangular' or 'hypershape'
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model
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
        "source_locations": ([(-0.5, 0.25)] if dimension == 2  # (0.5 * Lz, 0.25 * Lx)
                             else [(-0.5, 0.25, 0.5)]),  # (0.5 * Lz, 0.25 * Lx, 0.5 * Ly)
        "frequency": 5.,  # in Hz
        "receiver_locations": ([(-Lz, 0.), (-Lz, Lx), (0., 0.), (0., Lx)]
                               if dimension == 2
                               else [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                                     (0., 0., 0), (0., Lx, 0.),
                                     (-Lz, 0., Ly), (-Lz, Lx, Ly),
                                     (0., 0., Ly), (0., Lx, Ly)])
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
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km
    f_est : `float`, optional
        Factor for the stabilizing term in Eikonal Eq.
    dimension : `int`
        Dimension of the model (2 or 3)

    Returns
    -------
    wave_obj : `habc.HABCLayer`
        An instance of the HABCLayer class
    """

    # ============ MESH FEATURES ============

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    wave_obj = AcousticWave(dictionary=dictionary)

    # Mesh
    wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = conditional(wave_obj.mesh_x < 0.5, 3.0, 1.5)
    wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    wave_obj.mesh_ops.preamble_mesh_operations(wave_obj, f_est=f_est)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=wave_obj.path_save + "preamble/MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    wave_obj.layer_ops.critical_boundary_points(wave_obj)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=wave_obj.path_save + "preamble/EIK_")

    return wave_obj


def run_tools(wave_obj, method_extend, n_root=1):
    """Test the HABC tools in a bi-material model.

    Parameters
    ----------
    wave_obj : `habc.HABCLayer`
        An instance of the HABCLayer class
    method_extend : `str`
        Method to extend the velocity profile. Options: 'point_cloud' or 'nearest_point'
    n_root : `int`, optional
        n-th Root selected as the size of the absorbing layer. Default is 1

    Returns
    -------
    None
    """

    # Determining layer size
    wave_obj.layer_ops.layer_size_criterion(wave_obj.mesh_parameters.lmin, n_root=n_root)

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Creating mesh with absorbing layer
    wave_obj.layer_ops.create_mesh_with_layer(wave_obj)

    # Updating velocity model
    wave_obj.layer_ops.velocity_abc(wave_obj, method=method_extend)

    # Estimating computational resource usage
    ele_str = "Q" if wave_obj.mesh_ops.quadrilateral else "T"
    ext_str = "CLOUD" if method_extend == "point_cloud" else "NEARP"
    name_cost = wave_obj.layer_ops.path_case_abc + ele_str + "_" + ext_str + "_"
    comp_cost("tfin", tRef=tRef, user_name=name_cost)

    # Expected values
    expect_xmhalf = 3.
    expect_xphalf = 1.5
    tolerance = 0.03  # 3% tolerance

    # Create layer mask
    method_element = "DQ" if wave_obj.mesh_ops.quadrilateral else "DG"
    V = create_function_space(wave_obj.mesh, method_element, 0)

    # Clipping coordinates to the layer domain
    domain_layer = wave_obj.layer_ops.abc_domain_dimensions(full_hyp=False)
    layer_mask = layer_mask_field(
        wave_obj.mesh_ops.domain_dim, wave_obj.mesh, wave_obj.dimension,
        wave_obj.mesh_ops.get_spatial_coordinates_abc(wave_obj.mesh, domain_layer),
        V, type_marker='mask', name_mask='test_mask')

    # Extracting nodes from the layer field
    mask_nodes = wave_obj.mesh_ops.extract_node_positions(wave_obj.mesh, V,
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
    layer_cloud_xlt = point_cloud_field(wave_obj.mesh, pts_layer_xlt, wave_obj.c,
                                        wave_obj.mesh_parameters.tol)

    layer_cloud_xge = point_cloud_field(wave_obj.mesh, pts_layer_xge, wave_obj.c,
                                        wave_obj.mesh_parameters.tol)
    original_cloud_xlt = point_cloud_field(wave_obj.mesh, pts_original_xlt, wave_obj.c,
                                           wave_obj.mesh_parameters.tol)
    original_cloud_xge = point_cloud_field(wave_obj.mesh, pts_original_xge, wave_obj.c,
                                           wave_obj.mesh_parameters.tol)
    # Verify cloud values
    met_str = f"HABC Tools {ele_str}-{ext_str}" + \
        f" {wave_obj.layer_ops.case_abc[:-4]} {wave_obj.dimension}D. "
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
    wave_obj.layer_ops.rename_folder_habc()


@mark.parametrize("element_geometry, dimension",
                  [("T", 2),
                   ("Q", 2),
                   param("T", 3, marks=mark.slow),
                   param("Q", 3, marks=mark.slow)])
def test_habc_tools(element_geometry, dimension):
    """Test of HABC tools for 2D and 3D case.

    Parameters
    ----------
    element_geometry : `str`
        Type of finite element. 'T' for triangles or 'Q' for quadrilaterals
    dimension : `int`
        Dimension of the model (2 or 3)

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
    wave_obj = preamble_tools(dictionary, edge_length, f_est, dimension)

    # ============ HABC TOOLS ============

    try:
        # Testing tools for the HABC implementation
        for method_extend in ["point_cloud", "nearest_point"]:
            # Method to extend the velocity profile in the absorbing layer
            print("\n" + 30 * "=" + f"\nTesting Method: {method_extend}\n"
                  + 30 * "=", flush=True)

            # Determining the case for the folder name
            str_id = element_geometry + ("CL" if method_extend == "point_cloud" else "NP")
            wave_obj.case_abc, wave_obj.path_save, wave_obj.path_case_abc = \
                wave_obj.layer_ops.identify_abc_layer_case(
                    output_folder=wave_obj.output_folder+f"/ht_test{dimension}d{str_id}")

            # Running the HABC tools
            run_tools(wave_obj, method_extend)

    except ConvergenceError as e:
        fail(f"Checking HABC tools {dimension}D raised an exception: {str(e)}")
