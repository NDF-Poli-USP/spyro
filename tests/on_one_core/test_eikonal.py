from pytest import fail, mark, param
from firedrake import conditional, ConvergenceError
from numpy import isclose
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.utils.cost import comp_cost


def wave_dict(element_geometry, dimension, degree_eikonal, element_type):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element. 'T' for triangles or 'Q' for quadrilaterals
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D
    degree_eikonal : `int`
        Finite element order for the Eikonal equation. Should be 1 or 2
    element_type : `str`
        Finite element type. 'consistent' or 'underintegrated'

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

    # Define the domain size without the PML or AL. Here we'll assume a domain
    # with a width and depth of 1 km, and a thickness of 1 km for the 3D case.
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
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
    }

    # Define parameters for visualization
    str_ele = element_geometry + ("C" if element_type == 'consistent' else "U")
    dictionary["visualization"] = {  # Output folder
        "output_folder": f"output/eikonal_test{dimension}d/eik_test{dimension}d" + str_ele
    }

    return dictionary


def eikonal_analysis(dictionary, edge_length, f_est, element_type):
    """Run the the Eikonal analysis.

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km
    f_est : `float`
        Factor for the stabilizing term in Eikonal Eq
    element_type : `string`
        Finite element type. 'consistent' or 'underintegrated'

    Returns
    -------
    min_eik : `float`
        Minimum Eikonal value in miliseconds
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
    Wave_obj.mesh_ops.preamble_mesh_operations(
        Wave_obj, ele_type_eik=element_type, f_est=f_est)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=Wave_obj.path_save + "MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    Wave_obj.layer_ops.critical_boundary_points(Wave_obj)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=Wave_obj.path_save + "EIK_")

    # Extracting  minimum Eikonal
    min_eik = 1e3 * Wave_obj.layer_ops.eik_bnd[0][2]

    return min_eik


@mark.parametrize("element_geometry, dimension, element_type",
                  [("T", 2, "consistent"),
                   ("T", 2, "underintegrated"),
                   ("Q", 2, "consistent"),
                   ("Q", 2, "underintegrated"),
                   ("T", 3, "consistent"),
                   ("T", 3, "underintegrated"),
                   ("Q", 3, "consistent"),
                   ("Q", 3, "underintegrated")])
def test_eikonal(element_geometry, dimension, element_type):
    """Testing of eikonal for 2D and 3D case in Fig. 8 of Salas et al (2022).

    Parameters
    ----------
    element_geometry : `str`
        Type of finite element. 'T' for triangles or 'Q' for quadrilaterals
    dimension : `int`
        Dimension of the model (2 or 3)
    element_type : `str`
        Finite element type. 'consistent' or 'underintegrated'

    Returns
    -------
    None

    ==============================
    Eikonal for 2D model Δx = 100m
    ==============================
    eik_min = 83.333 ms (Theoretical value)

    Consistent elements
    -------------------
    f_est  T-ele   Q-ele
     0.01 66.836  65.002
     0.02 73.308  75.811
     0.03 77.178  79.845
     0.04 79.680  82.101
     0.05 81.498  83.744*
     0.06 82.942* 85.118
     0.07 84.160  86.345
     0.08 85.233  87.480

    Underintegrated elements
    ------------------------
    p = 2         p = 4
    f_est  T-ele  f_est  Q-ele
     0.07 82.630*  0.03 84.245 *
     0.08 84.272   0.04 85.593
     0.09 85.654   0.05 86.887

    ==============================
    Eikonal for 3D model Δx = 150m
    ==============================
    eik_min = 83.333 ms (Theoretical value)

    Consistent elements
    -------------------
    f_est  T-ele   Q-ele
     0.02  --/--  69.442
     0.03 76.777  70.974
     0.04 79.409  73.179
     0.05 82.273* 75.766
     0.06 85.347  78.548
     0.07 88.562  81.431*
     0.08 91.876  84.377

    Underintegrated elements
    ------------------------
    p = 2         p = 3
    f_est  T-ele  f_est   Q-ele
     0.07 85.178*  0.03 78.838
     0.08 87.990   0.04 80.940
     0.09 90.933   0.05 83.130*
     0.10 93.988   0.06 85.408

    p = 1-T-full integration
    f_est  T-ele
     0.06 96.300
     0.07 96.845
     0.08 97.277

    p = 3-T-full integration
    f_est   T-ele
     0.03  96.223
     0.04 101.550
     0.05 107.164
    """

    print("\n" + 60 * "=" + f"\nTesting Eikonal with {element_geometry}-{element_type} "
          + f"elements for {dimension}D case\n" + 60 * "=", flush=True)

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1 if dimension == 2 else 0.15

    # Eikonal degree and factor for the stabilizing term in Eikonal equation
    if dimension == 2:
        atol = 5e-3 if element_type == 'consistent' else 1.5e-2
        if element_geometry == "T":
            p_eik = 2 if element_type == 'consistent' else 2
            f_est = 0.06 if element_type == 'consistent' else 0.07

        else:
            p_eik = 2 if element_type == 'consistent' else 4
            f_est = 0.05 if element_type == 'consistent' else 0.03

    if dimension == 3:
        atol = 3e-2 if element_type == 'consistent' else 2.5e-2
        if element_geometry == "T":
            p_eik = 2 if element_type == 'consistent' else 2
            f_est = 0.05 if element_type == 'consistent' else 0.07
        else:
            p_eik = 2 if element_type == 'consistent' else 3
            f_est = 0.07 if element_type == 'consistent' else 0.05

    # Get simulation parameters
    print(f"\nMesh Size: {1e3 * edge_length:.4f} m", flush=True)
    print(f"Element Geometry: {element_geometry}", flush=True)
    print(f"Element Type: {element_type}", flush=True)
    print(f"Eikonal Degree: {p_eik}", flush=True)
    print(f"Eikonal Stabilizing Factor: {f_est:.2f}", flush=True)

    try:

        # ============ MESH AND EIKONAL ============

        # Create dictionary with parameters for the model
        dictionary = wave_dict(element_geometry, dimension, p_eik, element_type)

        # Creating mesh and performing eikonal analysis
        min_eik = round(eikonal_analysis(dictionary, edge_length, f_est, element_type), 3)

        thr_val = 83.333  # in ms
        assert isclose(min_eik / thr_val, 1., atol=atol), \
            f"✗ Minimum Eikonal {dimension}D Element-{element_geometry}-" + \
            f"{element_type} → Expected value {thr_val}, got {min_eik:.3f}"
        print(f"✓ Minimum Eikonal {dimension}D Element-{element_geometry}-{element_type} "
              f" Verified: expected {thr_val}, got {min_eik:.3f}", flush=True)

    except ConvergenceError as e:
        fail(f"Checking Eikonal {dimension}D Element-{element_geometry}-"
             f"{element_type} raised an exception: {str(e)}")
