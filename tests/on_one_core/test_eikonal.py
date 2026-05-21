import pytest
import warnings
import firedrake as fire
import spyro.habc.eik as eik
from numpy import isclose
from os import getcwd
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.meshing.meshing_habc import HABCMesh
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict(ele_geometry, dimension, degree_eikonal, element_type):
    """Create a dictionary with parameters for the model.

    Parameters
    ----------
    ele_geometry : `str`
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
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": ele_geometry,
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

    # Define the domain size without the PML or AL. Here we'll assume a
    # 1 x 1 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
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

    # Simulate for 1. seconds.
    dictionary["time_axis"] = {
        "final_time": 1.,    # Final time for event
        "dt": 0.001,  # timestep size in seconds
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
    }

    # Define parameters for visualization
    str_ele = ele_geometry + ("C" if element_type == 'consistent' else "U")
    dictionary["visualization"] = {  # Output folder
        "output_folder": f"output/eikonal_test{dimension}d/eik_test{dimension}d" + str_ele
    }

    return dictionary


def eikonal_analysis(dictionary, edge_length, f_est, ele_type='consistent'):
    """Run the the Eikonal analysis.

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km
    f_est : `float`
        Factor for the stabilizing term in Eikonal Eq.
    ele_type : `string`, optional
        Finite element type. 'consistent' or 'underintegrated'. Default is 'consistent'

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
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.mesh_ops.preamble_mesh_operations(
        Wave_obj, ele_type_eik=ele_type, f_est=f_est)

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


def test_eik_consistent_ele_2d():
    """Testing eikonal 2D with consitent elements in Fig. 8 of Salas et al (2022).

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
    """

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1

    # Element geometry
    ele_geometry_lst = ["T", "Q"]

    # Eikonal degree
    p_eik = 2

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.06, 0.05]

    for case in range(0, 2):

        # Get simulation parameters
        ele_geometry = ele_geometry_lst[case]
        f_est = f_est_lst[case]
        print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
        print("Element type: {}".format(ele_geometry), flush=True)
        print("Eikonal Degree: {}".format(p_eik), flush=True)
        print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

        try:
            # ============ MESH AND EIKONAL ============

            # Create dictionary with parameters for the model
            dict_2d = wave_dict(ele_geometry, 2, p_eik, 'consistent')

            # Creating mesh and performing eikonal analysis
            min_eik = round(eikonal_analysis(dict_2d, edge_length, f_est), 3)

            thr_val = 83.333  # in ms
            assert isclose(min_eik / thr_val, 1., atol=5e-3), \
                f"✗ Minimum Eikonal 2D Element-{ele_geometry}-Consistent " + \
                f"→ Expected value {thr_val}, got {min_eik:.3f}"
            print(f"✓ Minimum Eikonal 2D Element-{ele_geometry}-Consistent Verified: "
                  f"expected {thr_val}, got {min_eik:.3f}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Eikonal 2D Element-{ele_geometry}"
                        f"-Consistent raised an exception: {str(e)}")


@pytest.mark.slow
def test_eik_consistent_ele_3d():
    """Testing eikonal 3D with consitent elements in Fig. 8 of Salas et al (2022).

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

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.15

    # Element geometry
    ele_geometry_lst = ["T", "Q"]

    # Eikonal degree
    p_eik = 2

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.05, 0.07]

    for case in range(0, 2):

        # Get simulation parameters
        ele_geometry = ele_geometry_lst[case]
        f_est = f_est_lst[case]
        print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
        print("Element type: {}".format(ele_geometry), flush=True)
        print("Eikonal Degree: {}".format(p_eik), flush=True)
        print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

        # ============ MESH AND EIKONAL ============
        try:

            # Create dictionary with parameters for the model
            dict_3d = wave_dict(ele_geometry, 3, p_eik, 'consistent')

            # Creating mesh and performing eikonal analysis
            min_eik = round(eikonal_analysis(dict_3d, edge_length, f_est), 3)

            thr_val = 83.333  # in ms
            assert isclose(min_eik / thr_val, 1., atol=3e-2), \
                f"✗ Minimum Eikonal 3D Element-{ele_geometry}-Consistent " + \
                f"→ Expected value {thr_val}, got {min_eik:.3f}"
            print(f"✓ Minimum Eikonal 3D Element-{ele_geometry}-Consistent Verified: "
                  f"Expected {thr_val}, got = {min_eik:.3f}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Eikonal 3D Element-{ele_geometry}"
                        f"-Consistent raised an exception: {str(e)}")


def test_eik_underintegrated_ele_2d():
    """Testing eikonal 2D with underintegrated elements in Fig. 8 of Salas et al (2022).

    eik_min = 83.333 ms (Theoretical value)
    f_est  T-ele  f_est  Q-ele
     0.07 82.630*  0.03 84.245 *
     0.08 84.272   0.04 85.593
     0.09 85.654   0.05 86.887
    """

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1

    # Element geometry
    ele_geometry_lst = ["T", "Q"]

    # Eikonal degree
    degree_eikonal_lst = [2, 4]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.07, 0.03]

    for case in range(0, 2):

        # Get simulation parameters
        ele_geometry = ele_geometry_lst[case]
        p_eik = degree_eikonal_lst[case]
        f_est = f_est_lst[case]
        print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
        print("Element type: {}-Underintegrated".format(ele_geometry), flush=True)
        print("Eikonal Degree: {}".format(p_eik), flush=True)
        print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

        try:
            # ============ MESH AND EIKONAL ============

            # Create dictionary with parameters for the model
            dict_2d = wave_dict(ele_geometry, 2, p_eik, 'underintegrated')

            # Creating mesh and performing eikonal analysis
            min_eik = round(eikonal_analysis(
                dict_2d, edge_length, f_est, ele_type='underintegrated'), 3)

            thr_val = 83.333  # in ms
            assert isclose(min_eik / thr_val, 1., atol=1.5e-2), \
                f"✗ Minimum Eikonal 2D Element-{ele_geometry}-Underintegrated " + \
                f"→ Expected value {thr_val}, got {min_eik:.3f}"
            print(f"✓ Minimum Eikonal 2D Element-{ele_geometry}-Underintegrated "
                  f"Verified: Expected {thr_val}, got {min_eik:.3f}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Eikonal 2D Element-{ele_geometry}"
                        f"-Underintegrated raised an exception: {str(e)}")


@pytest.mark.slow
def test_eik_underintegrated_ele_3d():
    """Testing eikonal 3D with underintegrated elements in Fig. 8 of Salas et al (2022).

    eik_min = 83.333 ms (Theoretical value)
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

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.15

    # Element geometry
    ele_geometry_lst = ["T", "Q"]

    # Eikonal degree
    degree_eikonal_lst = [2, 3]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.07, 0.05]

    for case in range(0, 2):

        # Get simulation parameters
        ele_geometry = ele_geometry_lst[case]
        p_eik = degree_eikonal_lst[case]
        f_est = f_est_lst[case]
        print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
        print("Element type: {}-Underintegrated".format(ele_geometry), flush=True)
        print("Eikonal Degree: {}".format(p_eik), flush=True)
        print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

        try:
            # ============ MESH AND EIKONAL ============

            # Create dictionary with parameters for the model
            dict_3d = wave_dict(ele_geometry, 3, p_eik, 'underintegrated')

            # Creating mesh and performing eikonal analysis
            min_eik = round(eikonal_analysis(
                dict_3d, edge_length, f_est, ele_type='underintegrated'), 3)

            thr_val = 83.333  # in ms
            assert isclose(min_eik / thr_val, 1., atol=2.5e-2), \
                f"✗ Minimum Eikonal 3D Element-{ele_geometry}-Underintegrated " + \
                f"→ Expected value {thr_val}, got {min_eik:.3f}"
            print(f"✓ Minimum Eikonal 3D Element-{ele_geometry}-Underintegrated "
                  f"Verified: Expected {thr_val}, got {min_eik:.3f}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Eikonal 3D Element-{ele_geometry}"
                        f"-Underintegrated raised an exception: {str(e)}")
