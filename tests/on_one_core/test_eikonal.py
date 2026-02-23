import pytest
import warnings
import firedrake as fire
import spyro.habc.eik as eik
from numpy import isclose
from os import getcwd
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.meshing.meshing_habc import HABC_Mesh
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict_2d(element_type):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    element_type : `str`
        Type of finite element. 'T' for triangles or 'Q' for quadrilaterals

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
        "degree": 4,  # p order p<=4 for 2D
        "dimension": 2,  # dimension
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
    Lz, Lx, Ly = [1., 1., 0.]
    dictionary["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25)],  # (0.5 * Lz, 0.25 * Lx)
        "frequency": 5.,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-Lz, 0.), (-Lz, Lx), (0., 0.), (0., Lx)]
    }

    # Simulate for 2. seconds.
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
    }

    # Define parameters for visualization
    dictionary["visualization"] = {}

    return dictionary


def wave_dict_3d(element_type, degree_eikonal):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    element_type : `str`
        Type of finite element. 'T' for triangles or 'Q' for quadrilaterals
    degree_eikonal : `int`
        Finite element order for the Eikonal equation. Should be 1 or 2.

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
        "degree": 3,  # p order p<=3 for 3D
        "dimension": 3,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "automatic",
    }

    # Define the domain size without the PML or AL. Here we'll assume a
    # 1 x 1 x 1 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x, +-y sides) of the domain.
    Lz, Lx, Ly = [1., 1., 1.]  # in km
    dictionary["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25, 0.5)],  # (0.5*Lz, 0.25*Lx, 0.25*Ly)
        "frequency": 5.,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                               (0., 0., 0), (0., Lx, 0.),
                               (-Lz, 0., Ly), (-Lz, Lx, Ly),
                               (0., 0., Ly), (0., Lx, Ly)]
    }

    # Simulate for 1.5 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 1.5,    # Final time for event
        "dt": 0.001,  # timestep size in seconds
        "amplitude": 1.,  # The Ricker has an amplitude of 1.
        "output_frequency": 100,  # How frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # How frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "degree_eikonal": degree_eikonal,  # FEM order for the Eikonal analysis
    }

    # Define parameters for visualization
    dictionary["visualization"] = {}

    return dictionary


class HABC_Wave(AcousticWave, HABC_Mesh):
    '''
    Class HABC that determines absorbing layer size and parameters to be used

    Attributes
    ----------
    path_save : `string`
        Path to save data

    Methods
    -------
    None added to the ones inherited from AcousticWave and HABC_Mesh
    '''

    def __init__(self, dictionary=None, comm=None):
        '''
        Initialize the HABC class

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the HABC class
        comm : `object`, optional
            An object representing the communication interface
            for parallel processing. Default is None

        Returns
        -------
        None
        '''

        # Initializing the Wave class
        AcousticWave.__init__(self, dictionary=dictionary, comm=comm)

        # Path to save data
        self.path_save = getcwd() + "/output/eikonal_test"

        # Original domain dimensions
        dom_dim = (self.mesh_parameters.length_x,
                   self.mesh_parameters.length_z)

        if self.dimension == 2:  # 2D
            self.path_save += "2d/"

        if self.dimension == 3:  # 3D
            self.path_save += "3d/"
        dom_dim += (self.mesh_parameters.length_y,)

        # Initializing the Mesh class
        HABC_Mesh.__init__(
            self, dom_dim, dimension=self.dimension,
            quadrilateral=self.mesh_parameters.quadrilateral,
            p_eik=self.abc_deg_eikonal, comm=self.comm)


def critical_boundary_points(Wave_obj):
    '''
    Determine the critical points on domain boundaries of the original
    model to size an absorbing layer using the Eikonal criterion for HABCs.
    See Salas et al (2022) for details.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class

    Returns
    -------
    eik_bnd : `list`
        Properties on boundaries according to minimum values of Eikonal
        Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
        - pt_cr : Critical point coordinates
        - c_bnd : Propagation speed at critical point
        - eikmin : Eikonal value in seconds
        - z_par : Inverse of minimum Eikonal (Equivalent to c_bound / lref)
        - lref : Distance to the closest source
    '''

    # Initializing Eikonal object
    Eikonal = eik.HABC_Eikonal(Wave_obj)

    # Solving Eikonal
    Eikonal.solve_eik()

    # Identifying critical points
    eik_bnd = Eikonal.ident_crit_eik()

    return eik_bnd


def eikonal_analysis(dictionary, edge_length, f_est):
    '''
    Run the the Eikonal analysis

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km
    f_est : `float`, optional
        Factor for the stabilizing term in Eikonal Eq.

    Returns
    -------
    min_eik : `float`
        Minimum Eikonal value in miliseconds
    '''

    # ============ MESH FEATURES ============

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    Wave_obj = HABC_Wave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.preamble_mesh_operations(f_est=f_est)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=Wave_obj.path_save + "MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    eik_bnd = critical_boundary_points(Wave_obj)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef, user_name=Wave_obj.path_save + "EIK_")

    # Extracting  minimum Eikonal
    min_eik = 1e3 * eik_bnd[0][2]

    return min_eik


def test_loop_eikonal_2d():
    '''
    Loop for testing eikonal solver in 2D with the model
    in Fig. 8 of Salas et al. (2022)

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
    '''

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1

    # Factor for the stabilizing term in Eikonal equation

    for case in range(0, 2):

        # Element type
        ele_type_lst = ["T", "Q"]

        # Factor for the stabilizing term in Eikonal equation
        f_est_lst = [0.06, 0.05]

        # Get simulation parameters
        ele_type = ele_type_lst[case]
        f_est = f_est_lst[case]
        print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
        print("Element type: {}".format(ele_type), flush=True)
        print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

        try:
            # ============ MESH AND EIKONAL ============

            # Create dictionary with parameters for the model
            dict_2d = wave_dict_2d(ele_type)

            # Creating mesh and performing eikonal analysis
            min_eik = round(eikonal_analysis(dict_2d, edge_length, f_est), 3)

            thr_val = 83.333  # in ms
            assert isclose(min_eik / thr_val, 1., atol=5e-3), \
                f"❌ Minimum Eikonal 2D Element-{ele_type} " + \
                f"→ Expected value {thr_val}, got {min_eik:.3f}"
            print(f"✅ Minimum Eikonal 2D Verified: expected "
                  f"{thr_val}, got = {min_eik:.3f}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Eikonal 2D raised an exception: {str(e)}")


def test_loop_eikonal_3d():
    '''
    Loop for testing eikonal solver in 3D with the model
    in Fig. 8 of Salas et al. (2022)

    eik_min = 83.333 ms (Theoretical value)
    f_est  T-ele   Q-ele
     0.02  --/--  69.442
     0.03 76.777  70.974
     0.04 79.409  73.179
     0.05 82.273* 75.766
     0.06 85.347  78.548
     0.07 88.562  81.431*
     0.08 91.876  84.377
    '''

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.15

    # Factor for the stabilizing term in Eikonal equation

    for case in range(0, 1):

        # Element type
        ele_type_lst = ["T", "Q"]

        # Eikonal degree
        degree_eikonal_lst = [2, 2]

        # Factor for the stabilizing term in Eikonal equation
        f_est_lst = [0.05, 0.07]

        # Get simulation parameters
        ele_type = ele_type_lst[case]
        p_eik = degree_eikonal_lst[case]
        f_est = f_est_lst[case]
        print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
        print("Element type: {}".format(ele_type), flush=True)
        print("Eikonal Degree: {}".format(p_eik), flush=True)
        print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

        # ============ MESH AND EIKONAL ============
        try:

            # Create dictionary with parameters for the model
            dict_3d = wave_dict_3d(ele_type, p_eik)

            # Creating mesh and performing eikonal analysis
            min_eik = round(eikonal_analysis(dict_3d, edge_length, f_est), 3)

            thr_val = 83.333  # in ms

            assert isclose(min_eik / thr_val, 1., atol=3e-2), \
                f"❌ Minimum Eikonal 3D Element-{ele_type} " + \
                f"→ Expected value {thr_val}, got {min_eik:.3f}"
            print(f"✅ Minimum Eikonal 3D Verified: expected "
                  f"{thr_val}, got = {min_eik:.3f}", flush=True)

        except fire.ConvergenceError as e:
            pytest.fail(f"Checking Eikonal 3D raised an exception: {str(e)}")
