import firedrake as fire
import warnings
import spyro.habc.eik as eik
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.meshing.meshing_habc import HABC_Mesh
from os import getcwd
from spyro.utils.cost import comp_cost
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict():
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    None

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",  # "Q",
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
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-Lz, 0.), (-Lz, Lx), (0., 0.), (0., Lx)]
    }

    # Simulate for 2. seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": 0.001,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        # "layer_shape": layer_shape,  # Options: rectangular or hypershape
        # "degree_layer": degree_layer,  # Float >= 2 (hyp) or None (rec)
        # "degree_type": degree_type,  # Options: real or integer
        # "habc_reference_freq": habc_ref_freq,  # Options: source or boundary
        # "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        # "get_ref_model": False,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        # "forward_output": False,
        # "forward_output_filename": "output/forward/fw_output.pvd",
        # "fwi_velocity_model_output": False,
        # "velocity_model_filename": None,
        # "gradient_output": False,
        # "gradient_filename": None,
        # "acoustic_energy": False,  # Activate energy calculation
        # "acoustic_energy_filename": "output/preamble/acoustic_pot_energy",
    }

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
        self.path_save = getcwd() + "/output/eikonal_test2d/"

        # Original domain dimensions
        dom_dim = (self.mesh_parameters.length_x,
                   self.mesh_parameters.length_z)
        if self.dimension == 3:  # 3D
            self.path_save = path_save[:-3] + "3d/"
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
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
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


def test_loop_eikonal_2d():
    '''
    Loop for testing eikonal solver in 2D
    '''

    # ============ SIMULATION PARAMETERS ============

    case = 0  # Integer from 0 to 4

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length_lst = 0.1000  # [0.1000, 0.0625, 0.0500, 0.0250, 0.0200]

    # Factor for the stabilizing term in Eikonal equation
    f_est_lst = [0.06, 0.02, 0.02, 0.02, 0.04]

    # Get simulation parameters
    edge_length = edge_length_lst[case]
    f_est = f_est_lst[case]
    print("\nMesh Size: {:.4f} m".format(1e3 * edge_length), flush=True)
    print("Eikonal Stabilizing Factor: {:.2f}".format(f_est), flush=True)

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict()

    # Creating mesh and performing eikonal analysis
    eikonal_analysis(dictionary, edge_length, f_est)


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":
    test_loop_eikonal_2d()

# eik_min = 83.333 ms
# f_est   100m   62.5m     50m     25m     20m
#  0.01 66.836   --/--   --/--   --/--   --/--
#  0.02 73.308  83.907* 83.944* 83.812* 82.193
#  0.03 77.178  85.322  85.068  84.398   --/--
#  0.04 79.680  86.352  85.933  84.901  83.434*
#  0.05 81.498  87.263  86.718  85.375  83.863
#  0.06 82.942* 88.130  87.470  85.837  84.250
#  0.07 84.160  88.977  88.207  86.292  84.613
#  0.08 85.233  89.815  88.934  86.745  84.961
