import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="loopy")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="matplotlib.backends._backend_tk")
import firedrake as fire
import numpy as np
from spyro.utils.cost import comp_cost
import spyro.pml.pml_nsnc as pml


def wave_dict_2d(dt_usu):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    dt_usu: `float`
        Time step of the simulation

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
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
    # 1.00 x 1.00 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 1.,  # depth in km - always positive
        "Lx": 1.,  # width in km - always positive
        "Ly": 0.,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25)],
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)], # ToDo
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-1., 0.), (-1., 1.), (0., 1.), (0., 0.)]
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "PML",  # Activate HABC
        "exponent": 2,
        "R": 1e-6,
        "habc_reference_freq": "source",  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": True,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "output/forward/fw_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "acoustic_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": "output/pml_test2d/preamble/acoustic_pot_energy",
    }

    return dictionary


def wave_dict_3d(dt_usu):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    dt_usu: `float`
        Time step of the simulation

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model.
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
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
    # 1.00 x 1.00 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
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
        "source_locations": [(-0.5, 0.25, 0.5)],
        "frequency": 5.0,  # in Hz
        "delay": 1. / 3.,
        "delay_type": "time",  # "multiples_of_minimum" or "time"
        "receiver_locations": [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                               (0., 0., 0), (0., Lx, 0.),
                               (-Lz, 0., Ly), (-Lz, Lx, Ly),
                               (0., 0., Ly), (0., Lx, Ly)]
    }

    # Simulate for 1. seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 1.,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1.,  # the Ricker has an amplitude of 1.
        "output_frequency": 50,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 50,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "PML",  # Activate HABC
        "exponent": 2,
        "R": 1e-6,
        "habc_reference_freq": "source",  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": True,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "output/forward/fw_output_3d.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "acoustic_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": "output/pml_test3d/preamble/acoustic_pot_energy_3d",
    }

    return dictionary


def preamble_pml(dictionary, edge_length, f_est, dimension):
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
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    '''

    # ============ MESH FEATURES ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Create the acoustic wave object with HABCs
    Wave_obj = pml.PML_Wave(dictionary=dictionary,
                            bc_boundary_pml="Higdon",
                            output_folder=f"output/pml_test{dimension}d")

    # Mesh
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.preamble_mesh_operations(f_est=f_est)

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost("tini")

    # Finding critical points
    Wave_obj.critical_boundary_points()

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/EIK_")

    return Wave_obj


def run_reference(Wave_obj, max_divisor_tf=1):
    '''
    Run the infinite model to get the reference signal for the PML scheme

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    max_divisor_tf : `int`, optional
        Index to select the maximum divisor of the final time, converted
        to an integer according to the order of magnitude of the timestep
        size. The timestep size is set to the divisor, given by the index
        in descending order, less than or equal to the user's timestep
        size. If the value is 1, the timestep size is set as the maximum
        divisor. Default is 1

    Returns
    -------
    None
    '''

    # Reference to resource usage
    tRef = comp_cost("tini")

    # Computing reference get_reference_signal
    Wave_obj.infinite_model(check_dt=True, max_divisor_tf=max_divisor_tf)

    # Set model parameters for the HABC scheme
    Wave_obj.abc_get_ref_model = False

    # Estimating computational resource usage
    comp_cost("tfin", tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/INF_")


def pml_fig8(Wave_obj):
    '''
    Apply the PML scheme to the model in Fig. 8 of Salas et al. (2022)

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    Wave_obj.identify_abc_layer_case(
        output_folder=f"output/pml_test{Wave_obj.dimension}d")

    # Acquiring reference signal
    Wave_obj.get_reference_signal()

    # Determining layer size
    Wave_obj.layer_size_criterion(n_root=1)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_with_layer()

    # Updating velocity model
    Wave_obj.velocity_abc()

    # Building the PML layer (damping and BCs)
    Wave_obj.pml_layer()

    # Solving the forward problem
    Wave_obj.forward_solve()

    # Computing the error measures
    Wave_obj.error_measures_habc()

    # Plotting the solution at receivers and the error measures
    Wave_obj.comparison_plots()


def run_pml(Wave_obj):
    '''
    Run the PML scheme for different reference frequencies.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class

    Returns
    -------
    None
    '''

    # Reference frequency
    habc_ref_freq = "source"

    # Data to print on screen
    fref_str = "PML Reference Frequency: {}"

    # Reference frequency for sizing the hybrid absorbing layer
    Wave_obj.abc_reference_freq = habc_ref_freq
    print(fref_str.format(habc_ref_freq.capitalize()), flush=True)

    try:

        # Reference to resource usage
        tRef = comp_cost("tini")

        # Run the HABC scheme
        pml_fig8(Wave_obj)

        # Estimating computational resource usage
        comp_cost("tfin", tRef=tRef,
                  user_name=Wave_obj.path_case_abc)

        # Expected errors <= 5%
        comp_it = 100. * Wave_obj.max_errIt <= 5.
        comp_pk = 100. * Wave_obj.max_errPK <= 5.

        met_str = f"Checking PML {Wave_obj.dimension}D"
        cmp_str = "Expected errors <= 5%, got: Integral: "
        cmp_str += f"{Wave_obj.max_errIt:.2%} - Peak: {Wave_obj.max_errPK:.2%}"
        assert comp_it and comp_pk, "✗ " + met_str + " → " + cmp_str
        print("✓ " + met_str + " Verified: " + cmp_str, flush=True)

    except fire.ConvergenceError as e:
        pytest.fail(f"Checking PML {Wave_obj.dimension}D "
                    f"raised an exception: {str(e)}")


def test_pml_2d():
    '''
    Loop for applying the PML2D to the model in Fig. 8 of Salas et al. (2022)
    '''

    # ============ SIMULATION PARAMETERS ============

    # Mesh size (in km)
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.1

    # Timestep size (in seconds). Initial guess: edge_length / 50
    dt_usu = 0.001

    # Factor for the stabilizing term in Eikonal equation
    f_est = 0.06

    # Maximum divisor of the final time
    max_div_tf = 5

    # Get simulation parameters
    print(f"\nMesh Size: {1e3 * edge_length:.3f} m", flush=True)
    print(f"Timestep Size: {1e3 * dt_usu:.3f} ms", flush=True)
    print(f"Eikonal Stabilizing Factor: {f_est:.2f}", flush=True)
    print(f"Maximum Divisor of Final Time: {max_div_tf}", flush=True)

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict_2d(dt_usu)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_pml(dictionary, edge_length, f_est, 2)

    # ============ REFERENCE MODEL ============

    # Create the infinite model and get the reference signal
    run_reference(Wave_obj, max_divisor_tf=max_div_tf)

    # ============ PML SCHEME ============

    # Run the PML scheme
    run_pml(Wave_obj)


def test_pml_3d():
    '''
    Loop for applying the PML3D to the model in Fig. 8 of Salas et al. (2022)
    '''

    # ============ SIMULATION PARAMETERS ============

    # Mesh size in km
    # cpw: cells per wavelength
    # lba = minimum_velocity / source_frequency
    # edge_length = lba / cpw
    edge_length = 0.150

    # Timestep size (in seconds). Initial guess: edge_length / 50
    dt_usu = 0.002  # 0.0016  # 0.00125  # 0.001  # 0.0008  # 0.0005

    # Factor for the stabilizing term in Eikonal equation
    f_est = 0.05

    # Maximum divisor of the final time
    max_div_tf = 3  # 4  # 5  # 6  # 7  #8

    # Get simulation parameters
    print(f"\nMesh Size: {1e3 * edge_length:.3f} m", flush=True)
    print(f"Timestep Size: {1e3 * dt_usu:.3f} ms", flush=True)
    print(f"Eikonal Stabilizing Factor: {f_est:.2f}", flush=True)
    print(f"Maximum Divisor of Final Time: {max_div_tf}", flush=True)

    # ============ MESH AND EIKONAL ============

    # Create dictionary with parameters for the model
    dictionary = wave_dict_3d(dt_usu)

    # Creating mesh and performing eikonal analysis
    Wave_obj = preamble_pml(dictionary, edge_length, f_est, 3)

    # ============ REFERENCE MODEL ============

    # Create the infinite model and get the reference signal
    run_reference(Wave_obj, max_divisor_tf=max_div_tf)

    # ============ PML SCHEME ============

    # Run the PML scheme
    run_pml(Wave_obj)


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":
    test_pml_2d()
    # test_pml_3d()


'''
=================================================================
DATA FOR 2D MODEL - Ele = T
----------------------------

*EIKONAL
eik_min = 83.333 ms
f_est   100m
 0.01 66.836 
 0.02 73.308 
 0.03 77.178 
 0.04 79.680 
 0.05 81.498 
 0.06 82.942*
 0.07 84.160 
 0.08 85.233 

=================================================================
DATA FOR 3D MODEL - Ele = T
-----------------------------

*EIKONAL
eik_min = 83.333 ms
f_est   150m 
 0.03 76.777 
 0.04 79.409 
 0.05 82.273*
 0.06 85.347 
'''
