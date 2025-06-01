import firedrake as fire
import spyro.habc.habc as habc
import spyro.habc.eik as eik
from spyro.habc.cost import comp_cost
import ipdb


def wave_dict(layer_shape, degree_layer, habc_reference_freq, get_ref_model):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    layer_shape : `str`
        Shape of the absorbing layer, either 'rectangular' or 'hypershape'
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used
    habc_reference_freq : str
        Reference frequency for the layer size. Options: 'source' or 'boundary
    get_ref_model : `bool`
        If True, the infinite model is created. If False, the absorbing layer
        is created based on the model parameters.

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
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.5, 0.25)],
        "frequency": 5.0,  # in Hz
        "delay": 1.5,
        "receiver_locations": [(-1., 0.), (-1., 1.), (0., 1.), (0., 0.)]
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)],
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Integer >= 2. Only for hypershape
        "habc_reference_freq": habc_reference_freq,  # Options: source or boundary
        "degree_eikonal": 2,  # Finite element order for the Eikonal analysis
        "get_ref_model": get_ref_model,  # If True, the infinite model is created
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
        "acoustic_energy_filename": "output/preamble/acoustic_potential_energy",
    }

    return dictionary


def preamble_habc(dictionary, edge_length):
    '''
    Run the infinite model and the Rikonal analysis

    Parameters
    ----------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    edge_length : `float`
        Mesh size in km

    Returns
    -------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    '''

    # ============ MESH FEATURES ============
    # Reference to resource usage
    tRef = comp_cost('tini')

    # Create the acoustic wave object with HABCs
    Wave_obj = habc.HABC_Wave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond)

    # Preamble mesh operations
    Wave_obj.preamble_mesh_operations()

    # Estimating computational resource usage
    comp_cost('tfin', tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/MSH_")

    # ============ EIKONAL ANALYSIS ============
    # Reference to resource usage
    tRef = comp_cost('tini')

    # Initializing Eikonal object
    Eik_obj = eik.Eikonal(Wave_obj)

    # Finding critical points
    Wave_obj.critical_boundary_points(Eik_obj)

    # Estimating computational resource usage
    comp_cost('tfin', tRef=tRef,
              user_name=Wave_obj.path_save + "preamble/EIK_")

    return Wave_obj


def test_habc_fig8(Wave_obj, xCR_usu=None):
    '''
    Apply the HABC to the model in Fig. 8 of Salas et al. (2022).

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    xCR_usu : `float`, optional
        User-defined heuristic factor for the minimum damping ratio.
        Default is None, which defines an estimated value

    Returns
    -------
    None
    '''

    # Identifier for the current case study
    Wave_obj.identify_habc_case()

    # Acquiring reference signal
    Wave_obj.get_reference_signal()

    # Determining layer size
    Wave_obj.size_habc_criterion(n_root=1, layer_based_on_mesh=True)

    # Creating mesh with absorbing layer
    Wave_obj.create_mesh_habc()

    # Updating velocity model
    Wave_obj.velocity_habc()

    # Setting the damping profile within absorbing layer
    Wave_obj.damping_layer(xCR_usu=xCR_usu)

    # Applying NRBCs on outer boundary layer
    Wave_obj.cos_ang_HigdonBC()

    # Solving the forward problem
    Wave_obj.forward_solve()

    # Computing the error measures
    Wave_obj.error_measures_habc()

    # Plotting the solution at receivers
    Wave_obj.comparison_plots()


# Applying HABCs to the model in Fig. 8 of Salas et al. (2022)
if __name__ == "__main__":

    # Mesh size
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    edge_length_lst = [0.05]  # [0.05, 0.02, 0.01]

    degree_layer_lst = [None]  # [None, 2, 3, 4, 5]

    habc_reference_freq_lst = ["source"]  # ["source", "boundary"]

    get_ref_model = False

    loop_modeling = not get_ref_model

    for edge_length in edge_length_lst:

        # ============ MESH AND EIKONAL ============
        # Create dictionary with parameters for the model
        dictionary = wave_dict("rectangular", None, "source", get_ref_model)

        # Creating mesh and performing eikonal analysis
        Wave_obj = preamble_habc(dictionary, edge_length)

        # ============ REFERENCE MODEL ============
        if get_ref_model:
            # Reference to resource usage
            tRef = comp_cost('tini')

            # Computing reference signal
            Wave_obj.infinite_model()

            # Set model parameters for the HABC scheme
            Wave_obj.abc_get_ref_model = False

            # Estimating computational resource usage
            comp_cost('tfin', tRef=tRef,
                      user_name=Wave_obj.path_save + "preamble/INF_")

        # ============ HABC SCHEME ============
        if loop_modeling:

            # Loop for different layer shapes and degrees
            for habc_reference_freq in habc_reference_freq_lst:

                # Reference frequency for sizing the hybrid absorbing layer
                Wave_obj.abc_reference_freq = habc_reference_freq

                for degree_layer in degree_layer_lst:

                    # Update the layer shape and its degree
                    Wave_obj.abc_boundary_layer_shape = "hypershape" \
                        if degree_layer is not None else "rectangular"
                    Wave_obj.abc_deg_layer = degree_layer

                    # Reference to resource usage
                    tRef = comp_cost('tini')

                    # Run the HABC scheme
                    test_habc_fig8(Wave_obj, xCR_usu=1.495)

                    # Estimating computational resource usage
                    user_name = Wave_obj.path_save + Wave_obj.case_habc + "/"
                    comp_cost('tfin', tRef=tRef, user_name=user_name)


# 0.111 1.78 0.60 5.60e-08
# 0.158 1.75 0.65 5.47e-08
# 0.319 1.64 0.79 5.03e-08
# 0.637 1.47 1.07 4.33e-08
# 0.955 1.33 1.33 3.77e-08
# 1.103 1.28 1.45 3.55e-08
# 1.495 1.17 1.73 3.09e-08
# 1.887 1.10 2.01 2.75e-08
