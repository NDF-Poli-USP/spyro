import firedrake as fire
import spyro.habc.habc as habc
from spyro.utils.cost import comp_cost
import ipdb


def wave_dict(edge_dom, layer_shape, degree_layer, habc_reference_freq):
    '''
    Create a dictionary with parameters for the model.

    Parameters
    ----------
    edge_dom: `float`
        Edge of the cubic domain
    layer_shape : `str`
        Shape of the absorbing layer, either 'rectangular' or 'hypershape'
    degree_layer : `int` or `None`
        Degree of the hypershape layer, if applicable. If None, it is not used
    habc_reference_freq : str
        Reference frequency for the layer size. Options: 'source' or 'boundary

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
    dictionary["mesh"] = {
        "Lz": edge_dom,  # depth in km - always positive
        "Lx": edge_dom,  # width in km - always positive
        "Ly": edge_dom,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-edge_dom / 2., edge_dom / 4., edge_dom / 2.)],
        "frequency": 5.0,  # in Hz
        "delay": 1. / 3.,
        "delay_type": "time",  # "multiples_of_minimun" or "time"
        "receiver_locations": [(-edge_dom, 0., 0.), (-edge_dom, edge_dom, 0.),
                               (0., edge_dom, 0.), (0., 0., 0),
                               (-edge_dom, 0., edge_dom),
                               (-edge_dom, edge_dom, edge_dom),
                               (0., edge_dom, edge_dom), (0., 0., edge_dom)]
    }

    # Simulate for 1.0 seconds.
    dt_usu = 0.003
    fr_files = max(int(100 * 0.0005 / dt_usu), 1)
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.,    # Final time for event
        "dt": dt_usu,  # timestep size in seconds
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": fr_files,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": fr_files,  # how frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": True,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
        "layer_shape": layer_shape,  # Options: rectangular or hypershape
        "degree_layer": degree_layer,  # Integer >= 2. Only for hypershape
        "habc_reference_freq": habc_reference_freq,  # Options: source or boundary
        "degree_eikonal": 1,  # Finite element order for the Eikonal analysis
        "get_ref_model": False,  # If True, the infinite model is created
    }

    # Define parameters for visualization
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "output_3d/forward/fw_output_3d.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "acoustic_energy": True,  # Activate energy calculation
        "acoustic_energy_filename": "output_3d/preamble/acoustic_potential_energy",
    }

    return dictionary


def preamble_modal(dictionary, edge_length):
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

    # Create the acoustic wave object with HABCs
    Wave_obj = habc.HABC_Wave(dictionary=dictionary)

    # Mesh
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})

    # Initial velocity model
    Wave_obj.set_initial_velocity_model(constant=1.5)

    # Preamble mesh operations
    print("\nCreating Mesh and Initial Velocity Model")

    # Save a copy of the original mesh
    mesh_orig = fire.VTKFile(Wave_obj.path_save + "modal/mesh_orig.pvd")
    mesh_orig.write(Wave_obj.mesh)

    # Velocity profile model
    Wave_obj.c = fire.Function(Wave_obj.function_space, name='c_orig [km/s])')
    Wave_obj.c.interpolate(Wave_obj.initial_velocity_model)

    # Save initial velocity model
    vel_c = fire.VTKFile(Wave_obj.path_save + "modal/c_vel.pvd")
    vel_c.write(Wave_obj.c)

    return Wave_obj


def modal_fig8(Wave_obj, modal_solver):
    '''
    Compute the fundamental frequency of a homogenenous cubic domain.

    Parameters
    ----------
    Wave_obj : `habc.HABC_Wave`
        An instance of the HABC_Wave class
    modal_solver : `str`
        Method to use for solving the eigenvalue problem.
        Options: 'ARNOLDI', 'LANCZOS', 'LOBPCG' 'KRYLOVSCH_CH',
        'KRYLOVSCH_CG', 'KRYLOVSCH_GH' or 'KRYLOVSCH_GG'

    Returns
    -------
    None
    '''

    # Computing fundamental frequency
    Wave_obj.fundamental_frequency(method=modal_solver, monitor=True)


def test_loop_modal_3d():
    '''
    Loop for testing modals solvers in 3D.
    '''

    # ============ SIMULATION PARAMETERS ============

    # Edge of the cubic domain in km
    edge_dom = 2.0

    # Mesh size in km
    # cpw: cells per wavelength
    # lba = minimum_velocity /source_frequency
    # edge_length = lba / cpw
    # edge_length_lst = [0.150, 0.125, 0.100, 0.080]
    # edge_length_lst = [0.125, 0.100, 0.080]
    edge_length_lst = [0.080]

    # Modal solvers
    # modal_solver_lst = ['ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
    #                     'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG']
    # modal_solver_lst = ['LOBPCG', 'KRYLOVSCH_CH', 'KRYLOVSCH_CG',
    #                     'KRYLOVSCH_GH', 'KRYLOVSCH_GG']
    modal_solver_lst = ['KRYLOVSCH_CH', 'KRYLOVSCH_CG',
                        'KRYLOVSCH_GH', 'KRYLOVSCH_GG']

    # Data to print on screen
    mod_str = "\nModal Solver: {}"

    # Loop for different mesh size
    for edge_length in edge_length_lst:

        # Get simulation parameters
        print("\nMesh Size: {:.3f} km\n".format(edge_length))

        # ============ MODEL MESH ============

        # Create dictionary with parameters for the model
        dictionary = wave_dict(edge_dom, "rectangular", None, "source")

        # Creating model mesh
        Wave_obj = preamble_modal(dictionary, edge_length)

        # ============ MODAL ANALYSIS ============

        # Loop for different modal solvers
        for modal_solver in modal_solver_lst:

            # Modal solver
            print(mod_str.format(modal_solver))

            try:
                # Reference to resource usage
                tRef = comp_cost("tini")

                # Computing the fundamental frequency
                modal_fig8(Wave_obj, modal_solver)

                # Estimating computational resource usage
                comp_cost("tfin", tRef=tRef,
                          user_name=Wave_obj.path_save + "modal/MOD_")
            except Exception as e:
                print(f"Error Solving: {e}")


# Testing several modal solvers for 3D models
if __name__ == "__main__":
    test_loop_modal_3d()
