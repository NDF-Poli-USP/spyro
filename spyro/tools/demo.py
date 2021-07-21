### Demo to illustrate how a grid point density calcultor runs the experiments
import spyro
# First we need to define experiment parameters:
grid_point_calculator_parameters = {
    ## Experiment parameters
    'source_frequency' : 5.0, # Here we define the frequency of the Ricker wavelet source
    'minimum_velocity_in_the_domain' :  1.429, # The minimum velocity present in the domain.
    # if an homogeneous test case is used this velocity will be defined in the whole domain.
    'velocity_profile_type': 'homogeneous', # Either or heterogeneous. If heterogeneous is 
    #chosen be careful to have the desired velocity model below.
    'velocity_model_file_name': 'vel_z6.25m_x12.5m_exact.segy',
    'FEM_method_to_evaluate' : 'KMV', # FEM to evaluate such as `KMV` or `spectral` (GLL nodes on quads and hexas)
    'dimension' : 2, # Domain dimension. Either 2 or 3.
    'receiver_setup' : 'near', #Either near or line. Near defines a receiver grid near to the source,
    # line defines a line of point receivers with pre-established near and far offsets.

    ## Line search parameters
    'reference_degree': 5, # Degree to use in the reference case (int)
    'G_reference': 15.0, # grid point density to use in the reference case (float)
    'desired_degree': 4, # degree we are calculating G for. (int)
    'G_initial': 6.0, # Initial G for line search (float)
    'accepted_error_threshold': 0.05, 
    'g_accuracy': 1e-1
    }


G = spyro.tools.minimum_grid_point_calculator(grid_point_calculator_parameters)

