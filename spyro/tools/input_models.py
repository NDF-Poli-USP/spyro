import numpy as np
import spyro
import SeismicMesh

def create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = 'homogeneous', receiver_type = 'near'):
    ''' Creates models  with the correct parameters for for grid point calculation experiments.
    
    Parameters
    ----------
    frequency: `float`
        Source frequency to use in calculation
    degree: `int`
        Polynomial degree of finite element space
    method: `string`
        The finite element method choosen
    minimum_mesh_velocity: `float`
        Minimum velocity presented in the medium
    experiment_type: `string`
        Only options are `homogenous` or `heterogenous`
    receiver_type: `string`
        Options: `near`, `far` or `near_and_far`. Specifies receiver grid locations for experiment

    Returns
    -------
    model: Python `dictionary`
        Contains model options and parameters for use in Spyro
        

    '''
    model = {}
    # domain calculations
    if experiment_type == 'homogenous':
        lbda = minimum_mesh_velocity/frequency
        pml_fraction = lbda
        Lz = 30*lbda#100*lbda
        Real_Lz = Lz*(1. + 2*pml_fraction)
        Lx = 20*lbda#90*lbda
        Real_Lx = Lx*(1. + 1*pml_fraction)

        # source location
        source_coordinates = [(Real_Lz/2, Real_Lx/2)] #Source at the center. If this is changes receiver's bin has to also be changed.
        source_z = Real_Lz/2.
        source_x = Real_Lx/2.
        padz = Lz*pml_fraction
        padx = Lx*pml_fraction

    if experiment_type == 'heterogenous':
        #using the BP2004 velocity model
        
        Lz = 12000.0/1000.
        Lx = 67000.0/1000.
        pad = 1000./1000.
        Real_Lz = Lz+ pad
        Real_Lx = Lx+ 2*pad
        source_z = -1.0
        source_x = Real_Lx/2.
        source_coordinates = [(source_z,source_x)]
        SeismicMesh.write_velocity_model('vel_z6.25m_x12.5m_exact.segy', ofname = 'velocity_models/bp2004')
        padz = pad
        padx = pad
    
    if receiver_type == 'near' and experiment_type == 'homogenous':

        # time calculations
        tmin = 1./frequency
        final_time = 10*tmin #should be 35

        # receiver calculations

        receiver_bin_center1 = 5*lbda#20*lbda
        receiver_bin_width = 5*lbda#15*lbda
        receiver_quantity = 16#2500 # 50 squared

        bin1_startZ = source_z + receiver_bin_center1 - receiver_bin_width/2.
        bin1_endZ   = source_z + receiver_bin_center1 + receiver_bin_width/2.
        bin1_startX = source_x - receiver_bin_width/2.
        bin1_endX   = source_x + receiver_bin_width/2.

    if receiver_type == 'near' and experiment_type == 'heterogenous':

        # time calculations
        tmin = 1./frequency
        final_time = 2*10*tmin #should be 35

        # receiver calculations

        receiver_bin_center1 = 2.5*750.0/1000
        receiver_bin_width = 500.0/1000
        receiver_quantity_in_bin = 100#2500 # 50 squared

        bin1_startZ = source_z - receiver_bin_width/2.
        bin1_endZ   = source_z + receiver_bin_width/2.
        bin1_startX = source_x + receiver_bin_center1 - receiver_bin_width/2.
        bin1_endX   = source_x + receiver_bin_center1 + receiver_bin_width/2.

        receiver_coordinates = spyro.create_2d_grid(bin1_startZ, bin1_endZ, bin1_startX, bin1_endX, int(np.sqrt(receiver_quantity_in_bin)))

        receiver_bin_center2 = 6500.0/1000
        receiver_bin_width = 500.0/1000

        bin2_startZ = source_z - receiver_bin_width/2.
        bin2_endZ   = source_z + receiver_bin_width/2.
        bin2_startX = source_x + receiver_bin_center2 - receiver_bin_width/2.
        bin2_endX   = source_x + receiver_bin_center2 + receiver_bin_width/2.

        receiver_coordinates= receiver_coordinates + spyro.create_2d_grid(bin2_startZ, bin2_endZ, bin2_startX, bin2_endX, int(np.sqrt(receiver_quantity_in_bin))) 

        receiver_quantity = 2*receiver_quantity_in_bin

    



    elif receiver_type == 'far':
        raise ValueError('Far receivers minimum grid point calculation experiment not implemented because of computational limits.')
    
    # Choose method and parameters
    model["opts"] = {
        "method": method,
        "variant": None,
        "element": "tria",  # tria or tetra
        "degree": degree,  # p order
        "dimension": 2,  # dimension
    }

    model["PML"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
        "exponent": 1,
        "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
        "R": 0.001,  # theoretical reflection coefficient
        "lz": padz,  # thickness of the pml in the z-direction (km) - always positive
        "lx": padx,  # thickness of the pml in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
    }

    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "demos/mm_exact.msh",
        "initmodel": "velocity_models/bp2004.hdf5",
        "truemodel": "velocity_models/bp2004.hdf5",
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": source_coordinates,
        "source_mesh_point": True,
        "source_point_dof": False,
        "frequency": frequency,
        "delay": 1.0,
        "num_receivers": receiver_quantity,
        "receiver_locations": receiver_coordinates,
    }

    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": final_time,  # Final time for event
        "dt": 0.001,  # timestep size
        "nspool": 200,  # how frequently to output solution to pvds
        "fspool": 100,  # how frequently to save solution to RAM
    }  
    model["parallelism"] = {
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off.
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
    }
    model['testing_parameters'] = {
        'experiment_type': experiment_type,
        'minimum_mesh_velocity': minimum_mesh_velocity,
        'pml_fraction': padz/Lz,
        'receiver_type': receiver_type
    }

    # print(source_coordinates)
    # print(receiver_coordinates)
    return model

