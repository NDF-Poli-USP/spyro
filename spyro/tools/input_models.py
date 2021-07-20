import numpy as np
import spyro
import SeismicMesh

def create_3d_grid(start, end, num):
    """Create a 3d grid of `num**3` points between `start1`
    and `end1` and `start2` and `end2`

    Parameters
    ----------
    start: tuple of floats
        starting position coordinate
    end: tuple of floats
        ending position coordinate
    num: integer
        number of receivers between `start` and `end`

    Returns
    -------
    receiver_locations: a list of tuples

    """
    (start1, start2, start3) = start
    (end1, end2, end3)  = end
    x = np.linspace(start1, end1, num)
    y = np.linspace(start2, end2, num)
    z = np.linspace(start3, end3, num)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    return [tuple(point) for point in points]

def create_model_for_grid_point_calculation2D(degree):
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

    if minimum_mesh_velocity > 500:
        print("Warning: minimum mesh velocity seems to be in m/s, input should be in km/s", flush = True)
    # domain calculations
    pady = 0.0
    Ly = 0.0
    if experiment_type == 'homogeneous' and dimension == 2:
        lbda = minimum_mesh_velocity/frequency
        pml_fraction = lbda
        pad = lbda
        Lz = 40*lbda#100*lbda
        Real_Lz = Lz+ pad
        #print(Real_Lz)
        Lx = 30*lbda#90*lbda
        Real_Lx = Lx+ 2*pad

        # source location
        source_z = -Real_Lz/2.#1.0
        #print(source_z)
        source_x = Real_Lx/2.
        source_coordinates = [(source_z, source_x)] #Source at the center. If this is changes receiver's bin has to also be changed.
        padz = pad
        padx = pad

    if experiment_type == 'heterogeneous':
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
    
    if receiver_type == 'near' and experiment_type == 'homogeneous' and dimension == 2:

        # time calculations
        tmin = 1./frequency
        final_time = 20*tmin #should be 35

        # receiver calculations

        receiver_bin_center1 = 10*lbda#20*lbda
        receiver_bin_width = 5*lbda#15*lbda
        receiver_quantity = 36#2500 # 50 squared

        bin1_startZ = source_z + receiver_bin_center1 - receiver_bin_width/2.
        bin1_endZ   = source_z + receiver_bin_center1 + receiver_bin_width/2.
        bin1_startX = source_x - receiver_bin_width/2.
        bin1_endX   = source_x + receiver_bin_width/2.

        receiver_coordinates = spyro.create_2d_grid(bin1_startZ, bin1_endZ, bin1_startX, bin1_endX, int(np.sqrt(receiver_quantity)))

    if receiver_type == 'near' and experiment_type == 'heterogeneous':

        # time calculations
        tmin = 1./frequency
        final_time = 25*tmin #should be 35

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

    
    if receiver_type == 'line' and experiment_type == 'heterogeneous':

        # time calculations
        tmin = 1./frequency
        final_time = 2*10*tmin + 5.0 #should be 35

        # receiver calculations

        receiver_bin_center1 = 2000.0/1000
        receiver_bin_center2 = 10000.0/1000
        receiver_quantity = 500

        bin1_startZ = source_z 
        bin1_endZ   = source_z 
        bin1_startX = source_x + receiver_bin_center1
        bin1_endX   = source_x + receiver_bin_center2

        receiver_coordinates = spyro.create_transect( (bin1_startZ, bin1_startX), (bin1_endZ, bin1_endX), receiver_quantity)


    elif receiver_type == 'far':
        raise ValueError('Far receivers minimum grid point calculation experiment not implemented because of computational limits.')
    
    # Choose method and parameters
    model["opts"] = {
        "method": method,
        "variant": None,
        "element": "tria",  # tria or tetra
        "degree": degree,  # p order
        "dimension": dimension,  # dimension
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
        "ly": pady,  # thickness of the pml in the y-direction (km) - always positive
    }

    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
        "meshfile": "demos/mm_exact.msh",
        "initmodel": "velocity_models/bp2004.hdf5",
        "truemodel": "velocity_models/bp2004.hdf5",
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": source_coordinates,
        "source_mesh_point": False,
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

def create_model_for_grid_point_calculation3D(degree):
    
    model = {}
    frequency = 5.0
    minimum_mesh_velocity = 1.429
    lbda = minimum_mesh_velocity/frequency
    pad = lbda
    Lz = 15*lbda#100*lbda
    Real_Lz = Lz+ pad
    #print(Real_Lz)
    Lx = 30*lbda#90*lbda
    Real_Lx = Lx+ 2*pad
    Ly = Lx
    Real_Ly = Ly + 2*pad

    # source location
    source_z = -Real_Lz/2.#1.0
    #print(source_z)
    source_x = lbda*1.5
    source_y = Real_Ly/2.0
    source_coordinates = [(source_z, source_x, source_y)] #Source at the center. If this is changes receiver's bin has to also be changed.
    padz = pad
    padx = pad
    pady = pad

    # time calculations
    tmin = 1./frequency
    final_time = 20*tmin #should be 35

    # receiver calculations

    receiver_bin_center1 = 10*lbda#20*lbda
    receiver_bin_width = 5*lbda#15*lbda
    receiver_quantity = 36#2500 # 50 squared

    bin1_startZ = source_z - receiver_bin_width/2.
    bin1_endZ   = source_z + receiver_bin_width/2.
    bin1_startX = source_x + receiver_bin_center1 - receiver_bin_width/2.
    bin1_endX   = source_x + receiver_bin_center1 + receiver_bin_width/2.
    bin1_startY = source_y - receiver_bin_width/2.
    bin1_endY   = source_y + receiver_bin_width/2.

    receiver_coordinates = create_3d_grid( (bin1_startZ,bin1_startX,bin1_startY)  , (bin1_endZ,bin1_endX,bin1_endY)   , 6)
    # Choose method and parameters
    model["opts"] = {
        "method": 'KMV',
        "variant": None,
        "element": "tetra",  # tria or tetra
        'quadrature': 'KMV',
        "degree": degree,  # p order
        "dimension": 3,  # dimension
    }

    model["BCs"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
        "exponent": 1,
        "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
        "R": 0.001,  # theoretical reflection coefficient
        "lz": padz,  # thickness of the pml in the z-direction (km) - always positive
        "lx": padx,  # thickness of the pml in the x-direction (km) - always positive
        "ly": pady,  # thickness of the pml in the y-direction (km) - always positive
    }

    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": Ly,  # thickness in km - always positive
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": source_coordinates,
        "frequency": frequency,
        "delay": 1.0,
        "num_receivers": receiver_quantity,
        "receiver_locations": receiver_coordinates,
    }

    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": final_time,  # Final time for event
        "dt": 0.0002,  # timestep size
        "nspool": 200,  # how frequently to output solution to pvds
        "fspool": 100,  # how frequently to save solution to RAM
    }  
    model["parallelism"] = {
    "type": "spatial", 
    }

    # print(source_coordinates)
    # print(receiver_coordinates)
    return model

