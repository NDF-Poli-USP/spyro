import numpy as np

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


def create_model_for_grid_point_calculation(scale):
    
    model = {}
    frequency = 5.0
    minimum_mesh_velocity = 1.429
    lbda = minimum_mesh_velocity/frequency
    pad = scale*lbda
    Lz = scale*15*lbda#100*lbda
    Real_Lz = Lz+ pad
    #print(Real_Lz)
    Lx = scale*30*lbda#90*lbda
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


    receiver_coordinates = create_3d_grid( (bin1_startZ,bin1_startX,bin1_startY)  , (bin1_endZ,bin1_endX,bin1_endY)   , int(np.sqrt(receiver_quantity)))
    # Choose method and parameters
    model["opts"] = {
        "method": 'KMV',
        "variant": None,
        "element": "tetra",  # tria or tetra
        "degree": 3,  # p order
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
        "dt": 0.001,  # timestep size
        "nspool": 200,  # how frequently to output solution to pvds
        "fspool": 100,  # how frequently to save solution to RAM
    }  
    model["parallelism"] = {
    "type": "automatic", 
    }

    # print(source_coordinates)
    # print(receiver_coordinates)
    return model