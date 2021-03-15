import numpy as np
import spyro

def create_model_for_grid_point_calculation(frequency, degree, method, minimum_mesh_velocity, experiment_type = 'homogeneous', receiver_type = 'near'):
    ''' Creates models  with the correct parameters for for grid point calculation experiments.
    '''
    model = {}
    # domain calculations
    lbda = minimum_mesh_velocity/frequency
    pml_fraction = lbda
    if receiver_type == 'near':
        Lz = 200*lbda
        Real_Lz = Lz*(1. + 2*pml_fraction)
        Lx = 150*lbda
        Real_Lx = Lx*(1. + 1*pml_fraction)

        # source location
        center_coordinates = [(Real_Lz/2, Real_Lx/2)] #Source at the center. If this is changes receiver's bin has to also be changed.

        # time calculations
        tmin = 1./frequency
        final_time = 40*tmin #should be 35

        # receiver calculations

        receiver_bin_center1 = 20*lbda
        receiver_bin_width = 15*lbda
        receiver_quantity = 2500 # 50 squared

        bin1_startZ = Real_Lz/2. + receiver_bin_center1 - receiver_bin_width/2.
        bin1_endZ   = Real_Lz/2. + receiver_bin_center1 + receiver_bin_width/2.
        bin1_startX = Real_Lx/2. - receiver_bin_width/2.
        bin1_endX   = Real_Lx/2. + receiver_bin_width/2.

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
        "lz": Lz*pml_fraction,  # thickness of the pml in the z-direction (km) - always positive
        "lx": Lx*pml_fraction,  # thickness of the pml in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
    }

    model["mesh"] = {
        "Lz": Lz,  # depth in km - always positive
        "Lx": Lx,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "demos/mm_exact.msh",
        "initmodel": "demos/mm_init.hdf5",
        "truemodel": "demos/mm_exact.hdf5",
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": center_coordinates,
        "frequency": frequency,
        "delay": 1.0,
        "num_receivers": receiver_quantity,
        "receiver_locations": spyro.create_2d_grid(bin1_startZ, bin1_endZ, bin1_startX, bin1_endX, int(np.sqrt(receiver_quantity)))
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
        'experiment_type': 'homogeneous',
        'minimum_mesh_velocity': minimum_mesh_velocity,
        'pml_fraction': pml_fraction,
        'receiver_type': receiver_type
    }

    return model

