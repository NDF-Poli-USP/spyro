import spyro
from spyro.io.model_parameters import model_parameters
from copy import deepcopy

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped', # lumped, equispaced or DG, default is lumped
    "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
}
dictionary["synthetic_data"] = {    #For use only if you are using a synthetic test model
    "real_mesh_file": None,
    "real_velocity_file": None,
}
dictionary["inversion"] = {
    "perform_fwi": False, # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 0.9), 20
    ),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  #  Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}

def test_method_reader():
    test_dictionary = deepcopy(dictionary)
    test_dictionary["options"] = {
    "cell_type": None,  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": None, # lumped, equispaced or DG, default is lumped
    "method": None, # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    }
    # Trying out different method entries and seeing if all of them work for MLT
    test1 = False
    test_dictionary["options"]["method"] = 'MLT'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'mass_lumped_triangle':
        test1 = True

    test2 = False
    test_dictionary["options"]["method"] = 'KMV'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'mass_lumped_triangle':
        test2 = True

    test3 = False
    test_dictionary["options"]["method"] = 'mass_lumped_triangle'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'mass_lumped_triangle':
        test3 = True

    # Trying out different method entries for spectral quads
    test4 = False
    test_dictionary["options"]["method"] = 'spectral_quadrilateral'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'spectral_quadrilateral':
        test4 = True

    test5 = False
    test_dictionary["options"]["method"] = 'CG'
    test_dictionary["options"]["variant"] = 'GLL'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'spectral_quadrilateral':
        test5 = True
    
    test6 = False
    test_dictionary["options"]["method"] = 'SEM'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'spectral_quadrilateral':
        test6 = True
    
    #Trying out some entries for other less used methods
    test7 = False
    test_dictionary["options"]["method"] = 'DG_triangle'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'DG_triangle':
        test7 = True
    
    test8 = False
    test_dictionary["options"]["method"] = 'DG_quadrilateral'
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'DG_quadrilateral':
        test8 = True
    
    test9 = False
    test_dictionary["options"]["method"] = 'CG'
    test_dictionary["options"]["variant"] = None
    model = model_parameters(dictionary=test_dictionary)
    if model.method == 'CG':
        test9 = True
    
    assert all([test1, test2, test3,test4, test5, test6, test7, test8, test9])
    
def test_cell_type_reader():
    ct_dictionary = deepcopy(dictionary)
    ct_dictionary["options"] = {
    "cell_type": None,  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": None, # lumped, equispaced or DG, default is lumped
    "method": None, # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    }
    # Testing lumped cases
    ct_dictionary["options"]["variant"] = 'lumped'

    test1 = False
    ct_dictionary["options"]["cell_type"] = 'triangle'
    model = model_parameters(dictionary=ct_dictionary)
    if model.method == 'mass_lumped_triangle':
        test1 = True
    
    test2 = False
    ct_dictionary["options"]["cell_type"] = 'quadrilateral'
    model = model_parameters(dictionary=ct_dictionary)
    if model.method == 'spectral_quadrilateral':
        test2 = True
    
    # Testing equispaced cases
    ct_dictionary["options"]["variant"] = 'equispaced'

    test3 = False
    ct_dictionary["options"]["cell_type"] = 'triangle'
    model = model_parameters(dictionary=ct_dictionary)
    if model.method == 'CG_triangle':
        test3 = True
    
    test4 = False
    ct_dictionary["options"]["cell_type"] = 'quadrilateral'
    model = model_parameters(dictionary=ct_dictionary)
    if model.method == 'CG_quadrilateral':
        test4 = True
    
    # Testing DG cases
    ct_dictionary["options"]["variant"] = 'DG'

    test5 = False
    ct_dictionary["options"]["cell_type"] = 'triangle'
    model = model_parameters(dictionary=ct_dictionary)
    if model.method == 'DG_triangle':
        test5 = True
    
    test6 = False
    ct_dictionary["options"]["cell_type"] = 'quadrilateral'
    model = model_parameters(dictionary=ct_dictionary)
    if model.method == 'DG_quadrilateral':
        test6 = True
    
    assert all([test1, test2, test3, test4, test5, test6])


print('END')