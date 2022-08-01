import spyro

default_optimization_parameters = 

default_dictionary = {}
default_dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedrals (T) or quadrilaterals (Q)
    "variant": 'lumped', # lumped, equispaced or DG, default is lumped
    "method": "MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
default_dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
default_dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
}
default_dictionary["synthetic_data"] = {    #For use only if you are using a synthetic test model
    "real_mesh_file": None,
    "real_velocity_file": None,
}
default_dictionary["inversion"] = {
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": default_optimization_parameters,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
default_dictionary["absorving_boundary_conditions"] = {
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
default_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 0.9), 20
    ),
}

# Simulate for 2.0 seconds.
default_dictionary["time_axis"] = {
    "initial_time": 0.0,  #  Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}

class model_parameters:
    def __init__(self, dictionary=default_dictionary):
        self.input_dictionary = dictionary
        self.method = model_parameters.method
        self.degree = model_parameters.degree
        self.dimension = model_parameters.dimension
        self.final_time = model_parameters.final_time
        self.dt = model_parameters.dt
        self.initial_velocity_model = model_parameters.get_initial_velocity_model()
        self.function_space = None
        self.foward_output_file = 'forward_output.pvd'
        self.current_time = 0.0
        self.solver_parameters = model_parameters.solver_parameters
        self.c = self.initial_velocity_model

        @property
        def method(self):
            return 
        
