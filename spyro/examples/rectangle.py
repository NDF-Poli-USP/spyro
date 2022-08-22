from spyro import create_transect
from spyro.examples.example_model import Example_model
from spyro import Wave
import firedrake as fire

rectangle_optimization_parameters = {
    "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
    "Step": {
        "Type": "Augmented Lagrangian",
        "Augmented Lagrangian": {
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 5.0,
        },
        "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
    },
    "Status Test": {
        "Gradient Tolerance": 1e-16,
        "Iteration Limit": None,
        "Step Tolerance": 1.0e-16,
    },
}

rectangle_dictionary = {}
rectangle_dictionary["options"] = {
    "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped', # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    "automatic_adjoint": False,
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
rectangle_dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
rectangle_dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "user_mesh": None,
}
rectangle_dictionary["synthetic_data"] = {    #For use only if you are using a synthetic test model or a forward only simulation -adicionar discrição para modelo direto
    "real_mesh_file": None,
    "real_velocity_file": None,
    "velocity_conditional": None,
}
rectangle_dictionary["inversion"] = {
    "perform_fwi": False, # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": rectangle_optimization_parameters,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
rectangle_dictionary["absorving_boundary_conditions"] = {
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
rectangle_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.3)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": create_transect(
        (-0.10, 0.1), (-0.10, 0.9), 20
    ),
}

# Simulate for 2.0 seconds.
rectangle_dictionary["time_axis"] = {
    "initial_time": 0.0,  #  Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}

rectangle_dictionary["visualization"] = {
    "forward_output" : True,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}


class Rectangle_parameters(Example_model):
    def __init__(self, dictionary=None, example_dictionary= rectangle_dictionary, comm = None):
        super().__init__(dictionary=dictionary,default_dictionary=example_dictionary,comm=comm)
        self._rectangle_mesh()
        self._rectangle_velocity_model()
        self.velocity_model_type = "conditional"
    
    def _rectangle_mesh(self):
        nz = 100
        nx = 100
        Lz = self.Lz
        Lx = self.Lx
        if self.cell_type == 'quadrilateral':
            quadrilateral = True
        else:
            quadrilateral = False
        self.user_mesh = fire.RectangleMesh(nz, nx, Lz, Lx, quadrilateral=quadrilateral)
    
    def _rectangle_velocity_model(self):
        x, y = fire.SpatialCoordinate(self.mesh)
        xc = 0.5
        yc = 0.5
        rc = 0.5
        c_salt = 4.6
        c_not_salt = 1.6
        cond = fire.conditional( (x-xc)**2 + (y-yc)**2 < rc**2 ,  c_salt , c_not_salt)
        self.velocity_conditional=cond

class Rectangle(Wave):
    def __init__(self, model_dictionary = None, comm = None):     
        model_parameters = Rectangle_parameters(dictionary=model_dictionary, comm = comm)
        super().__init__(model_parameters = model_parameters, comm = comm)





