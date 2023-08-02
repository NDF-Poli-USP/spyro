from pyclbr import Function
import spyro
from firedrake import RectangleMesh, conditional, UnitSquareMesh
from spyro.habc import eikonal
import firedrake as fire

from spyro.io.model_parameters import Model_parameters

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped', # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "user_mesh": None,
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.3)],
    "frequency": 5.0,
    "delay": 1.5,
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
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output" : True,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

Model = Model_parameters(dictionary=dictionary) 

user_mesh = UnitSquareMesh(100,100)
user_mesh.coordinates.dat.data[:,0] *= -1.0
Model.set_mesh(user_mesh=user_mesh)

Wave = spyro.AcousticWave(model_parameters=Model)

x,y = Wave.get_spatial_coordinates()
# V = fire.FunctionSpace(Wave.mesh, 'CG',1)
# ux = fire.Function(V).interpolate(x)
# uy = fire.Function(V).interpolate(y)
# print(ux.dat.data[:])
# print(uy.dat.data[:])

sources = Wave.sources
# print(sources.receiver_locations)
# print(sources.cellIDs)
# print(sources.cellNodeMaps)


Wave.set_initial_velocity_model(conditional = conditional(x < -0.5 ,1.5,3.0 ))
Wave._get_initial_velocity_model()
Wave.c = Wave.initial_velocity_model

# eikonal(Wave)


Wave.forward_solve()



print("END")

