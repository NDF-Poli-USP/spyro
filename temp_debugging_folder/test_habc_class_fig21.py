import spyro
from firedrake import RectangleMesh, conditional, UnitSquareMesh, Function, FunctionSpace, File
from spyro.habc import HABC
import firedrake as fire
import numpy as np

from spyro.io.model_parameters import Model_parameters

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
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
    "Lz": 3.,  # depth in km - always positive
    "Lx": 9.2,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "user_mesh": None,
    # "mesh_file": "meshes/marmousi_f5.0_degree4_pad0.0.msh",  # if you want to use a mesh file, specify it here
}
dictionary["synthetic_data"] = {    #For use only if you are using a synthetic test model or a forward only simulation -adicionar discrição para modelo direto
    "real_velocity_file": "velmodel.segy",
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.025,9.2-5.65)],# (-0.025, 2.0)],#[(-0.6, 4.8-1.68)],#, (-0.605, 1.7), (-0.61, 1.7), (-0.615, 1.7)],#, (-0.1, 1.5), (-0.1, 2.0), (-0.1, 2.5), (-0.1, 3.0)],
    "frequency": 6.0,
    "delay": 1.5,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 4.0), 20
    ),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.0003,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
}#342 9.2x742km 12,5m

dictionary["visualization"] = {
    "forward_output" : True,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

Model = Model_parameters(dictionary=dictionary)


Lx = 9.2
Lz = 3.0
user_mesh = fire.RectangleMesh(int(Lz*12), int(Lx*12), Lz, Lx, diagonal="crossed")
user_mesh.coordinates.dat.data[:, 0] *= -1.0
Model.set_mesh(user_mesh=user_mesh)
# Wave_no_habc.forward_solve()

Wave_no_habc = spyro.AcousticWave(model_parameters=Model)
Wave_no_habc._get_initial_velocity_model()
Wave_no_habc.c = Wave_no_habc.initial_velocity_model
habc = HABC(Wave_no_habc, h_min=0.0125)


print("END")


