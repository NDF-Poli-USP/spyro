import spyro
from spyro.habc import HABC
import firedrake as fire
import math


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 2,  # p order
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
    "mesh_type": "firedrake_mesh",
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
# We also specify to record the solution at a microphone near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.5, 0.25)],#, (-0.605, 1.7), (-0.61, 1.7), (-0.615, 1.7)],#, (-0.1, 1.5), (-0.1, 2.0), (-0.1, 2.5), (-0.1, 3.0)],
    "frequency": 5.0,
    "delay": 1.5,
    "receiver_locations": spyro.create_transect(
        (-0.10, 0.1), (-0.10, 0.9), 20
    ),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
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

Wave_no_habc = spyro.AcousticWave(dictionary=dictionary)

Lx = 1
Lz = 1
user_mesh = fire.RectangleMesh(120, 120, Lz, Lx, diagonal="crossed")
user_mesh.coordinates.dat.data[:, 0] *= -1.0
z, x = fire.SpatialCoordinate(user_mesh)

cond = fire.conditional(x < 0.5, 3.0, 1.5)
        

Wave_no_habc.set_mesh(user_mesh=user_mesh)

Wave_no_habc.set_initial_velocity_model(conditional=cond)
Wave_no_habc._get_initial_velocity_model()

Wave_no_habc.c = Wave_no_habc.initial_velocity_model

habc = HABC(Wave_no_habc, h_min=0.00833)

eikonal = habc.eikonal

min_value = eikonal.min_value
max_value = eikonal.max_value

paper_min = 0.085
paper_max = 0.56

test_min = math.isclose(min_value, paper_min, rel_tol=0.1)
test_max = math.isclose(max_value, paper_max, rel_tol=0.2)

# Verificar valores das distancias como lref e velocidades

print("END")

# Final time
# Lf = F_Inf*F_wl*lmbda/AspRatio = cmin*T = lmbda*f0*T, lmbda = c/f0
# tf = round(pH['F_Inf'] * pH['F_wl'] / (pH['f0'] * factGeom), 4)

# R = Lx/Ly
# Fig 8 : F_Inf = 1 + 3/R, R=2
# Mar: F_Inf = 1 + 2.3 /R, R = 9.2/3

# --ACU--
# *INI
# Options for F_L: [0.3196, 0.4616, 0.4972, 0.6747, 0.7102]
# Elements for F_L: [9, 13, 14, 19, 20] 
# *FIN
# Options for F_L: [0.8878, 1.2429, 1.3494, 1.8821, 1.9531]
# Elements for F_L: [25, 35, 38, 53, 55] 


# Coisas para fazer: Tentar colocar isso no modelo