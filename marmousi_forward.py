import spyro
from copy import deepcopy
from firedrake import VTKFile
import sys


degree = 4
frequency = 5.0
final_time = 4.0


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": degree,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "length_z": 3.5,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "length_x": 17.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "output_filename": "trial01.msh",
    "cells_per_wavelength": 3.0,
    "frequency": frequency,
    "segy_velocity_model": "velocity_models/vp_marmousi-ii.segy",
    "mesh_type": "gmsh_mesh",
    "grade": 0.05,
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.01, 4.0), (-0.01, 12.0), 16),
    "frequency": frequency,
    "delay": 1.0/frequency,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-0.1, 4.0), (-0.1, 12.0), 100),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}
dictionary["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_output.pvd",
}


def test_real_shot_record_generation_parallel():
    real_wave = spyro.AcousticWave(dictionary=dictionary)
    real_wave.set_initial_velocity_model(new_file="velocity_models/vp_marmousi-ii.segy", fast_interpolate=True)
    VTKFile("vp.pvd").write(real_wave.c)
    real_wave.forward_solve()
    spyro.io.save_shots(real_wave)


if __name__ == "__main__":
    test_real_shot_record_generation_parallel()
