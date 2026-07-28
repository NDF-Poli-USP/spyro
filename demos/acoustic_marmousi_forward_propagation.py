"""Demo script for running a marmousi forward case.

This case is without mesh generation. Therefore, we need the mesh file.

This demo has automatic parallelism set up and 1 shots, therefore we need
a 1n (n positive integer) number of cores in mpiexec -n N_CORES to run. 
You can experiment with a different number of cores if desired, but you
will probably only notice an improvement up to 2 cores. In my computer the
runtimes for each case are:
Core count    |   runtime (s)
1             |  14.45
2             |  10.87
3             |  10.46
4             |  11.00

Look at the rule of thumb proposed on https://www.firedrakeproject.org/parallelism.html
to figure out why.

The script just runs a single acoustic marmousi forward propagation.
"""

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
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.01, 8.0)],
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
    real_dictionary = deepcopy(dictionary)
    real_dictionary["mesh"]["mesh_file"] = "meshes/real5hz.msh"

    real_wave = spyro.AcousticWave(dictionary=real_dictionary)
    real_wave.set_initial_velocity_model(new_file="velocity_models/vp_marmousi-ii.segy", fast_interpolate=True)
    real_wave.forward_solve()
    VTKFile("vp.pvd").write(real_wave.c)
    spyro.io.save_shots(real_wave)


if __name__ == "__main__":
    test_real_shot_record_generation_parallel()
