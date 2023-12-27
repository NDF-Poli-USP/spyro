import numpy as np
import matplotlib.pyplot as plt
from firedrake import File
import firedrake as fire
import spyro


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "method": "MLT",  # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 1,  # p order
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
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.5, 1.5)],
    "frequency": 5.0,
    "delay": 1.5,
    "delay_type": "multiples_of_minimun",
    "receiver_locations": spyro.create_transect((-2.9, 0.1), (-2.9, 2.9), 100),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": True,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": True,
}


def test_gradient():

    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(mesh_parameters={"dx": 0.05})
    Wave_obj_exact.set_initial_velocity_model(
        expression="4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((x - 1.5) ** 2 + (z + 1.5) ** 2)))",
        output=True
    )
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.receivers_output

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(mesh_parameters={"dx": 0.05})
    Wave_obj_guess.set_initial_velocity_model(constant=3.0)
    Wave_obj_guess.forward_solve()
    forward_solution = Wave_obj_guess.forward_solution
    rec_out_guess = Wave_obj_guess.receivers_output

    misfit = rec_out_exact - rec_out_guess
    misfit_old = np.load("misfit_old.npy")

    Jm = spyro.utils.compute_functional(Wave_obj_guess, misfit_old)
    print(f"Cost functional : {Jm}")

    # compute the gradient of the control (to be verified)
    dJ = Wave_obj_guess.gradient_solve(misfit=misfit_old, forward_solution=forward_solution)
    # dJ.dat.data[:] = dJ.dat.data[:] * mask.dat.data[:]
    File("gradient.pvd").write(dJ)

    steps = [1e-3, 1e-4, 1e-5]  # step length

    errors = []
    for step in steps:

        Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
        Wave_obj_guess.set_mesh(mesh_parameters={"dx": 0.05})
        Wave_obj_guess.set_initial_velocity_model(constant=3.0)
        Wave_obj_guess.initial_velocity_model = Wave_obj_guess.initial_velocity_model + step*dJ
        Wave_obj_guess.forward_solve()
        forward_solution = Wave_obj_guess.forward_solution
        rec_out_guess = Wave_obj_guess.receivers_output

        Jp = spyro.utils.compute_functional(Wave_obj_guess, rec_out_exact - rec_out_guess)
        projnorm = fire.assemble(dJ * dJ * fire.dx(scheme=Wave_obj_guess.quadrature_rule))
        fd_grad = (Jp - Jm) / step
        print(
            "\n Cost functional for step "
            + str(step)
            + " : "
            + str(Jp)
            + ", fd approx.: "
            + str(fd_grad)
            + ", grad'*dir : "
            + str(projnorm)
            + " \n ",
        )
        error = 100 * ((fd_grad - projnorm) / projnorm)

        errors.append(error)
        print(f"Error : {error}")
        # step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 5.0).all()
    print("END")


if __name__ == "__main__":
    test_gradient()
