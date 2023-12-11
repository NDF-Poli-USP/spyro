import numpy as np
import os
import spyro
import pytest


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "method": "MLT",  # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
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
    "Lz": 1.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
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
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 0.9), 20),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 2.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
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
}

functional = spyro.utils.compute_functional


def test_gradient():

    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(mesh_parameters={"dx": 0.02})
    Wave_obj_exact.set_initial_velocity_model(
        expression="4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))"
    )
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.receivers_output

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(mesh_parameters={"dx": 0.02})
    Wave_obj_guess.set_initial_velocity_model(constant=4.0)
    Wave_obj_guess.forward_solve()
    rec_out_guess = Wave_obj_guess.receivers_output

    misfit = rec_out_exact - rec_out_guess

    # Jm = functional(options, misfit)
    # print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # # compute the gradient of the control (to be verified)
    # dJ = gradient(options, mesh, comm, vp_guess, receivers, p_guess, misfit)
    # dJ.dat.data[:] = dJ.dat.data[:] * mask.dat.data[:]
    # File("gradient.pvd").write(dJ)

    # steps = [1e-3, 1e-4, 1e-5]  # , 1e-6]  # step length

    # delta_m = Function(V)  # model direction (random)
    # delta_m.assign(dJ)

    # # this deepcopy is important otherwise pertubations accumulate
    # vp_original = vp_guess.copy(deepcopy=True)

    # errors = []
    # for step in steps:  # range(3):
    #     # steps.append(step)
    #     # perturb the model and calculate the functional (again)
    #     # J(m + delta_m*h)
    #     vp_guess = vp_original + step * delta_m
    #     _, p_guess_recv = forward(
    #         options,
    #         mesh,
    #         comm,
    #         vp_guess,
    #         sources,
    #         wavelet,
    #         receivers,
    #     )

    #     Jp = functional(options, p_exact_recv - p_guess_recv)
    #     projnorm = assemble(mask * dJ * delta_m * dx(scheme=qr_x))
    #     fd_grad = (Jp - Jm) / step
    #     print(
    #         "\n Cost functional for step "
    #         + str(step)
    #         + " : "
    #         + str(Jp)
    #         + ", fd approx.: "
    #         + str(fd_grad)
    #         + ", grad'*dir : "
    #         + str(projnorm)
    #         + " \n ",
    #     )

    #     errors.append(100 * ((fd_grad - projnorm) / projnorm))
    #     # step /= 2

    # # all errors less than 1 %
    # errors = np.array(errors)
    # assert (np.abs(errors) < 5.0).all()


if __name__ == "__main__":
    test_gradient()
