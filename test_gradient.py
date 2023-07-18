import numpy as np
from firedrake import *
import spyro
import pytest


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": 'lumped', # lumped, equispaced or DG, default is lumped
    "method": "MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
    "automatic_adjoint": False,
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
    "mesh_file": "test/meshes/Uniform2D.msh",
}
dictionary["synthetic_data"] = {    #For use only if you are using a synthetic test model or a forward only simulation -adicionar discrição para modelo direto
    "real_mesh_file": None,
    "real_velocity_file": None,
}
dictionary["inversion"] = {
    "perform_fwi": False, # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": None,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
}

# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(1.5, -0.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect(
        (0.1, -2.90), (2.9, -2.90), 100
    ),
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  #  Initial time for event
    "final_time": 1.00,  # Final time for event
    "dt": 0.001,  # timestep size
    "output_frequency": 9999,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}
dictionary["visualization"] = {
    "forward_output" : True,
    "gradient_output": True,
}


# outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")

functional = spyro.utils.compute_functional


def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0
        + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
        # 5.0 + 0.5 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
    )
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)
    return vp_guess


def test_gradient(dictionary):

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    V = Wave_obj.function_space
    mesh = Wave_obj.mesh

    vp_exact = _make_vp_exact(V, mesh)

    mask = Function(V).assign(1.0)

    vp_guess = _make_vp_guess(V, mesh)

    # simulate the exact model
    Wave_obj.set_initial_velocity_model(velocity_model_function=vp_exact)
    Wave_obj.forward_solve()

    p_exact = Wave_obj.forward_solution
    p_exact_receivers = Wave_obj.forward_solution_receivers

    # simulate the guess model
    Wave_obj.current_time = 0.0
    Wave_obj.set_initial_velocity_model(velocity_model_function=vp_guess)
    Wave_obj.forward_solve()
    
    p_guess = Wave_obj.forward_solution
    p_guess_receivers = Wave_obj.forward_solution_receivers

    quad_rule = Wave_obj.quadrature_rule

    Jm = functional(Wave_obj, misfit)
    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # compute the gradient of the control (to be verified)
    dJ = Wave_obj.gradient()
    dJ.dat.data[:] = dJ.dat.data[:]*mask.dat.data[:]
    File("gradient.pvd").write(dJ)

    steps = [1e-3, 1e-4, 1e-5]  # , 1e-6]  # step length

    delta_m = Function(V)  # model direction (random)
    delta_m.assign(dJ)

    # this deepcopy is important otherwise pertubations accumulate
    vp_original = vp_guess.copy(deepcopy=True)

    errors = []
    for step in steps:  # range(3):
        # steps.append(step)
        # perturb the model and calculate the functional (again)
        # J(m + delta_m*h)
        Wave_obj.current_time = 0.0
        vp_guess = vp_original + step * delta_m
        Wave_obj.set_initial_velocity_model(velocity_model_function=vp_guess)
        _, p_guess_recv = Wave_obj.forward_solve()

        Jp = functional(Wave_obj, p_exact_receivers - p_guess_recv)
        projnorm = assemble(mask * dJ * delta_m * dx(scheme=quad_rule))
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

        errors.append(100 * ((fd_grad - projnorm) / projnorm))
        # step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 5.0).all()


if __name__ == "__main__":
    test_gradient(dictionary)
