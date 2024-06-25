import numpy as np
import os
from firedrake import *
import spyro
from spyro.domains import quadrature
import pytest

from .inputfiles.Model1_gradient_2d import model
from .inputfiles.Model1_gradient_2d_pml import model_pml


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "method": "MLT",  # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
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
    "Lz": 1.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
}
dictionary[
    "synthetic_data"
] = {  # For use only if you are using a synthetic test model or a forward only simulation -adicionar discrição para modelo direto
    "real_mesh_file": None,
    "real_velocity_file": None,
}
dictionary["inversion"] = {
    "perform_fwi": False,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": None,
}

# Specify a 250-m PML on the three sides of the domain to damp outgoing waves.
dictionary["absorving_boundary_conditions"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
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
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
    "adjoint_output": False,
    "adjoint_filename": None,
}
outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")

# forward = spyro.solvers.forward
# gradient = spyro.solvers.gradient
functional = spyro.utils.compute_functional


def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
    )
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_exact_pml(V, mesh):
    """Create a half space"""
    z, x = SpatialCoordinate(mesh)
    velocity = conditional(z > -0.5, 1.5, 4.0)
    vp_exact = Function(V, name="vp").interpolate(velocity)
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)
    return vp_guess


@pytest.mark.skip(reason="not yet implemented")
def test_gradient():
    _test_gradient(model)


@pytest.mark.skip(reason="no way of currently testing this")
def test_gradient_pml():
    _test_gradient(model_pml, pml=True)


def _test_gradient(options, pml=False):
    comm = spyro.utils.mpi_init(options)

    mesh, V = spyro.basicio.read_mesh(options, comm)

    if pml:
        vp_exact = _make_vp_exact_pml(V, mesh)
        z, x = SpatialCoordinate(mesh)
        Lx = model_pml["mesh"]["Lx"]
        Lz = model_pml["mesh"]["Lz"]
        x1 = 0.0
        z2 = -Lz
        boxx1 = Function(V).interpolate(conditional(x > x1, 1.0, 0.0))
        boxx2 = Function(V).interpolate(conditional(x < Lx, 1.0, 0.0))
        boxz1 = Function(V).interpolate(conditional(z > z2, 1.0, 0.0))
        mask = Function(V).interpolate(boxx1 * boxx2 * boxz1)
        File("mask.pvd").write(mask)
    else:
        vp_exact = _make_vp_exact(V, mesh)

        mask = Function(V).assign(1.0)

    vp_guess = _make_vp_guess(V, mesh)

    sources = spyro.Sources(options, mesh, V, comm)

    receivers = spyro.Receivers(options, mesh, V, comm)

    wavelet = spyro.full_ricker_wavelet(
        options["timeaxis"]["dt"],
        options["timeaxis"]["tf"],
        options["acquisition"]["frequency"],
    )

    # simulate the exact model
    p_exact, p_exact_recv = forward(
        options,
        mesh,
        comm,
        vp_exact,
        sources,
        wavelet,
        receivers,
    )

    # simulate the guess model
    p_guess, p_guess_recv = forward(
        options,
        mesh,
        comm,
        vp_guess,
        sources,
        wavelet,
        receivers,
    )

    misfit = p_exact_recv - p_guess_recv

    qr_x, _, _ = quadrature.quadrature_rules(V)

    Jm = functional(options, misfit)
    print("\n Cost functional at fixed point : " + str(Jm) + " \n ")

    # compute the gradient of the control (to be verified)
    dJ = gradient(options, mesh, comm, vp_guess, receivers, p_guess, misfit)
    dJ.dat.data[:] = dJ.dat.data[:] * mask.dat.data[:]
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
        vp_guess = vp_original + step * delta_m
        _, p_guess_recv = forward(
            options,
            mesh,
            comm,
            vp_guess,
            sources,
            wavelet,
            receivers,
        )

        Jp = functional(options, p_exact_recv - p_guess_recv)
        projnorm = assemble(mask * dJ * delta_m * dx(scheme=qr_x))
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
    test_gradient(model)
