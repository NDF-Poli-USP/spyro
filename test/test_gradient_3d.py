import os

from firedrake import *

import spyro

from .inputfiles.Model1_Leapfrog_adjoint_3d import model


def test_gradient_3d():

    comm = spyro.utils.mpi_init(model)

    mesh, V = spyro.io.read_mesh(model, comm)

    num_sources = model["acquisition"]["num_sources"]

    # Create a circle with higher velocity in the center
    z, x, y = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0 + (0.002 - sqrt((z + 0.25) ** 2 + (x - 0.25) ** 2 + (y - 0.25) ** 2))
    )
    File("exact_vel.pvd").write(vp_exact)
    # The guess is a uniform velocity of 4.0 km/s
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)

    vp_guess_2 = Function(V)
    dJ = Function(V)

    sources = spyro.Sources(model, mesh, V, comm).create()
    receivers = spyro.Receivers(model, mesh, V, comm).create()

    Jtmp = 0.0
    # Compute the gradient of the functional
    for isour in range(num_sources):
        p_exact, p_exact_recv, _ = spyro.solvers.Leapfrog(
            model, mesh, comm, vp_exact, sources, receivers, source_num=isour
        )
        p_guess, p_guess_recv, psi_guess = spyro.solvers.Leapfrog(
            model, mesh, comm, vp_guess, sources, receivers, source_num=isour
        )
        residual = spyro.utils.evaluate_misfit(model, comm, p_guess_recv, p_exact_recv)
        Jtmp += spyro.utils.compute_functional(model, comm, residual)
        grad = spyro.solvers.Leapfrog_adjoint(
            model, mesh, comm, vp_guess, p_guess, residual, psi_sol=psi_guess
        )
        dJ.dat.data[:] += grad.dat.data[:]

    # for visualization only
    outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")

    # Normalize the gradient so that it has a len of 1.0
    norm2_dJ = norm(dJ) ** 2
    dJ.dat.data[:] /= norm2_dJ
    outfile_total_gradient.write(dJ, name="TotalGradient")

    # The total functional from all shots
    J0 = Jtmp
    print("Cost functional: " + str(J0))

    # fileout = open("grad_fin_dif_test.txt", "w")
    epsilon = 1e-10
    vp_guess_2.dat.data[:] = vp_guess.dat.data[:] + epsilon * dJ.dat.data[:]
    File("vp_guess2.pvd").write(vp_guess_2)
    J = 0
    for isour in range(num_sources):
        p_guess, p_guess_recv, _ = spyro.solvers.Leapfrog(
            model, mesh, comm, vp_guess_2, sources, receivers, source_num=isour
        )
        residual = spyro.utils.evaluate_misfit(model, comm, p_guess_recv, p_exact_recv)
        J += spyro.utils.compute_functional(model, comm, residual)

    assert (J - J0) < 1e-8
    # fileout.write(str(epsilon) + " " + str(J - J0) + "\n")
    # fileout.close


if __name__ == "__main__":
    test_gradient()
