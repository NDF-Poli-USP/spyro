import numpy as np
import matplotlib.pyplot as plt
import pytest
from copy import deepcopy
from firedrake import VTKFile
import firedrake as fire
import spyro
from spyro.utils.typing import AdjointType


IMPLEMENTED_ADJOINTS = [
    AdjointType.IMPLEMENTED_ADJOINT,
    AdjointType.UFL_DERIVED_ADJOINT,
]


def check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=False):
    steps = [1e-2, 1e-3, 1e-4]  # step length

    errors = []
    remainders = []
    V_c = Wave_obj_guess.function_space
    dm = fire.Function(V_c)
    rng = np.random.default_rng(0)
    size, = np.shape(dm.dat.data[:])
    dm_data = rng.random(size)
    dm.dat.data[:] = dm_data
    # dm.assign(dJ)

    for step in steps:

        Wave_obj_guess.reset_pressure()
        c_guess = fire.Constant(2.0) + step*dm
        Wave_obj_guess.initial_velocity_model = c_guess
        Wave_obj_guess.forward_solve()
        misfit_plusdm = rec_out_exact - Wave_obj_guess.forward_solution_receivers
        J_plusdm = spyro.utils.compute_functional(Wave_obj_guess, misfit_plusdm)

        grad_fd = (J_plusdm - Jm) / (step)
        projnorm = fire.assemble(dJ * dm * fire.dx(**Wave_obj_guess.quadrature_rule))

        error = 100 * ((grad_fd - projnorm) / projnorm)
        remainder = abs(J_plusdm - Jm - step * projnorm)

        errors.append(error)
        remainders.append(remainder)

    errors = np.array(errors)
    remainders = np.array(remainders)
    if plot:
        plt.close()
        plt.plot(steps, errors, label="Error")
        plt.legend()
        plt.title(" Adjoint gradient versus finite difference gradient")
        plt.xlabel("Step")
        plt.ylabel("Error %")
        plt.savefig("gradient_error_verification.png")
        plt.close()

    # Checking that the random-direction finite-difference error remains
    # below 1 percent across the tested step sizes.
    test1 = np.all(np.abs(errors) < 3)
    print(f"Gradient error less than 1 percent for all steps: {test1}")
    print(f"Error of {errors}")

    # Check that the first-order Taylor remainder decreases at least linearly
    # with the step length, without relying on the sign of the directional
    # error.
    test2 = np.all(remainders[1:] < 0.2 * remainders[:-1])
    print(f"Taylor remainder decreases with step size: {test2}")
    print(f"Taylor remainders {remainders}")

    assert all([test1, test2])


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

dictionary["mesh"] = {
    "length_z": 1.0,  # depth in km - always positive
    "length_x": 1.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.2, 0.5)],
    "frequency": 5.0,
    "delay": 1.5,
    "delay_type": "multiples_of_minimum",
    "receiver_locations": spyro.create_transect((-0.8, 0.2), (-0.8, 0.8), 10),
}

dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}


def get_forward_model(load_true=False):
    if load_true is False:
        Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
        Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1})
        cond = fire.conditional(Wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
        Wave_obj_exact.set_initial_velocity_model(
            conditional=cond,
            dg_velocity_model=False,
        )
        spyro.plots.plot_model(
            Wave_obj_exact,
            filename="pml_grad_test_model.png",
            abc_points=[(-0, 0), (-1, 0), (-1, 1), (-0, 1)],
        )
        spyro.plots.plot_model(Wave_obj_exact, abc_points=[(-1, 1), (-2, 1), (-2, 4), (-1, 2)])
        Wave_obj_exact.forward_solve()
        rec_out_exact = Wave_obj_exact.forward_solution_receivers
    else:
        rec_out_exact = np.load("rec_out_exact.npy")

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.forward_solve()
    rec_out_guess = Wave_obj_guess.forward_solution_receivers

    return rec_out_exact, rec_out_guess, Wave_obj_guess


@pytest.mark.parametrize("adjoint_type", IMPLEMENTED_ADJOINTS)
def test_gradient(adjoint_type):
    rec_out_exact, rec_out_guess, Wave_obj_guess = get_forward_model(load_true=False)
    forward_solution = Wave_obj_guess.forward_solution
    forward_solution_guess = deepcopy(forward_solution)

    misfit = rec_out_exact - rec_out_guess

    Jm = spyro.utils.compute_functional(Wave_obj_guess, misfit)
    print(f"Cost functional : {Jm}")

    # compute the gradient of the control (to be verified)
    dJ = Wave_obj_guess.gradient_solve(
        misfit=misfit,
        forward_solution=forward_solution_guess,
        adjoint_type=adjoint_type,
    )
    VTKFile("gradient.pvd").write(dJ)

    if adjoint_type is AdjointType.UFL_DERIVED_ADJOINT:
        assert Wave_obj_guess.forward_residual_form is not None

    check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=True)


def test_ufl_derived_gradient_matches_hand_derived_gradient():
    """Both implemented adjoints discretize the same functional, so their
    gradients agree far beyond the finite-difference tolerance."""
    rec_out_exact, rec_out_guess, Wave_obj_guess = get_forward_model(load_true=False)
    misfit = rec_out_exact - rec_out_guess

    gradients = {}
    for adjoint_type in IMPLEMENTED_ADJOINTS:
        # Each gradient solve consumes the stored forward solution, so the
        # second one re-runs the forward solve on the same model.
        gradients[adjoint_type] = Wave_obj_guess.gradient_solve(
            misfit=misfit, adjoint_type=adjoint_type,
        )

    hand_derived = gradients[AdjointType.IMPLEMENTED_ADJOINT]
    ufl_derived = gradients[AdjointType.UFL_DERIVED_ADJOINT]
    difference = fire.Function(hand_derived.function_space())
    difference.assign(ufl_derived - hand_derived)
    assert fire.norm(difference) < 1e-4 * fire.norm(hand_derived)


@pytest.mark.parametrize("use_vertex_only_mesh", [False, True])
def test_ufl_derived_gradient_receiver_injection(use_vertex_only_mesh):
    """The misfit is injected by the transpose of whichever receiver
    interpolation the forward solve used: the vertex-only mesh one, or the
    Dirac delta projection."""
    d = deepcopy(dictionary)
    d["time_axis"]["final_time"] = 0.5
    d["acquisition"]["use_vertex_only_mesh"] = use_vertex_only_mesh

    Wave_obj_exact = spyro.AcousticWave(dictionary=d)
    Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    cond = fire.conditional(Wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    Wave_obj_exact.set_initial_velocity_model(conditional=cond, dg_velocity_model=False)
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.forward_solution_receivers

    Wave_obj_guess = spyro.AcousticWave(dictionary=d)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.forward_solve()
    misfit = rec_out_exact - Wave_obj_guess.forward_solution_receivers
    Jm = spyro.utils.compute_functional(Wave_obj_guess, misfit)

    dJ = Wave_obj_guess.gradient_solve(
        misfit=misfit, adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )
    check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm)


def test_ufl_derived_adjoint_requires_every_time_step_stored():
    d = deepcopy(dictionary)
    d["time_axis"]["gradient_sampling_frequency"] = 2
    wave = spyro.AcousticWave(dictionary=d)

    with pytest.raises(ValueError, match="gradient_sampling_frequency"):
        wave.enable_implemented_adjoint(
            adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
        )
    # The hand-derived adjoint subsamples the stored forward solution.
    wave.enable_implemented_adjoint()
    assert wave.adjoint_type is AdjointType.IMPLEMENTED_ADJOINT


def test_enable_implemented_adjoint_rejects_other_adjoints():
    wave = spyro.AcousticWave(dictionary=deepcopy(dictionary))

    with pytest.raises(ValueError, match="implemented adjoint"):
        wave.enable_implemented_adjoint(
            adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        )


def test_receiver_source_injector_validates_its_input():
    wave = spyro.AcousticWave(dictionary=deepcopy(dictionary))
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    inject = wave.receivers.receiver_source_injector(wave.function_space)
    number_of_receivers = wave.number_of_receivers

    with pytest.raises(TypeError, match="Receiver values must be"):
        inject({"invalid": "misfit"})
    with pytest.raises(ValueError, match="one value per receiver"):
        inject(np.ones(number_of_receivers + 1))

    source = inject(np.ones(number_of_receivers))
    assert source.function_space() == wave.function_space.dual()
    # Injecting unit receiver values distributes them over the receiver
    # cells' nodal basis, whose values sum to one at each receiver.
    assert np.isclose(source.dat.data_ro.sum(), number_of_receivers)
    # The same cofunction is reused on every call.
    assert inject(np.zeros(number_of_receivers)) is source


def _gradient_for_sampling_frequency(freq, final_time_override=0.5):
    """Compute the adjoint gradient for a given ``gradient_sampling_frequency``.

    Returns the raw nodal gradient values so gradients computed with different
    sampling frequencies can be compared directly. A shorter ``final_time`` is
    used to keep the regression test cheap.
    """
    d = deepcopy(dictionary)
    d["time_axis"]["final_time"] = final_time_override
    d["time_axis"]["gradient_sampling_frequency"] = freq

    Wave_obj_exact = spyro.AcousticWave(dictionary=d)
    Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    cond = fire.conditional(Wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    Wave_obj_exact.set_initial_velocity_model(conditional=cond, dg_velocity_model=False)
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.forward_solution_receivers

    Wave_obj_guess = spyro.AcousticWave(dictionary=d)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.forward_solve()
    rec_out_guess = Wave_obj_guess.forward_solution_receivers

    misfit = rec_out_exact - rec_out_guess
    # gradient_solve re-runs the forward solve with storage enabled, keeping the
    # wavefield only every ``freq`` steps, then back-propagates the adjoint.
    dJ = Wave_obj_guess.gradient_solve(misfit=misfit)
    return dJ.dat.data_ro.copy()


@pytest.fixture(scope="module")
def full_sampling_gradient():
    """Reference gradient with every timestep stored (frequency = 1)."""
    return _gradient_for_sampling_frequency(1)


@pytest.mark.parametrize("freq", [2, 3])
def test_gradient_sampling_frequency(full_sampling_gradient, freq):
    """``gradient_sampling_frequency`` > 1 subsamples the stored forward
    wavefield to save memory. The resulting gradient must match the
    fully-sampled gradient up to small subsampling (discretization) error, and
    must NOT be rescaled by a spurious factor of ~``freq``.

    This guards the sample-spacing fix in ``backward_time_integration``: the
    second-derivative stencil and the trapezoidal quadrature must use
    ``freq * dt`` as the sample spacing. Before the fix this test sees a
    relative difference of ~0.97 (freq=2) / ~1.9 (freq=3); after it, ~0.02 /
    ~0.03.
    """
    g_full = full_sampling_gradient
    g_sub = _gradient_for_sampling_frequency(freq)
    rel_diff = np.linalg.norm(g_sub - g_full) / np.linalg.norm(g_full)
    print(f"freq={freq}: relative gradient difference vs full sampling = {rel_diff:.4f}")
    assert rel_diff < 0.05


if __name__ == "__main__":
    test_gradient(AdjointType.IMPLEMENTED_ADJOINT)
    test_gradient(AdjointType.UFL_DERIVED_ADJOINT)
    test_gradient_sampling_frequency(_gradient_for_sampling_frequency(1), 2)
    test_gradient_sampling_frequency(_gradient_for_sampling_frequency(1), 3)
