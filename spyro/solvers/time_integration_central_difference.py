import firedrake as fire
import numpy as np

from . import helpers
from .. import utils
from ..utils.typing import AdjointType


def central_difference(wave, source_ids=[0]):
    """
    Perform central difference time integration for wave propagation.

    Parameters:
    -----------
    wave: Spyro object
        The Wave object containing the necessary data and parameters.
    source_ids: list of ints (optional)
        The ID of the sources being propagated. Defaults to [0].

    Returns:
    --------
        tuple:
            A tuple containing the forward solution and the receiver output.

    Notes:
    ------
    Use ``LinearVariationalSolver`` with per-step source updates through
    ``wave.rhs_no_pml_source()`` before ``wave.solver.solve()``.
    """
    store_forward_time_steps = getattr(wave, "store_forward_time_steps", True)
    compute_functional = getattr(wave, "_compute_functional", False)
    store_misfit = getattr(wave, "_store_misfit", False)
    adjoint_type = getattr(wave, "adjoint_type", AdjointType.NONE)

    if wave.sources is not None:
        wave.sources.current_sources = source_ids
        rhs_forcing = fire.Cofunction(wave.function_space.dual())

    wave.field_logger.start_logging(source_ids)
    wave.comm.comm.barrier()

    t = wave.current_time
    nt = int(wave.final_time / wave.dt) + 1  # number of timesteps
    usol = []
    if store_forward_time_steps:
        usol = [
            fire.Function(wave.function_space, name=wave.get_function_name())
            for t in range(nt)
            if t % wave.gradient_sampling_frequency == 0
        ]
    if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
        wave.automated_adjoint.start_recording()
    if wave.sources is not None and wave.use_vertex_only_mesh:
        # source_cof is a cofunction that represents a point source,
        # being one at a point and zero elsewhere.
        source_cof = wave.sources.source_cofunction()
        interpolate_receivers = wave.receivers.receiver_interpolator(
            wave.vstate)
    usol_recv = []
    save_step = 0
    if compute_functional:
        J = 0.0
    if store_misfit:
        wave.misfit = []
    for step in range(nt):
        # Basic way of applying sources
        wave.update_source_expression(t)

        if wave.sources is not None:
            if wave.use_vertex_only_mesh:
                wave.rhs_no_pml_source().assign(fire.assemble(
                    wave.sources.wavelet[step] * source_cof))
            else:
                wave.rhs_no_pml_source().assign(
                    wave.sources.apply_source(rhs_forcing, step))
        wave.solver.solve()

        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate
        if wave.use_vertex_only_mesh:
            usol_recv.append(fire.assemble(interpolate_receivers))
        else:
            usol_recv.append(wave.get_receivers_output())

        if store_forward_time_steps and step % wave.gradient_sampling_frequency == 0:
            usol[save_step].assign(wave.get_function())
            save_step += 1

        if (step - 1) % wave.output_frequency == 0:
            assert (
                fire.norm(wave.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            wave.field_logger.log(t)
            helpers.display_progress(wave.comm, t)

        if compute_functional:
            if wave.use_vertex_only_mesh:
                if isinstance(wave.real_shot_record[step], np.ndarray):
                    real_shot = fire.Function(
                        usol_recv[-1].function_space(),
                        val=wave.real_shot_record[step],
                    )
                    residual_step = real_shot - usol_recv[-1]
                elif isinstance(wave.real_shot_record[step], fire.Function):
                    residual_step = wave.real_shot_record[step] - usol_recv[-1]
                else:
                    raise ValueError(
                        "Unsupported type for real_shot_record. "
                        "Must be either a numpy array or a Firedrake Function."
                    )
            else:
                residual_step = wave.real_shot_record[step] - usol_recv[-1]

            weight = 0.5 if step == 0 or step == nt - 1 else 1.0
            if wave.use_vertex_only_mesh:
                J += fire.assemble(
                    0.5 * weight * fire.inner(residual_step, residual_step) * fire.dx
                ) * wave.dt
            else:
                J += 0.5 * weight * np.sum(residual_step ** 2) * wave.dt
            if store_misfit:
                wave.misfit.append(residual_step)

        t = step * float(wave.dt)

    if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
        wave.automated_adjoint.stop_recording()
        wave.forward_solution = wave.vstate
    else:
        wave.forward_solution = usol

    wave.current_time = t
    helpers.display_progress(wave.comm, t)
    usol_recv = helpers.fill(
        usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, wave.comm)

    wave.receivers_output = usol_recv
    
    if compute_functional:
        wave.functional_value = J

    wave.field_logger.stop_logging()
