import firedrake as fire

from . import helpers
from .. import utils


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
    """
    if wave.sources is not None:
        wave.sources.current_sources = source_ids
        rhs_forcing = fire.Cofunction(wave.function_space.dual())

    wave.field_logger.start_logging(source_ids)
    wave.comm.comm.barrier()

    t = wave.current_time
    nt = int(wave.final_time / wave.dt) + 1  # number of timesteps
    usol = [
        fire.Function(wave.function_space, name=wave.get_function_name())
        for t in range(nt)
        if t % wave.gradient_sampling_frequency == 0
    ]
    if wave.sources is not None and wave.use_vertex_only_mesh:
        q_s = wave.sources.source_cofunction()
        interpolate_receivers = wave.receivers.receiver_interpolator(
            wave.get_function())
    usol_recv = []
    save_step = 0
    for step in range(nt):
        # Basic way of applying sources
        wave.update_source_expression(t)
        fire.assemble(wave.rhs, tensor=wave.B)

        # More efficient way of applying sources
        if wave.sources is not None:
            B0 = wave.rhs_no_pml()
            if wave.use_vertex_only_mesh:
                B0 += fire.assemble(wave.sources.wavelet[step] * q_s)
            else:
                B0 += wave.sources.apply_source(rhs_forcing, step)
        wave.solver.solve(wave.next_vstate, wave.B)

        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate
        if wave.use_vertex_only_mesh:
            usol_recv.append(fire.assemble(interpolate_receivers))
        else:
            usol_recv.append(wave.get_receivers_output())

        if step % wave.gradient_sampling_frequency == 0:
            usol[save_step].assign(wave.get_function())
            save_step += 1

        if (step - 1) % wave.output_frequency == 0:
            assert (
                fire.norm(wave.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            wave.field_logger.log(t)
            helpers.display_progress(wave.comm, t)

        t = step * float(wave.dt)

    wave.current_time = t
    helpers.display_progress(wave.comm, t)
    usol_recv = helpers.fill(
        usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, wave.comm)

    wave.receivers_output = usol_recv
    wave.forward_solution = usol
    wave.forward_solution_receivers = usol_recv

    wave.field_logger.stop_logging()
    return usol, usol_recv

