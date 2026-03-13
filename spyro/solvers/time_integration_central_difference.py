import firedrake as fire

from . import helpers
from .. import utils
import numpy as np


def central_difference(wave, source_ids=[0], **kwargs):
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
        None

    Notes:
    ------
    Use ``LinearVariationalSolver`` with per-step source updates through
    ``wave.rhs_no_pml_source()`` before ``wave.solver.solve()``.
    """
    if wave.sources is not None:
        wave.set_active_sources(source_ids)
        rhs_forcing = fire.Cofunction(wave.function_space.dual())

    wave.field_logger.start_logging(source_ids)
    wave.comm.comm.barrier()

    t = wave.current_time
    nt = int(wave.final_time / wave.dt) + 1  # number of timesteps
    compute_functional = wave.compute_functional
    usol = [
        fire.Function(wave.function_space, name=wave.get_function_name())
        for t in range(nt)
        if t % wave.gradient_sampling_frequency == 0
    ]
    if wave.sources is not None and wave.use_vertex_only_mesh:
        # source_cof is a cofunction that represents a point source,
        # being one at a point and zero elsewhere.
        source_cof = wave.update_source_control(source_ids)
        interpolate_receivers = wave.receivers.receiver_interpolator(
            wave.get_function())
    usol_recv = []
    save_step = 0
    wave.functional = None
    true_recv = None
    if compute_functional:
        if not wave.use_vertex_only_mesh:
            raise ValueError(
                "compute_functional=True requires use_vertex_only_mesh=True "
                "so the receiver-space functional can be annotated."
            )
        true_recv = kwargs.get("true_recv", None)
        if not isinstance(true_recv, (list, np.ndarray)):
            raise ValueError(
                "true_recv should be a list or numpy array when "
                "wave.compute_functional is True."
            )
        true_recv_functions = wave.update_true_receiver_data(
            true_recv,
            target_space=interpolate_receivers.target_space,
        )
        Jm = 0.
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
            recv = fire.assemble(interpolate_receivers)
            usol_recv.append(recv)
            if compute_functional:
                rec_out_exact = true_recv_functions[step]
                misfit = rec_out_exact - recv
                Jm += 0.5 * fire.assemble(fire.inner(misfit, misfit) * fire.dx)

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
    if compute_functional:
        wave.functional = Jm

    wave.receivers_data = usol_recv
    usol_recv = helpers.fill(
        usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
    )
    wave.receivers_data = utils.utils.communicate(usol_recv, wave.comm)
    wave.forward_solution = usol

    wave.field_logger.stop_logging()
