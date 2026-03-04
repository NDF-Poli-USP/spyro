import firedrake as fire

from . import helpers
from .. import utils
import numpy as np


def central_difference(
        wave, store_receivers_output, compute_functional, source_ids=[0],
        **kwargs
):
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
        # source_cof is a cofunction that represents a point source,
        # being one at a point and zero elsewhere.
        source_cof = wave.sources.source_cofunction()
        interpolate_receivers = wave.receivers.receiver_interpolator(
            wave.vstate)
    usol_recv = []
    save_step = 0
    if compute_functional:
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
            if store_receivers_output:
                usol_recv.append(recv)
            # check if compute_functional is True
            if compute_functional:
                true_recv = kwargs.get("true_recv", None)
                if isinstance(true_recv, list):
                    rec_out_exact = fire.Function(interpolate_receivers.target_space)
                    if isinstance(true_recv[step], fire.Function):
                        rec_out_exact.dat.data_wo[:] = true_recv[step].dat.data_ro[:]
                    elif isinstance(true_recv[step], np.ndarray):
                        rec_out_exact.dat.data_wo[:] = true_recv[step]
                    else:
                        raise ValueError("Elements of true_recv should be either Firedrake Functions or numpy arrays.")
                else:
                    raise ValueError("true_recv should be a list when compute_functional is True.")
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
    if store_receivers_output:
        wave.receivers_output = usol_recv
        usol_recv = helpers.fill(
            usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
        )
        usol_recv = utils.utils.communicate(usol_recv, wave.comm)

        wave.forward_solution = usol
        wave.forward_solution_receivers = usol_recv

        wave.field_logger.stop_logging()
        return usol, usol_recv
    if compute_functional:
        return Jm
