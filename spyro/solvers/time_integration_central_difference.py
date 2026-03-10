import firedrake as fire

from . import helpers
from .. import utils
import numpy as np


def _observed_receivers_at_step(true_receivers, step, target_space):
    if true_receivers is None:
        raise ValueError(
            "wave.true_receivers must be set when compute_functional is True."
        )

    if isinstance(true_receivers, np.ndarray):
        receiver_step = true_receivers[step]
    elif isinstance(true_receivers, (list, tuple)):
        receiver_step = true_receivers[step]
    else:
        raise ValueError(
            "wave.true_receivers should be either a NumPy array or a per-step sequence."
        )

    observed_receivers = fire.Function(target_space)
    if isinstance(receiver_step, fire.Function):
        observed_receivers.dat.data_wo[:] = receiver_step.dat.data_ro[:]
    else:
        try:
            observed_receivers.dat.data_wo[:] = np.asarray(
                receiver_step, dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each entry of wave.true_receivers should be either a Firedrake Function or array-like receiver data."
            ) from exc

    return observed_receivers


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
            wave.get_function())
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
            usol_recv.append(recv)
            # check if compute_functional is True
            if compute_functional:
                rec_out_exact = _observed_receivers_at_step(
                    wave.true_receivers,
                    step,
                    interpolate_receivers.target_space,
                )
                misfit = rec_out_exact - recv
                time_weight = 0.25 if step in (0, nt - 1) else 0.5
                Jm += (
                    time_weight
                    * float(wave.dt)
                    * fire.assemble(fire.inner(misfit, misfit) * fire.dx)
                )

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

    if len(usol_recv) > 0:
        receivers_output = helpers.fill(
            usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
        )
        receivers_output = utils.utils.communicate(receivers_output, wave.comm)
    else:
        receivers_output = None

    wave.forward_solution = usol
    wave.receivers_output = receivers_output
    wave.forward_solution_receivers = receivers_output

    wave.field_logger.stop_logging()

    if compute_functional:
        return Jm
    if store_receivers_output:
        return usol, receivers_output
    return usol
