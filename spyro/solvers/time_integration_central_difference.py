import firedrake as fire
import numpy as np

from . import helpers
from .. import utils
from ..utils.typing import AdjointType


def _get_real_shot_step(wave, step):
    real_shot_record = wave.real_shot_record

    if (
        wave.current_sources is not None
        and isinstance(real_shot_record, np.ndarray)
    ):
        if real_shot_record.ndim == 3:
            return real_shot_record[wave.current_sources[0], step]
        if real_shot_record.ndim == 2:
            return real_shot_record[step]

    if (
        wave.current_sources is not None
        and isinstance(real_shot_record, (list, tuple))
    ):
        current_source = wave.current_sources[0]
        if len(real_shot_record) > current_source:
            source_record = real_shot_record[current_source]
            if (
                isinstance(source_record, np.ndarray)
                and source_record.ndim == 2
            ):
                return source_record[step]

    return real_shot_record[step]


def central_difference(wave, source_ids=None):
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
    if source_ids is None:
        source_ids = [0]

    store_forward_time_steps = wave.store_forward_time_steps
    compute_functional = wave._compute_functional
    store_misfit = wave._store_misfit
    adjoint_type = wave.adjoint_type
    # Interpator if using vertex-only mesh with sources and receivers.
    # Will be None otherwise.
    interpolate_receivers = None

    if wave.sources is not None:
        wave.sources.current_sources = source_ids

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
    if wave.use_vertex_only_mesh:
        # source_cof is a cofunction that represents a point source,
        wave.source_cofunction.assign(wave.sources.source_cofunction())
        # being one at a point and zero elsewhere.
        interpolate_receivers = wave.receivers.receiver_interpolator(
            wave.vstate)
    usol_recv = []
    save_step = 0
    if compute_functional:
        J = 0.0
    if store_misfit:
        wave.misfit = []
    for step in range(nt):
        if wave.sources is not None:
            if wave.use_vertex_only_mesh:
                wave.rhs_no_pml_source().assign(fire.assemble(
                    wave.sources.wavelet[step] * wave.source_cofunction))
            else:
                # Basic way of applying sources
                wave.update_source_expression(t)
                wave.rhs_no_pml_source().assign(
                    wave.sources.apply_source(wave.source_cofunction, step))
        wave.solver.solve()

        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate
        if wave.use_vertex_only_mesh:
            usol_recv.append(fire.assemble(interpolate_receivers))
        else:
            usol_recv.append(wave.get_receivers_output())

        if (
            store_forward_time_steps
            and step % wave.gradient_sampling_frequency == 0
        ):
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
                residual_step = observed_step - usol_recv[-1]
            J += utils.compute_functional(
                wave, residual_step, per_step=True, step=step, nsteps=nt)
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
