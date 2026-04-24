import firedrake as fire
import numpy as np

from . import helpers
from .. import utils
from ..utils.typing import FunctionalEvaluationMode


def _propagate_forward_central_difference(wave_obj, source_ids):
    """Advance the forward solve with the central-difference scheme.

    This is an internal helper used by :meth:`Wave.wave_propagator`. It updates
    the solver state in place.

    Parameters
    ----------
    wave_obj: Wave
        The wave solver object containing all necessary information to perform
        the forward solve.
    source_ids: list of int
        List of source IDs to simulate.
    """
    if wave_obj.sources is not None:
        wave_obj.sources.current_sources = source_ids
        rhs_forcing = fire.Cofunction(wave_obj.function_space.dual())

    wave_obj.field_logger.start_logging(source_ids)
    wave_obj.comm.comm.barrier()
    functional_mode = wave_obj.functional_evaluation_mode
    compute_functional = functional_mode is not None
    t = wave_obj.current_time
    nt = int(wave_obj.final_time / wave_obj.dt) + 1  # number of timesteps
    usol = [
        fire.Function(wave_obj.function_space, name=wave_obj.get_function_name())
        for t in range(nt)
        if t % wave_obj.gradient_sampling_frequency == 0
    ]
    source_cof = None
    interpolate_receivers = None
    if wave_obj.sources is not None and wave_obj.use_vertex_only_mesh:
        # source_cof is a cofunction that represents a point source,
        # being one at a point and zero elsewhere.
        source_cof = wave_obj.sources.source_cofunction()
        interpolate_receivers = wave_obj.receivers.receiver_interpolator(
            wave_obj.vstate)
    usol_recv = []
    save_step = 0
    if functional_mode is FunctionalEvaluationMode.PER_TIMESTEP:
        J = 0.0
    for step in range(nt):
        # Basic way of applying sources
        wave_obj.update_source_expression(t)

        if wave_obj.sources is not None:
            if wave_obj.use_vertex_only_mesh:
                wave_obj.rhs_no_pml_source().assign(fire.assemble(
                    wave_obj.sources.wavelet[step] * source_cof))
            else:
                wave_obj.rhs_no_pml_source().assign(
                    wave_obj.sources.apply_source(rhs_forcing, step))
        wave_obj.solver.solve()

        wave_obj.prev_vstate = wave_obj.vstate
        wave_obj.vstate = wave_obj.next_vstate
        if wave_obj.use_vertex_only_mesh:
            usol_recv.append(fire.assemble(interpolate_receivers))
        else:
            usol_recv.append(wave.get_forward_solution_receivers())

        if step % wave_obj.gradient_sampling_frequency == 0:
            usol[save_step].assign(wave_obj.get_function())
            save_step += 1

        if (step - 1) % wave_obj.output_frequency == 0:
            assert (
                fire.norm(wave_obj.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            wave_obj.field_logger.log(t)
            helpers.display_progress(wave_obj.comm, t)

        if functional_mode is FunctionalEvaluationMode.PER_TIMESTEP:
            observed_step = utils.get_real_shot_step(wave_obj, step)
            if wave_obj.use_vertex_only_mesh:
                if isinstance(observed_step, np.ndarray):
                    real_shot = fire.Function(
                        usol_recv[-1].function_space(),
                        val=observed_step,
                    )
                    misfit_step = real_shot - usol_recv[-1]
                elif isinstance(observed_step, fire.Function):
                    misfit_step = observed_step - usol_recv[-1]
                else:
                    raise ValueError(
                        "Unsupported type for real_shot_record. Must be "
                        "either a numpy array or a Firedrake Function."
                    )
            else:
                misfit_step = observed_step - usol_recv[-1]
            J += utils.compute_functional(
                wave_obj, misfit_step, evaluation_mode=FunctionalEvaluationMode.PER_TIMESTEP,
                step=step, nsteps=nt
            )

        t = step * float(wave_obj.dt)

    wave_obj.current_time = t
    helpers.display_progress(wave_obj.comm, t)
    usol_recv = helpers.fill(
        usol_recv, wave_obj.receivers.is_local, nt, wave_obj.receivers.number_of_points
    )
    
    usol_recv = utils.utils.communicate(usol_recv, wave.comm)

    wave_obj.forward_solution = usol
    wave_obj.forward_solution_receivers = usol_recv

    if functional_mode is FunctionalEvaluationMode.AFTER_SOLVE:
        observed_shot = utils.get_real_shot_record(wave_obj)
        misfit = observed_shot - usol_recv
        J = utils.compute_functional(wave_obj, misfit)
    if compute_functional:
        wave_obj.functional_value = J
    else:
        wave_obj.functional_value = None

    wave_obj.field_logger.stop_logging()

