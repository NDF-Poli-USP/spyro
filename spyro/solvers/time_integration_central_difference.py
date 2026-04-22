import firedrake as fire
import numpy as np

from . import helpers
from .. import utils
from ..utils.typing import FunctionalEvaluationMode


def _propagate_forward_with_central_difference(solver, shot_ids=None):
    """Advance the forward solve with the central-difference scheme.

    This is an internal helper used by :meth:`Wave.wave_propagator`. It updates
    the solver state in place instead of returning the forward solution and
    receiver data directly.

    Parameters
    ----------
    solver: Wave
        The wave solver object containing all necessary information to perform
        the forward solve.
    shot_ids: list of int, optional
        List of shot IDs to simulate. If None, defaults to [0].
    """
    if shot_ids is None:
        shot_ids = [0]
    if solver.sources is not None:
        solver.sources.current_sources = shot_ids
        rhs_forcing = fire.Cofunction(solver.function_space.dual())

    solver.field_logger.start_logging(shot_ids)
    solver.comm.comm.barrier()
    functional_mode = solver.functional_evaluation_mode
    compute_functional = functional_mode is not None
    t = solver.current_time
    nt = int(solver.final_time / solver.dt) + 1  # number of timesteps
    usol = [
        fire.Function(solver.function_space, name=solver.get_function_name())
        for t in range(nt)
        if t % solver.gradient_sampling_frequency == 0
    ]
    source_cof = None
    interpolate_receivers = None
    if solver.sources is not None and solver.use_vertex_only_mesh:
        # source_cof is a cofunction that represents a point source,
        # being one at a point and zero elsewhere.
        source_cof = solver.sources.source_cofunction()
        interpolate_receivers = solver.receivers.receiver_interpolator(
            solver.vstate)
    usol_recv = []
    save_step = 0
    if functional_mode is FunctionalEvaluationMode.PER_TIMESTEP:
        J = 0.0
    for step in range(nt):
        # Basic way of applying sources
        solver.update_source_expression(t)

        if solver.sources is not None:
            if solver.use_vertex_only_mesh:
                solver.rhs_no_pml_source().assign(fire.assemble(
                    solver.sources.wavelet[step] * source_cof))
            else:
                solver.rhs_no_pml_source().assign(
                    solver.sources.apply_source(rhs_forcing, step))
        solver.solver.solve()

        solver.prev_vstate = solver.vstate
        solver.vstate = solver.next_vstate
        if solver.use_vertex_only_mesh:
            usol_recv.append(fire.assemble(interpolate_receivers))
        else:
            usol_recv.append(solver.get_receivers_output())

        if step % solver.gradient_sampling_frequency == 0:
            usol[save_step].assign(solver.get_function())
            save_step += 1

        if (step - 1) % solver.output_frequency == 0:
            assert (
                fire.norm(solver.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            solver.field_logger.log(t)
            helpers.display_progress(solver.comm, t)

        if functional_mode is FunctionalEvaluationMode.PER_TIMESTEP:
            observed_step = solver.sources.get_real_shot_step(solver, step)
            if solver.use_vertex_only_mesh:
                if isinstance(observed_step, np.ndarray):
                    real_shot = fire.Function(
                        usol_recv[-1].function_space(),
                        val=observed_step,
                    )
                    residual_step = real_shot - usol_recv[-1]
                elif isinstance(observed_step, fire.Function):
                    residual_step = observed_step - usol_recv[-1]
                else:
                    raise ValueError(
                        "Unsupported type for real_shot_record. Must be "
                        "either a numpy array or a Firedrake Function."
                    )
            else:
                residual_step = observed_step - usol_recv[-1]
            J += utils.compute_functional(
                solver, residual_step, per_step=True, step=step, nsteps=nt
            )

        t = step * float(solver.dt)

    solver.current_time = t
    helpers.display_progress(solver.comm, t)
    usol_recv = helpers.fill(
        usol_recv, solver.receivers.is_local, nt, solver.receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, solver.comm)

    solver.receivers_output = usol_recv
    solver.forward_solution = usol
    solver.forward_solution_receivers = usol_recv
    if functional_mode is FunctionalEvaluationMode.AFTER_SOLVE:
        observed_shot = solver.sources.get_real_shot_record(solver)
        residual = observed_shot - usol_recv
        J = utils.compute_functional(solver, residual)
    if compute_functional:
        solver.functional_value = J
    else:
        solver.functional_value = None

    solver.field_logger.stop_logging()
