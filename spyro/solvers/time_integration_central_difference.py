import firedrake as fire
import numpy as np

from . import helpers
from .. import utils
from ..utils.typing import FunctionalEvaluationMode, AdjointType


def advance_central_difference_state(wave_obj):
    """Advance central-difference state registers by one solver step.

    The ``vstate`` accessors hide the concrete state layout: acoustic PML
    advances ``X_nm1 <- X_n <- X_np1``, while non-PML advances
    ``u_nm1 <- u_n <- u_np1``.
    """
    wave_obj.prev_vstate = wave_obj.vstate
    wave_obj.vstate = wave_obj.next_vstate


def solve_central_difference_step(wave_obj, solver=None):
    """Solve one central-difference step and advance state registers."""
    if solver is None:
        solver = wave_obj.solver
    solver.solve()
    advance_central_difference_state(wave_obj)


def _forward_state_storage_space(wave_obj):
    """Return the space needed by the implemented-adjoint forward replay."""
    if (
        wave_obj.abc_boundary_layer_type == "PML"
        and wave_obj.forward_residual_form is not None
    ):
        return wave_obj.mixed_function_space
    return wave_obj.function_space


def _store_forward_state(wave_obj, target):
    """Store the forward state needed by the implemented adjoint."""
    if (
        wave_obj.abc_boundary_layer_type == "PML"
        and target.function_space() == wave_obj.mixed_function_space
    ):
        target.assign(wave_obj.vstate)
    else:
        target.assign(wave_obj.get_function())


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

    adjoint_type = wave_obj.adjoint_type

    wave_obj.field_logger.start_logging(source_ids)
    wave_obj.comm.comm.barrier()

    functional_mode = wave_obj.functional_evaluation_mode
    compute_functional = functional_mode is not None

    t = wave_obj.current_time
    nt = int(wave_obj.final_time / wave_obj.dt) + 1  # number of timesteps
    usol = None
    if wave_obj.store_forward_time_steps:
        state_space = _forward_state_storage_space(wave_obj)
        usol = [
            fire.Function(state_space, name=wave_obj.get_function_name())
            for t in range(nt)
            if t % wave_obj.gradient_sampling_frequency == 0
        ]
    source_cof = None
    interpolate_receivers = None
    master_source_W = None
    if wave_obj.sources is not None and wave_obj.use_vertex_only_mesh:
        # source_cof is a cofunction that represents a point source,
        # being one at a point and zero elsewhere.
        source_cof = wave_obj.sources.source_cofunction()

        if wave_obj.abc_boundary_layer_type == "PML":
            pressure_expr = fire.split(wave_obj.X_n)[0]
        else:
            pressure_expr = wave_obj.u_n
        interpolate_receivers = wave_obj.receivers.receiver_interpolator(
            pressure_expr)
        if (
            wave_obj.abc_boundary_layer_type == "PML"
            and wave_obj.source_function is not None
        ):
            master_source_W = fire.Cofunction(
                wave_obj.source_function.function_space()
            )
            master_source_W.sub(0).assign(source_cof)

    usol_recv = []
    receiver_array = None
    receiver_buffer = None
    save_step = 0
    real_shot_record = None
    if compute_functional:
        J = 0.0
        real_shot_record = utils.get_real_shot_record(wave_obj)
        # Reset misfit to None at the start of the solve to avoid
        # using stale misfit values from previous solves.
        wave_obj.misfit = None
        wave_obj.misfit = []

    if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
        wave_obj.automated_adjoint.start_recording()

    for step in range(nt):
        # Basic way of applying sources
        wave_obj.update_source_expression(t)

        if wave_obj.sources is not None:
            if wave_obj.use_vertex_only_mesh:
                if master_source_W is not None:
                    wave_obj.source_function.assign(
                        wave_obj.sources.wavelet[step] * master_source_W
                    )
                else:
                    wave_obj.rhs_no_pml_source().assign(fire.assemble(
                        wave_obj.sources.wavelet[step] * source_cof))
            else:
                wave_obj.rhs_no_pml_source().assign(
                    wave_obj.sources.apply_source(rhs_forcing, step))

        solve_central_difference_step(wave_obj)

        if wave_obj.use_vertex_only_mesh:
            if receiver_buffer is None:
                receiver_buffer = fire.assemble(interpolate_receivers)
                receiver_shape = receiver_buffer.dat.data_ro.shape
                receiver_array = np.empty((nt,) + receiver_shape, dtype=float)
            else:
                fire.assemble(interpolate_receivers, tensor=receiver_buffer)
            receiver_array[step] = receiver_buffer.dat.data_ro
            if functional_mode is FunctionalEvaluationMode.PER_TIMESTEP:
                usol_recv.append(receiver_buffer.copy(deepcopy=True))
        else:
            usol_recv.append(wave_obj.get_forward_solution_receivers())

        if (
            wave_obj.store_forward_time_steps
            and step % wave_obj.gradient_sampling_frequency == 0
        ):
            _store_forward_state(wave_obj, usol[save_step])
            save_step += 1

        if (step - 1) % wave_obj.output_frequency == 0:
            assert (
                fire.norm(wave_obj.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            wave_obj.field_logger.log(t)
            helpers.display_progress(wave_obj.comm, t)

        if functional_mode is FunctionalEvaluationMode.PER_TIMESTEP:
            if wave_obj.use_vertex_only_mesh:
                if isinstance(real_shot_record[step], np.ndarray):
                    real_shot = fire.Function(
                        usol_recv[-1].function_space(),
                        val=real_shot_record[step],
                    )
                    misfit_step = real_shot - usol_recv[-1]
                elif isinstance(real_shot_record[step], fire.Function):
                    misfit_step = real_shot_record[step] - usol_recv[-1]
                else:
                    raise ValueError(
                        "Unsupported type for real_shot_record. Must be "
                        "either a numpy array or a Firedrake Function."
                    )
            else:
                misfit_step = real_shot_record[step] - usol_recv[-1]
            wave_obj.misfit.append(misfit_step)
            J += utils.compute_functional(
                wave_obj, misfit_step, evaluation_mode=FunctionalEvaluationMode.PER_TIMESTEP,
                step=step, nsteps=nt
            )

        t = step * float(wave_obj.dt)

    wave_obj.current_time = t

    helpers.display_progress(wave_obj.comm, t)
    if receiver_array is not None and functional_mode is not FunctionalEvaluationMode.PER_TIMESTEP:
        usol_recv = receiver_array
    usol_recv = helpers.fill(
        usol_recv, wave_obj.receivers.is_local, nt, wave_obj.receivers.number_of_points
    )

    usol_recv = utils.utils.communicate(usol_recv, wave_obj.comm)

    if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
        wave_obj.automated_adjoint.stop_recording()
        # Will store only the final value of the functional.
        # Note: for the automated adjoint, the solutions are save in the pyadjoint tape,
        # so we don't need to store them here in the wave object.
        wave_obj.forward_solution = wave_obj.vstate
    else:
        # Store the entire forward solution at receiver locations
        # for use in the implemented adjoint.
        wave_obj.forward_solution = usol

    wave_obj.forward_solution_receivers = usol_recv

    if functional_mode is FunctionalEvaluationMode.AFTER_SOLVE:
        wave_obj.misfit = real_shot_record - usol_recv
        J = utils.compute_functional(wave_obj, wave_obj.misfit)

    if compute_functional:
        wave_obj.functional_value = J
    else:
        wave_obj.functional_value = None

    wave_obj.field_logger.stop_logging()
