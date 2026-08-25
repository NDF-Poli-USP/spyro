import firedrake as fire
import numpy as np

from . import helpers
from .. import utils
from ..utils.typing import FunctionalEvaluationMode, AdjointType, AbsorbingBCsType


def _propagate_forward_central_difference(wave, source_ids):
    """Advance the forward solve with the central-difference scheme.

    This is an internal helper used by :meth:`wave.wave_propagator`. It updates
    the solver state in place.

    Parameters
    ----------
    wave: Wave
        The wave solver object containing all necessary information to perform
        the forward solve.
    source_ids: list of int
        List of source IDs to simulate.
    """
    if wave.sources is not None:
        wave.sources.current_sources = source_ids
        rhs_forcing = fire.Cofunction(wave.function_space.dual())

    adjoint_type = wave.adjoint_type

    wave.field_logger.start_logging(source_ids)
    wave.comm.comm.barrier()

    functional_mode = wave.functional_evaluation_mode
    compute_functional = functional_mode is not None

    t = wave.current_time
    nt = int(wave.final_time / wave.dt) + 1  # number of timesteps
    usol = None
    if wave.store_forward_time_steps:
        # Snapshots are appended as the solve advances rather than allocated
        # up front. The final footprint is the same, but nothing is reserved
        # before it is needed: a solve that aborts early (e.g. the numerical
        # instability check below) no longer has to first allocate the whole
        # nt-step wavefield, which for fine meshes/small dt is many GB.
        usol = []
    source_cof = None
    interpolate_receivers = None
    master_source_W = None
    if wave.sources is not None and wave.use_vertex_only_mesh:
        # source_cof is a cofunction that represents a point source,
        # being one at a point and zero elsewhere.
        source_cof = wave.sources.source_cofunction()

        if wave.abc_type == AbsorbingBCsType.PML:
            pressure_expr = fire.split(wave.X_n)[0]
        else:
            pressure_expr = wave.u_n
        interpolate_receivers = wave.receivers.receiver_interpolator(
            pressure_expr)
        if (
            wave.abc_type == AbsorbingBCsType.PML
            and wave.source_function is not None
        ):
            master_source_W = fire.Cofunction(
                wave.source_function.function_space()
            )
            master_source_W.sub(0).assign(source_cof)

    usol_recv = []
    receiver_array = None
    receiver_buffer = None
    # Reused accumulator for the point-source cofunction, so the per-timestep
    # source assembly writes into an existing tensor instead of allocating a
    # fresh Cofunction on every step.
    source_buffer = None
    save_step = 0
    real_shot_record = None
    if compute_functional:
        J = 0.0
        real_shot_record = utils.get_real_shot_record(wave)
        # Reset misfit to None at the start of the solve to avoid
        # using stale misfit values from previous solves.
        wave.misfit = None
        wave.misfit = []

    steps = range(nt)
    if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
        # ``nt`` is only settled here: ``dt`` may have been replaced by
        # ``get_and_set_maximum_dt`` after the adjoint was enabled. The
        # checkpoint schedule is built for exactly this many steps, and the
        # loop is wrapped so the tape records where each time step ends.
        wave.automated_adjoint.start_recording(total_steps=nt)
        steps = wave.automated_adjoint.timestep_iterator(steps)

    for step in steps:
        # Basic way of applying sources
        wave.update_source_expression(t)

        if wave.sources is not None:
            if wave.use_vertex_only_mesh:
                if master_source_W is not None:
                    wave.source_function.assign(
                        wave.sources.wavelet[step] * master_source_W
                    )
                else:
                    if source_buffer is None:
                        source_buffer = fire.assemble(
                            wave.sources.wavelet[step] * source_cof)
                    else:
                        fire.assemble(
                            wave.sources.wavelet[step] * source_cof,
                            tensor=source_buffer)
                    wave.rhs_no_pml_source().assign(source_buffer)
            else:
                wave.rhs_no_pml_source().assign(
                    wave.sources.apply_source(rhs_forcing, step))

        wave.solver.solve()
        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate

        if wave.use_vertex_only_mesh:
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
            usol_recv.append(wave.get_forward_solution_receivers())

        if (
            wave.store_forward_time_steps
            and step % wave.gradient_sampling_frequency == 0
        ):
            snapshot = fire.Function(
                wave.function_space, name=wave.get_function_name()
            )
            snapshot.assign(wave.get_function())
            usol.append(snapshot)
            save_step += 1

        if (step - 1) % wave.output_frequency == 0:
            assert (
                fire.norm(wave.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            wave.field_logger.log(t)
            helpers.display_progress(wave.comm, t)

        if functional_mode is FunctionalEvaluationMode.PER_TIMESTEP:
            if wave.use_vertex_only_mesh:
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
            wave.misfit.append(misfit_step)
            J += utils.compute_functional(
                wave, misfit_step, evaluation_mode=FunctionalEvaluationMode.PER_TIMESTEP,
                step=step, nsteps=nt
            )

        t = step * float(wave.dt)

    wave.current_time = t

    helpers.display_progress(wave.comm, t)
    if receiver_array is not None and functional_mode is not FunctionalEvaluationMode.PER_TIMESTEP:
        usol_recv = receiver_array
    usol_recv = helpers.fill(
        usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
    )

    usol_recv = utils.utils.communicate(usol_recv, wave.comm)

    if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
        wave.automated_adjoint.stop_recording()
        # Will store only the final value of the functional.
        # Note: for the automated adjoint, the solutions are save in the pyadjoint tape,
        # so we don't need to store them here in the wave object.
        wave.forward_solution = wave.vstate
    else:
        # Store the entire forward solution at receiver locations
        # for use in the implemented adjoint.
        wave.forward_solution = usol

    wave.forward_solution_receivers = usol_recv

    if functional_mode is FunctionalEvaluationMode.AFTER_SOLVE:
        wave.misfit = real_shot_record - usol_recv
        J = utils.compute_functional(wave, wave.misfit)

    if compute_functional:
        wave.functional_value = J
    else:
        wave.functional_value = None

    wave.field_logger.stop_logging()
