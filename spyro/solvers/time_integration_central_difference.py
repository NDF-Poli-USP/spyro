import firedrake as fire
import numpy as np

from . import helpers
from .. import utils
from ..utils.typing import FunctionalEvaluationMode, AdjointType


def _receiver_function_from_array(wave_obj, receiver_space, receiver_values):
    """Build a receiver-space Function from global or local receiver values."""
    real_shot = fire.Function(receiver_space)
    receiver_values = np.asarray(receiver_values)
    local_shape = real_shot.dat.data.shape

    if receiver_values.shape == local_shape:
        real_shot.dat.data[:] = receiver_values
        return real_shot

    if receiver_values.size == real_shot.dat.data.size:
        real_shot.dat.data[:] = receiver_values.reshape(local_shape)
        return real_shot

    if receiver_values.shape[0] != wave_obj.receivers.number_of_points:
        raise ValueError(
            "Observed receiver data has incompatible shape "
            f"{receiver_values.shape}; expected either local shape "
            f"{local_shape} or global receiver count "
            f"{wave_obj.receivers.number_of_points}."
        )

    receiver_locations = np.asarray(wave_obj.receivers.point_locations)
    local_coordinates = receiver_space.mesh().coordinates.dat.data_ro
    local_receiver_ids = []
    for coordinate in local_coordinates:
        distances = np.linalg.norm(receiver_locations - coordinate, axis=1)
        receiver_id = int(np.argmin(distances))
        if distances[receiver_id] > 1.0e-10:
            raise ValueError(
                "Could not map local VertexOnlyMesh coordinate "
                f"{coordinate} to a receiver location."
            )
        local_receiver_ids.append(receiver_id)

    local_values = np.ascontiguousarray(receiver_values[local_receiver_ids])
    if local_values.shape != local_shape:
        raise ValueError(
            "Localized observed receiver data has incompatible shape "
            f"{local_values.shape}; expected {local_shape}."
        )

    real_shot.dat.data[:] = local_values
    return real_shot


def _receiver_functions_to_array(wave_obj, receiver_functions):
    """Convert local receiver Functions to a global receiver array."""
    if len(receiver_functions) == 0:
        return np.asarray(receiver_functions)

    receiver_space = receiver_functions[0].function_space()
    receiver_locations = np.asarray(wave_obj.receivers.point_locations)
    local_coordinates = receiver_space.mesh().coordinates.dat.data_ro
    local_receiver_ids = []
    for coordinate in local_coordinates:
        distances = np.linalg.norm(receiver_locations - coordinate, axis=1)
        receiver_id = int(np.argmin(distances))
        if distances[receiver_id] > 1.0e-10:
            raise ValueError(
                "Could not map local VertexOnlyMesh coordinate "
                f"{coordinate} to a receiver location."
            )
        local_receiver_ids.append(receiver_id)

    local_shape = receiver_functions[0].dat.data_ro.shape
    value_shape = local_shape[1:]
    receiver_shape = (len(receiver_functions), wave_obj.receivers.number_of_points)
    receiver_shape += value_shape
    receiver_array = np.full(receiver_shape, -99999.0)
    for step, receiver_function in enumerate(receiver_functions):
        local_values = receiver_function.dat.data_ro
        if local_values.shape != local_shape:
            raise ValueError(
                "Inconsistent local receiver Function shape "
                f"{local_values.shape}; expected {local_shape}."
            )
        receiver_array[step, local_receiver_ids] = local_values

    return utils.utils.communicate(receiver_array, wave_obj.comm)


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
        usol = [
            fire.Function(wave_obj.function_space, name=wave_obj.get_function_name())
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

        wave_obj.solver.solve()
        wave_obj.prev_vstate = wave_obj.vstate
        wave_obj.vstate = wave_obj.next_vstate

        if wave_obj.use_vertex_only_mesh:
            usol_recv.append(fire.assemble(interpolate_receivers))
        else:
            usol_recv.append(wave_obj.get_forward_solution_receivers())

        if (
            wave_obj.store_forward_time_steps
            and step % wave_obj.gradient_sampling_frequency == 0
        ):
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
            if wave_obj.use_vertex_only_mesh:
                if isinstance(real_shot_record[step], np.ndarray):
                    real_shot = _receiver_function_from_array(
                        wave_obj,
                        usol_recv[-1].function_space(),
                        real_shot_record[step],
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
    if adjoint_type == AdjointType.AUTOMATED_ADJOINT and wave_obj.use_vertex_only_mesh:
        usol_recv = _receiver_functions_to_array(wave_obj, usol_recv)
    else:
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
