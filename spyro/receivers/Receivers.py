from firedrake import *  # noqa: F403
from spyro.receivers.dirac_delta_projector import Delta_projector
from ..utils.typing import WaveType
import numpy as np
from ..tools.version_control import is_firedrake_new


if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate


class Receivers(Delta_projector):
    """Project data defined on a triangular mesh to a
    set of 2D/3D coordinates for variable spatial order
    using Lagrange interpolation.

    Can interpolate receiveir values that do not coincide with
    mesh or DOF points

    ...

    Attributes
    ----------
    mesh : Firedrake.mesh
        mesh where receivers are located
    V: Firedrake.FunctionSpace object
        The space of the finite elements
    my_ensemble: Firedrake.ensemble_communicator
        An ensemble communicator
    dimension: int
        The dimension of the space
    degree: int
        Degree of the function space
    receiver_locations: list
        List of tuples containing all receiver locations
    num_receivers: int
        Number of receivers
    quadrilateral: boolean
        Boolean that specifies if cells are quadrilateral
    is_local: list of booleans
        List that checks if receivers are present in cores
        spatial paralelism

    Methods
    -------
    build_maps()
        Calculates and stores tabulations for interpolation
    interpolate(field)
        Interpolates field value at receiver locations
    apply_receivers_as_source(rhs_forcing, residual, IT)
        Applies receivers as source with values from residual
        in timestep IT, for usage with adjoint propagation
    """

    def __init__(self, wave_object, **kwargs):
        """Initializes class and gets all receiver parameters from
        input file.
        Parameters
        ----------
        wave_object: :class: 'Wave' object
            Waveform object that contains all simulation parameters
        Returns
        -------
        Receivers: :class: 'Receiver' object
        """
        super().__init__(wave_object)
        self.point_locations = wave_object.receiver_locations
        if wave_object.use_vertex_only_mesh:
            reorder = kwargs.get("reorder", True)
            vom_tolerance = kwargs.get("vom_tolerance", 1e-10)
            vom_missing_points_behaviour = kwargs.get("vom_missing_points_behaviour", "error")
            vom_redundant = kwargs.get("vom_redundant", False)
            vom_name = kwargs.get("vom_name", "receivers_vom")
            self.vom = VertexOnlyMesh(
                wave_object.mesh,
                self.point_locations,
                reorder=reorder,
                tolerance=vom_tolerance,
                missing_points_behaviour=vom_missing_points_behaviour,
                redundant=vom_redundant,
                name=vom_name
            )

        if self.dimension == 3 and wave_object.automatic_adjoint:
            # self.column_x = model["acquisition"]["num_rec_x_columns"]
            # self.column_y = model["acquisition"]["num_rec_y_columns"]
            # self.column_z = model["acquisition"]["num_rec_z_columns"]
            # self.number_of_points = self.column_x*self.column_y
            raise ValueError("Implement this later")
        else:
            self.number_of_points = wave_object.number_of_receivers

        self.is_local = [0] * self.number_of_points
        if not self.automatic_adjoint:
            self.build_maps()

    def apply_receivers_as_source(self, rhs_forcing, residual, IT):
        """The adjoint operation of interpolation (injection)

        Injects residual, and timestep IT, at receiver locations
        as source and stores their value in the right hand side
        operator rhs_forcing.

        Parameters
        ----------
        rhs_forcing: object
            Firedrake assembled right hand side operator to store values
        residual: list
            List of residual values at different receiver locations
            and timesteps
        IT: int
            Desired time step number to get residual value from

        Returns
        -------
        rhs_forcing: object
            Firedrake assembled right hand side operator with injected values
        """
        for rid in range(self.number_of_points):
            value = residual[IT][rid]
            if self.is_local[rid]:
                idx = np.int_(self.cellNodeMaps[rid])
                phis = self.cell_tabulations[rid]

                tmp = np.dot(phis, value)
                rhs_forcing.dat.data_with_halos[idx] += tmp
            else:
                tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing

    def receiver_interpolator(self, f):
        """Return an interpolator object.

        Parameters
        ----------
        f : firedrake.Function
            A function to interpolate at receiver locations.

        Returns
        -------
        firedrake.Interpolate
            An interpolation operator used to interpolate a firedrake
            function at the receiver locations.
        """
        if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
            V_r = VectorFunctionSpace(self.vom, "DG", 0)
        elif self.wave_type == WaveType.ISOTROPIC_ACOUSTIC:
            V_r = FunctionSpace(self.vom, "DG", 0)
        else:
            raise ValueError("Invalid wave type")
        return interpolate(f, V_r)

    def new_at(self, udat, receiver_id):
        return super().new_at(udat, receiver_id)

    def global_receiver_values_from_vom(receiver_functions, ensemble, receiver_mesh):
    """Gather local VOM receiver values in global receiver order.

    Parameters
    ----------
    receiver_functions : list of firedrake.Function
        Receiver values sampled on the distributed VOM, one function per time
        step. Each function only stores the receiver points owned by the local
        spatial MPI rank.
    ensemble : firedrake.Ensemble
        Spyro ensemble communicator.
    receiver_mesh : firedrake.Mesh
        The VOM on which the receiver functions are defined.

    Returns
    -------
    numpy.ndarray
        Shot record with shape ``(n_timesteps, n_receivers)``. The receiver
        axis follows the original order supplied in the model dictionary.

    Notes
    -----
    The sampled receiver function lives on the distributed VOM. Its local data
    width is therefore the number of receiver points owned by the rank, not the
    global receiver count. This is why the older ``fill`` and ``MPI.MAX``
    reduction path is not valid for VOM data.

    The intermediate ``input_ordered_function`` is a Firedrake function on the
    VOM's ``input_ordering`` mesh. Firedrake builds this mesh so that receiver
    points are laid out in the same order as the user input. Interpolating the
    local VOM function onto this mesh gives each rank its local contiguous slice
    of the global shot-record order. Concatenating those slices across the
    spatial communicator produces the global record expected by the rest of
    Spyro.
    """
    if not receiver_functions:
        return np.asarray(receiver_functions)
    receiver_function_space = receiver_functions[0].function_space()
    input_space = FunctionSpace(
        receiver_mesh.input_ordering,
        receiver_function_space.ufl_element())
    receiver_values = []
    for receiver_function in receiver_functions:
        input_ordered_function = assemble(
            interpolate(receiver_function, input_space)
        )
        local_values = np.asarray(input_ordered_function.dat.data_ro)
        local_values = np.array(local_values, copy=True)
        all_values = ensemble.comm.allgather(local_values)
        receiver_values.append(np.concatenate(all_values, axis=0))

    return np.asarray(receiver_values)


def global_receiver_step_to_vom(observed_step, receiver_function_space, receiver_mesh):
    """"Project one global receiver record onto the local VOM space.

    Parameters
    ----------
    observed_step : array_like
        Receiver data for one time step in Spyro's public shot-record layout:
        one value per receiver, ordered as in the model dictionary.
    receiver_function_space : firedrake.FunctionSpace
        Function space of the local VOM receiver values for the same time step.
    receiver_mesh : firedrake.Mesh
        The VOM on which the receiver functions are defined.

    Returns
    -------
    firedrake.Function
        ``observed_step`` represented on ``receiver_function_space``. On each
        MPI rank this function contains only the receiver values owned by that
        rank's local VOM partition.

    Notes
    -----
    This is the inverse operation of
    :func:`_global_receiver_values_from_vom` for a single time step. The global
    array is first copied into the local slice of the VOM ``input_ordering``
    function. Firedrake then interpolates from ``input_ordering`` back to the
    distributed VOM, giving a function that can be subtracted from the local
    simulated receiver function when computing per-timestep misfit values.
    """
    # input_space = _input_ordering_function_space(receiver_function_space)
    input_space = FunctionSpace(
        receiver_mesh.input_ordering,
        receiver_function_space.ufl_element())
    observed_input_ordering = Function(input_space)
    observed_step = np.asarray(observed_step)

    local_count = observed_input_ordering.dat.data_wo.shape[0]
    all_counts = input_space.comm.allgather(local_count)
    offset = sum(all_counts[:input_space.comm.rank])
    observed_input_ordering.dat.data_wo[:] = observed_step[offset:offset + local_count]

    return assemble(
        interpolate(observed_input_ordering, receiver_function_space)
    )


