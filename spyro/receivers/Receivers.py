"""Receivers class for evaluating efficiently point data."""

from firedrake import *  # noqa: F403
from spyro.receivers.dirac_delta_projector import Delta_projector
from ..domains.space import create_function_space
from ..utils.typing import WaveType
import numpy as np
from ..tools.version_control import is_firedrake_new

if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate


class Receivers(Delta_projector):
    """Eveluate data at points based on Dirac Delta projection.

    Can interpolate receiver values at any point in the domain.
    These points do not need to coincide with points of the mesh or DOFs.

    ...

    Attributes
    ----------
    mesh : Firedrake.mesh
        mesh where receivers are located
    V : Firedrake.FunctionSpace object
        The space of the finite elements
    my_ensemble : Firedrake.ensemble_communicator
        An ensemble communicator
    dimension : int
        The dimension of the space
    degree : int
        Degree of the function space
    receiver_locations : list
        List of tuples containing all receiver locations
    num_receivers : int
        Number of receivers
    quadrilateral : boolean
        Boolean that specifies if cells are quadrilateral
    is_local : list of booleans
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

    def __init__(self, wave):
        """Initialize class and gets all receiver parameters from input file.

        Parameters
        ----------
        wave : :class: 'Wave' object
            Waveform object that contains all simulation parameters

        Returns
        -------
        Receivers : :class: 'Receiver' object
        """
        super().__init__(wave)
        self.point_locations = wave.receiver_locations

        if self.dimension == 3 and wave.automatic_adjoint:
            # self.column_x = model["acquisition"]["num_rec_x_columns"]
            # self.column_y = model["acquisition"]["num_rec_y_columns"]
            # self.column_z = model["acquisition"]["num_rec_z_columns"]
            # self.number_of_points = self.column_x*self.column_y
            raise ValueError("Implement this later")
        else:
            self.number_of_points = wave.number_of_receivers

        self.is_local = [0] * self.number_of_points
        # Set by receiver_interpolator: input-order index of each receiver
        # owned by this spatial rank in the vertex-only mesh.
        self.vom_input_indices = None
        if not self.automatic_adjoint:
            self.build_maps()

    def apply_receivers_as_source(self, rhs_forcing, residual, IT):
        """Inject receivers as a source.

        The adjoint operation of interpolation (injection).

        Injects residual, and timestep IT, at receiver locations
        as source and stores their value in the right hand side
        operator rhs_forcing.

        Parameters
        ----------
        rhs_forcing : object
            Firedrake assembled right hand side operator to store values
        residual : list
            List of residual values at different receiver locations
            and timesteps
        IT : int
            Desired time step number to get residual value from

        Returns
        -------
        rhs_forcing : object
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

    def receiver_interpolator(
        self,
        f,
        reorder=True,
        vom_tolerance=None,
        vom_missing_points_behaviour="error",
        vom_redundant=True,
        vom_name=None,
    ):
        """Return an interpolator object.

        Parameters
        ----------
        f : firedrake.Function
            A function to interpolate at receiver locations.
        reorder : bool, optional
            Flag indicating whether to reorder meshes for better cache
            locality. If not supplied, the default value in
            ``parameters["reorder_meshes"]`` is used.
        vom_missing_points_behaviour : {'warn', 'error', 'ignore'}, optional
            What to do when vertices that are outside of the mesh are
            discarded. If ``'warn'``, a warning is printed. If ``'error'``,
            a :class:`~.VertexOnlyMeshMissingPointsError` is raised. If
            ``'ignore'``, nothing is done. Default is ``'error'``.
        vom_tolerance : float, optional
            The relative tolerance (i.e. as defined on the reference cell) for
            the distance a point can be from a mesh cell and still be
            considered to be in the cell. Note that this tolerance uses an L1
            distance (aka 'manhattan', 'taxicab' or rectilinear distance), so
            it scales with the dimension of the mesh. The default is the
            parent mesh's ``tolerance`` property. Changing this from the
            default causes the parent mesh's spatial index to be rebuilt,
            which can take some time.
        vom_redundant : bool, optional
            If ``True``, the mesh is built using only the vertices specified on
            rank 0. If ``False``, the mesh is built using the vertices
            specified by each rank. Care must be taken when using
            ``redundant=False``; see the note below for more information.
        vom_name : str, optional
            The name of the vertex-only mesh. Default is ``None``.

        Returns
        -------
        firedrake.Interpolate
            An interpolation operator used to interpolate a firedrake
            function at the receiver locations.
        """
        vom = VertexOnlyMesh(
            self.mesh,
            self.point_locations,
            reorder=reorder,
            tolerance=vom_tolerance,
            missing_points_behaviour=vom_missing_points_behaviour,
            redundant=vom_redundant,
            name=vom_name,
        )
        self.vom_input_indices = self._vom_input_indices(vom)
        if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
            V_r = create_function_space(vom, "DG0", 0, dim=self.dimension)
        elif self.wave_type == WaveType.ISOTROPIC_ACOUSTIC:
            V_r = create_function_space(vom, "DG0", 0)
        else:
            raise ValueError("Invalid wave type")
        return interpolate(f, V_r)

    def _vom_input_indices(self, vom):
        """Input-order index of every receiver owned by this spatial rank.

        A vertex-only mesh keeps on each rank only the points inside its mesh
        partition, in an order of its own. ``vom.input_ordering`` is the same
        point cloud in the order the user supplied it, held by the rank that
        supplied it (rank 0 when ``redundant=True``). Interpolating each
        point's input position from there onto ``vom`` gives, for every
        locally owned receiver, its column in the global shot record.

        Parameters
        ----------
        vom : firedrake.VertexOnlyMesh
            Vertex-only mesh built from ``self.point_locations``.

        Returns
        -------
        numpy.ndarray
            Integer array with one entry per locally owned receiver.
        """
        comm = self.mesh.comm
        position = Function(create_function_space(vom.input_ordering, "DG0", 0))
        counts = comm.allgather(position.dat.data_ro.shape[0])
        if sum(counts) != self.number_of_points:
            raise ValueError(
                f"The vertex-only mesh was built from {sum(counts)} points "
                f"but there are {self.number_of_points} receivers. Every "
                "rank supplies the full receiver list, so the mesh must be "
                "built with redundant=True."
            )
        offset = sum(counts[: comm.rank])
        position.dat.data_wo[:] = np.arange(offset, offset + counts[comm.rank])
        local_position = assemble(
            interpolate(position, create_function_space(vom, "DG0", 0))
        )
        return np.rint(local_position.dat.data_ro).astype(int)

    def gather_receiver_record(self, local_record):
        """Assemble the global shot record from the samples of every rank.

        Parameters
        ----------
        local_record : array_like
            ``(nt, n_local)`` (``(nt, n_local, dim)`` for vector fields)
            samples of the receivers owned by this rank, in the order of the
            vertex-only mesh built by :meth:`receiver_interpolator`.

        Returns
        -------
        numpy.ndarray
            ``(nt, number_of_points[, dim])`` record with the receivers in
            the order of the model dictionary, identical on every rank of
            the spatial communicator. Receivers outside the mesh (only
            possible with ``vom_missing_points_behaviour != "error"``) are
            ``nan``.
        """
        if self.vom_input_indices is None:
            raise RuntimeError(
                "receiver_interpolator must be called before gathering the "
                "receiver record."
            )
        comm = self.mesh.comm
        local_record = np.asarray(local_record)
        global_shape = (
            (local_record.shape[0], self.number_of_points)
            + local_record.shape[2:]
        )
        global_record = np.full(global_shape, np.nan)
        for indices, record in zip(
            comm.allgather(self.vom_input_indices),
            comm.allgather(local_record),
        ):
            global_record[:, indices] = record
        return global_record

    def local_receiver_values(self, global_values):
        """Restrict one time step of a global record to this rank's receivers.

        Parameters
        ----------
        global_values : array_like
            ``(number_of_points[, dim])`` values in the order of the model
            dictionary.

        Returns
        -------
        numpy.ndarray
            Values of the receivers owned by this rank, in the order of the
            vertex-only mesh built by :meth:`receiver_interpolator`.
        """
        if self.vom_input_indices is None:
            raise RuntimeError(
                "receiver_interpolator must be called before restricting a "
                "receiver record."
            )
        return np.asarray(global_values)[self.vom_input_indices]

    def new_at(self, udat, receiver_id):
        """Evaluate data at a point."""
        return super().new_at(udat, receiver_id)
