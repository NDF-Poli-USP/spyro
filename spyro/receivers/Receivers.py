"""Receivers class for evaluating efficiently point data."""

from functools import cached_property

from firedrake import assemble, Cofunction, VertexOnlyMesh
from firedrake.functionspaceimpl import WithGeometry
from spyro.receivers.dirac_delta_projector import Delta_projector
from ..domains.space import create_function_space
from ..utils.typing import WaveType
import numpy as np
from ..tools.version_control import is_firedrake_new

if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate
else:
    from firedrake import interpolate


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
    receiver_source_injector(target_space)
        Builds the vertex-only-mesh counterpart of
        apply_receivers_as_source, for usage with adjoint propagation
    receiver_interpolator(f)
        Builds the vertex-only-mesh interpolation of a field at the
        receiver locations
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
            # ``is_local`` holds the cell id of each receiver, ``None`` when
            # the receiver is not on this rank; cell 0 is a valid cell.
            if self.is_local[rid] is not None:
                idx = np.int_(self.cellNodeMaps[rid])
                phis = self.cell_tabulations[rid]

                # ``value`` is a scalar for a scalar field and a vector for a
                # vector one, so the outer product scales the tabulated basis
                # values by every component of the receiver value.
                tmp = np.multiply.outer(phis, value)
                rhs_forcing.dat.data_with_halos[idx] += tmp
            else:
                tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing

    @cached_property
    def vertex_only_mesh(self) -> VertexOnlyMesh:
        """Vertex-only mesh at the receiver locations.

        Built once per instance, with the same options as
        :meth:`receiver_interpolator` uses by default, so the receiver
        ordering it defines is the one the forward solve records the
        receiver data in.

        Returns
        -------
        firedrake.VertexOnlyMesh
            Mesh whose vertices are the receiver locations.
        """
        return VertexOnlyMesh(
            self.mesh,
            self.point_locations,
            reorder=True,
            missing_points_behaviour="error",
            redundant=True,
        )

    @cached_property
    def receiver_function_space(self) -> WithGeometry:
        """Space of one time step of receiver data on :attr:`vertex_only_mesh`.

        Returns
        -------
        firedrake.functionspaceimpl.WithGeometry
            Piecewise-constant space on the vertex-only mesh, scalar for
            acoustic waves and vector-valued for elastic ones.
        """
        return self._receiver_function_space(self.vertex_only_mesh)

    def _receiver_function_space(self, receiver_mesh) -> WithGeometry:
        """Return the receiver-data space of this wave type on ``receiver_mesh``.

        Parameters
        ----------
        receiver_mesh : firedrake.VertexOnlyMesh
            Vertex-only mesh at the receiver locations.

        Returns
        -------
        firedrake.functionspaceimpl.WithGeometry
            Piecewise-constant space on ``receiver_mesh``, scalar for
            acoustic waves and vector-valued for elastic ones.

        Raises
        ------
        ValueError
            If the wave type has no receiver-data space.
        """
        if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
            return create_function_space(
                receiver_mesh, "DG0", 0, dim=self.dimension,
            )
        elif self.wave_type == WaveType.ISOTROPIC_ACOUSTIC:
            return create_function_space(receiver_mesh, "DG0", 0)
        else:
            raise ValueError("Invalid wave type")

    def receiver_source_injector(
        self, target_space: WithGeometry,
    ) -> "ReceiverSourceInjector":
        """Return the adjoint of the receiver interpolation into ``target_space``.

        This is the vertex-only-mesh counterpart of
        :meth:`apply_receivers_as_source`: it injects one time step of
        receiver data as a source in the dual of ``target_space``. The
        operator is assembled once, so the backward time loop only writes
        the receiver values of each step into it.

        Parameters
        ----------
        target_space : firedrake.functionspaceimpl.WithGeometry
            Space the receivers read from, and whose dual receives the
            injected values.

        Returns
        -------
        ReceiverSourceInjector
            Callable mapping one time step of receiver data to a cofunction
            on ``target_space``.
        """
        return ReceiverSourceInjector(self.receiver_function_space, target_space)

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
        V_r = self._receiver_function_space(vom)
        return interpolate(f, V_r)

    def new_at(self, udat, receiver_id):
        """Evaluate data at a point."""
        return super().new_at(udat, receiver_id)


class ReceiverSourceInjector:
    """Adjoint of the receiver interpolation, assembled once.

    The forward solve reads the receivers by interpolating the solution onto
    a vertex-only mesh. The adjoint solve needs the transpose of that map: a
    time step of receiver values injected as a source in the dual of the
    space the receivers read from. The adjoint interpolation is built once
    here and re-assembled into the same cofunction on every call.

    Parameters
    ----------
    receiver_space : firedrake.functionspaceimpl.WithGeometry
        Space of one time step of receiver data, on the vertex-only mesh.
    target_space : firedrake.functionspaceimpl.WithGeometry
        Space the receivers read from.

    Attributes
    ----------
    receiver_values : firedrake.Cofunction
        Receiver values of the step being injected, in the dual of the
        receiver space.
    source : firedrake.Cofunction
        Injected source, in the dual of the target space. Reused between
        calls.
    adjoint_interpolation : firedrake.Interpolate
        Symbolic adjoint interpolation of ``receiver_values`` into
        ``source``.
    """

    def __init__(self, receiver_space: WithGeometry, target_space: WithGeometry):
        self.receiver_values = Cofunction(receiver_space.dual())
        self.source = Cofunction(target_space.dual())
        (coargument,) = self.source.arguments()
        self.adjoint_interpolation = interpolate(coargument, self.receiver_values)

    def __call__(self, values) -> Cofunction:
        """Inject one time step of receiver data.

        Parameters
        ----------
        values : firedrake.Function or array_like
            Receiver values of the step, as the ``Function`` on the receiver
            space the forward solve produced while accumulating the
            functional, or as an array in the order of the vertex-only mesh
            (the order the forward solve records receiver data in), of shape
            ``(n_receivers,)`` for a scalar field or
            ``(n_receivers, dimension)`` for a vector one.

        Returns
        -------
        firedrake.Cofunction
            The injected source. It is the same cofunction on every call,
            overwritten each time.

        Raises
        ------
        TypeError
            If ``values`` is neither a Firedrake ``Function`` nor array-like.
        ValueError
            If ``values`` does not hold one value per receiver.
        """
        try:
            data = values.dat.data_ro
        except AttributeError as exc:
            if not isinstance(values, (np.ndarray, list, tuple)):
                raise TypeError(
                    "Receiver values must be a Firedrake Function or "
                    f"array-like receiver data, received {type(values).__name__}.",
                ) from exc
            data = np.asarray(values, dtype=float)
        receiver_values = self.receiver_values.dat.data
        if data.shape != receiver_values.shape:
            raise ValueError(
                "Receiver values must hold one value per receiver, of shape "
                f"{receiver_values.shape}; received shape {data.shape}.",
            )
        receiver_values[:] = data
        return assemble(self.adjoint_interpolation, tensor=self.source)
