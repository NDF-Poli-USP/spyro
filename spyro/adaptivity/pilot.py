"""Two-level pilot: trace-error estimate and adjoint error indicator.

The pilot of a mesh :math:`M_h` solves the forward problem on :math:`M_h` and
on its uniform refinement :math:`M_f` (each triangle split in four), with the
same element and the same time step. The relative difference of the receiver
traces, restricted to a time window, estimates the discretization error of
:math:`M_h`.

For the adjoint strategy the refined solve is recorded with spyro's
automated adjoint (pyadjoint), with the functional

.. math::

    J(u) = \\frac{1}{2} \\sum_n w_n\\, \\Delta t \\sum_r m_{nr}
           \\left(d_{nr}(u) - d_{nr}(P u_h)\\right)^2,

where :math:`d(Pu_h)` are the traces of the prolonged coarse solution sampled
on :math:`M_f`, :math:`m` is the window and :math:`w_n` the trapezoidal
weights. The reverse sweep of the tape
(:meth:`~spyro.solvers.automatic_differentiation_solver.AutomatedAdjoint.compute_adjoint`,
which skips the velocity derivative) gives, on each time-step solve, the
discrete adjoint :math:`\\lambda^n` of the refined scheme. The
dual-weighted residual of the prolonged coarse trajectory is then localized
on the refined cells :math:`K`,

.. math::

    \\eta_K = \\sum_n R^n_f(P u_h)(\\lambda^n \\psi_K),

with :math:`R^n_f` the residual of the refined explicit step (spyro's own
variational form, without the point source) and :math:`\\psi_K` the
indicator function of :math:`K`. Because the scheme is linear and :math:`J`
quadratic, the identity

.. math::

    \\sum_K \\eta_K - \\sum_n f^n(\\lambda^n) = \\pm 2 J(u_f)

holds to round-off (:math:`f^n` is the point source), and is reported as a
check. No adjoint equation is written by hand. With a free surface the
Dirichlet nodes carry a zero adjoint, so they add nothing to :math:`\\eta_K`.

The indicator is only needed to build a new mesh. With
``skip_indicator_below`` the adjoint of a shot is skipped when its trace error
is already below the acceptance threshold, and recomputed only if another shot
ends above it.
"""

from __future__ import annotations

import copy
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import firedrake as fire
import ufl
from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock
from pyadjoint import get_working_tape

from ..solvers.acoustic_wave import AcousticWave

#: ``window(times, shot)`` returns a boolean array ``(n_times, n_receivers)``.
Window = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class PilotResult:
    """Outcome of a two-level pilot.

    Parameters
    ----------
    mesh_file : str
        Coarse mesh of the pilot.
    dofs_coarse : int
        Degrees of freedom of the coarse space.
    dofs_fine : int
        Degrees of freedom of the refined space.
    steps : int
        Number of time steps of each solve.
    dt : float
        Common time step (s).
    trace_errors : dict of int to float
        Relative windowed trace error of each shot, coarse versus refined.
    seconds : dict of int to list of float
        Wall time of the refined forward, the coarse forward and the adjoint
        (including the indicator) of each shot.
    centroids : numpy.ndarray or None
        Refined-cell centroids ``(z, x)``, gathered on rank 0 when an
        indicator was computed.
    indicators : numpy.ndarray or None
        Cell indicator of each shot, ``(n_shots, n_cells)``, on rank 0.
    identity_ratios : dict of int to float
        :math:`(\\sum_K \\eta_K - \\sum_n f^n(\\lambda^n)) / 2J` of each shot
        with an adjoint (:math:`\\pm 1` up to round-off when accumulating every
        step).
    source_terms : dict of int to float
        :math:`\\sum_n f^n(\\lambda^n)` of each shot with an adjoint.
    adjoint_shots : list of int
        Shots whose adjoint and indicator were computed.
    repeated_shots : list of int
        Shots propagated a second time, to compute an adjoint first skipped
        (see :meth:`TwoLevelPilot.run`).
    """

    mesh_file: str
    dofs_coarse: int
    dofs_fine: int
    steps: int
    dt: float
    trace_errors: dict[int, float] = field(default_factory=dict)
    seconds: dict[int, list[float]] = field(default_factory=dict)
    centroids: np.ndarray | None = None
    indicators: np.ndarray | None = None
    identity_ratios: dict[int, float] = field(default_factory=dict)
    source_terms: dict[int, float] = field(default_factory=dict)
    adjoint_shots: list[int] = field(default_factory=list)
    repeated_shots: list[int] = field(default_factory=list)

    @property
    def worst_trace_error(self) -> float:
        """Largest trace error over the shots.

        Returns
        -------
        float
            :math:`\\max_s \\eta_s`.
        """
        return max(self.trace_errors.values())


def nodal_velocity(function_space: fire.FunctionSpace,
                   velocity: Callable[[np.ndarray, np.ndarray], np.ndarray]
                   ) -> fire.Function:
    """Sample a velocity callable at the nodes of a function space.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Scalar Lagrange-type space (e.g. KMV).
    velocity : callable
        ``velocity(z, x)`` returning the velocity at the points.

    Returns
    -------
    firedrake.Function
        Velocity in ``function_space``.
    """
    mesh = function_space.mesh()
    coordinates = fire.Function(
        fire.VectorFunctionSpace(mesh, function_space.ufl_element())
    ).interpolate(mesh.coordinates)
    c = fire.Function(function_space, name="velocity")
    points = coordinates.dat.data_ro_with_halos
    c.dat.data_with_halos[:] = velocity(points[:, 0], points[:, 1])
    return c


class _ReceiverSampler:
    """Sample a field at the receivers, in input order, on every rank.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Space of the sampled field.
    receivers : numpy.ndarray
        Receiver coordinates ``(z, x)``.
    """

    def __init__(self, function_space: fire.FunctionSpace,
                 receivers: np.ndarray) -> None:
        self.comm = function_space.mesh().comm
        vom = fire.VertexOnlyMesh(function_space.mesh(), np.asarray(receivers))
        self.points = fire.Function(fire.FunctionSpace(vom, "DG", 0))
        self.ordered = fire.Function(fire.FunctionSpace(vom.input_ordering, "DG", 0))

    def __call__(self, u: fire.Function) -> np.ndarray:
        """Return the values of ``u`` at the receivers.

        Parameters
        ----------
        u : firedrake.Function
            Field to sample.

        Returns
        -------
        numpy.ndarray
            Values in input order, broadcast to all ranks.
        """
        self.points.interpolate(u)
        self.ordered.interpolate(self.points)
        values = self.ordered.dat.data_ro.copy() if self.comm.rank == 0 else None
        return self.comm.bcast(values, root=0)


class TwoLevelPilot:
    """Run two-level pilots on meshes, with spyro's acoustic solver.

    Each shot is propagated by its own :class:`~spyro.AcousticWave`, built from
    ``dictionary`` with that single source, so every shot has its own tape and
    its own adjoint.

    Parameters
    ----------
    dictionary : dict
        spyro model dictionary describing options, acquisition (all sources),
        time axis and boundary conditions. Its mesh entries are replaced by
        the pilot's meshes and its time step by the common pilot time step.
    velocity : callable
        ``velocity(z, x)``, sampled at the nodes of both spaces.
    window : callable, optional
        Time-receiver window of the trace error and of the adjoint functional,
        ``window(times, shot)``. ``None`` uses the full traces.
    dt_fraction : float, optional
        Fraction of the critical time step of the refined mesh. Default ``0.5``.
    accumulate_every : int, optional
        Accumulate the indicator every this many steps, weighting each sample
        accordingly. ``1`` (the default) is exact; larger values are cheaper
        and approximate.
    checkpointing : bool, optional
        Manage the refined tape with a checkpoint schedule (see
        :meth:`spyro.solvers.wave.Wave.enable_automated_adjoint`). Default
        ``False`` keeps every step on the tape.
    snapshots : int, optional
        Checkpoints kept in memory when ``checkpointing`` is on; ``None``
        keeps every step and never recomputes. Default ``None``.

    Raises
    ------
    NotImplementedError
        If the dictionary selects a PML, whose residual is not localized here.
    """

    def __init__(self, dictionary: dict,
                 velocity: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 window: Window | None = None, dt_fraction: float = 0.5,
                 accumulate_every: int = 1, checkpointing: bool = False,
                 snapshots: int | None = None) -> None:
        abc = dictionary.get("absorving_boundary_conditions", {})
        if abc.get("status", False) and str(abc.get("abc_type")).lower() == "pml":
            raise NotImplementedError("The adjoint indicator does not support PML.")
        if accumulate_every < 1:
            raise ValueError("accumulate_every must be a positive integer.")
        self.dictionary = copy.deepcopy(dictionary)
        self.velocity = velocity
        self.window = window
        self.dt_fraction = dt_fraction
        self.accumulate_every = accumulate_every
        self.checkpointing = checkpointing
        self.snapshots = snapshots
        acquisition = self.dictionary["acquisition"]
        self.sources = [tuple(s) for s in acquisition["source_locations"]]
        self.receivers = np.asarray(acquisition["receiver_locations"], dtype=float)
        self.final_time = float(self.dictionary["time_axis"]["final_time"])

    @property
    def number_of_shots(self) -> int:
        """Number of sources of the acquisition.

        Returns
        -------
        int
            Number of shots.
        """
        return len(self.sources)

    def _wave(self, shot: int, mesh: fire.MeshGeometry | None) -> AcousticWave:
        """Build the wave of one shot, optionally on a mesh.

        Parameters
        ----------
        shot : int
            Source index.
        mesh : firedrake.MeshGeometry or None
            Mesh to use; ``None`` leaves the wave without a mesh.

        Returns
        -------
        AcousticWave
            The wave, with its velocity set when a mesh is given.
        """
        dictionary = copy.deepcopy(self.dictionary)
        dictionary["parallelism"] = {"type": "spatial"}
        dictionary["acquisition"]["source_locations"] = [self.sources[shot]]
        dictionary["acquisition"]["use_vertex_only_mesh"] = True
        dictionary["mesh"] = {**dictionary.get("mesh", {}),
                              "mesh_type": "user_mesh", "mesh_file": None}
        dictionary["time_axis"]["gradient_sampling_frequency"] = 1
        dictionary["time_axis"]["output_frequency"] = 10**9
        wave = AcousticWave(dictionary=dictionary)
        if mesh is not None:
            wave.set_mesh(user_mesh=mesh)
            wave.set_initial_velocity_model(
                velocity_model_function=nodal_velocity(wave.function_space,
                                                       self.velocity))
        return wave

    def _mask(self, shot: int, steps: int, dt: float) -> np.ndarray:
        """Window of one shot on the record time axis.

        Parameters
        ----------
        shot : int
            Source index.
        steps : int
            Number of recorded steps.
        dt : float
            Time step.

        Returns
        -------
        numpy.ndarray
            Boolean array ``(steps, n_receivers)``.
        """
        if self.window is None:
            return np.ones((steps, len(self.receivers)), dtype=bool)
        # spyro records the state after each step: sample n is at (n + 1) dt.
        return np.asarray(self.window((np.arange(steps) + 1) * dt, shot), dtype=bool)

    def run(self, mesh_file: str, compute_indicator: bool = False,
            skip_indicator_below: float | None = None) -> PilotResult:
        """Run the pilot of a mesh.

        Parameters
        ----------
        mesh_file : str
            Coarse mesh (gmsh file in spyro's ``(z, x)`` convention).
        compute_indicator : bool, optional
            Also compute the adjoint indicator. Default ``False``.
        skip_indicator_below : float, optional
            Trace error at or below which the indicator is not needed, because
            the mesh is then accepted. A shot whose error is at most this value
            skips its adjoint; if the worst shot ends above it, the skipped
            shots are propagated again with the adjoint, so the indicator is
            either complete or absent. ``None`` (the default) never skips.

        Returns
        -------
        PilotResult
            Trace errors, costs and, when requested and needed, the indicator.
        """
        # Every spyro wave installs a new ensemble and drops the previous one,
        # so the meshes live on an ensemble held here for the whole run.
        ensemble = self._wave(0, None).comm
        comm = ensemble.comm
        hierarchy = fire.MeshHierarchy(fire.Mesh(str(mesh_file), comm=comm), 1)
        coarse_mesh, fine_mesh = hierarchy[0], hierarchy[1]
        result = self._new_result(str(mesh_file), coarse_mesh, fine_mesh)

        cells = {}
        for shot in range(self.number_of_shots):
            cells[shot] = self._run_shot(shot, coarse_mesh, fine_mesh, result,
                                         compute_indicator, skip_indicator_below)
        if not compute_indicator:
            return result
        skipped = [shot for shot, values in cells.items() if values is None]
        if skipped and result.worst_trace_error <= skip_indicator_below:
            # The mesh is accepted: no indicator is needed.
            return result
        for shot in skipped:
            result.repeated_shots.append(shot)
            cells[shot] = self._run_shot(shot, coarse_mesh, fine_mesh, result,
                                         True, None)

        space = fire.VectorFunctionSpace(fine_mesh, "DG", 0)
        centroids = fire.Function(space).interpolate(
            fire.SpatialCoordinate(fine_mesh)).dat.data_ro.copy()
        local = np.array([cells[shot] for shot in range(self.number_of_shots)])
        gathered = comm.gather((centroids, local), root=0)
        if comm.rank == 0:
            result.centroids = np.concatenate([g[0] for g in gathered])
            result.indicators = np.concatenate([g[1] for g in gathered], axis=1)
        return result

    def _new_result(self, mesh_file: str, coarse_mesh: fire.MeshGeometry,
                    fine_mesh: fire.MeshGeometry) -> PilotResult:
        """Common time step and space sizes of a pilot, before any shot runs.

        Parameters
        ----------
        mesh_file : str
            Coarse mesh file.
        coarse_mesh : firedrake.MeshGeometry
            Coarse mesh.
        fine_mesh : firedrake.MeshGeometry
            Its uniform refinement.

        Returns
        -------
        PilotResult
            Empty result whose time step is ``dt_fraction`` of the critical
            step of the refined mesh.
        """
        coarse = self._wave(0, coarse_mesh)
        fine = self._wave(0, fine_mesh)
        # spyro's estimate of the stability limit, rounded down so that the
        # final time is a whole number of steps.
        dt = fine.get_and_set_maximum_dt(fraction=self.dt_fraction)
        result = PilotResult(mesh_file=mesh_file,
                             dofs_coarse=coarse.function_space.dim(),
                             dofs_fine=fine.function_space.dim(),
                             steps=int(self.final_time / dt) + 1, dt=dt)
        del coarse, fine
        gc.collect()
        return result

    def _run_shot(self, shot: int, coarse_mesh: fire.MeshGeometry,
                  fine_mesh: fire.MeshGeometry, result: PilotResult,
                  compute_indicator: bool, skip_below: float | None
                  ) -> np.ndarray | None:
        """Propagate one shot on both levels and, if asked, its adjoint.

        Parameters
        ----------
        shot : int
            Source index.
        coarse_mesh : firedrake.MeshGeometry
            Coarse mesh.
        fine_mesh : firedrake.MeshGeometry
            Its uniform refinement.
        result : PilotResult
            Receives the trace error, the wall times (added to those of an
            earlier run of the same shot) and the adjoint bookkeeping.
        compute_indicator : bool
            Compute the adjoint indicator of the shot.
        skip_below : float or None
            Skip the adjoint when the trace error of the shot is at most this.

        Returns
        -------
        numpy.ndarray or None
            Indicator of the locally owned refined cells, or ``None`` when no
            adjoint was computed.
        """
        coarse = self._wave(shot, coarse_mesh)
        fine = self._wave(shot, fine_mesh)
        coarse.dt = result.dt
        fine.dt = result.dt
        coarse.store_forward_time_steps = compute_indicator
        mask = self._mask(shot, result.steps, result.dt)

        t0 = time.perf_counter()
        coarse.forward_solve()
        t1 = time.perf_counter()
        cell_values, t_adjoint = None, 0.0
        if compute_indicator:
            cell_values, t_adjoint = self._refined_with_adjoint(
                shot, coarse, fine, mask, result, skip_below)
        else:
            fine.forward_solve()
        t2 = time.perf_counter()

        result.trace_errors[shot] = self._trace_error(coarse, fine, mask)
        previous = result.seconds.get(shot, [0.0, 0.0, 0.0])
        result.seconds[shot] = [previous[0] + t2 - t1 - t_adjoint,
                                previous[1] + t1 - t0, previous[2] + t_adjoint]
        if cell_values is not None:
            result.adjoint_shots.append(shot)
        if fine.automated_adjoint is not None:
            fine.automated_adjoint.clear_tape()
        del coarse, fine
        gc.collect()
        return cell_values

    @staticmethod
    def _trace_error(coarse: AcousticWave, fine: AcousticWave,
                     mask: np.ndarray) -> float:
        """Relative windowed trace error of the coarse level.

        Parameters
        ----------
        coarse : AcousticWave
            Coarse wave, already solved.
        fine : AcousticWave
            Refined wave, already solved.
        mask : numpy.ndarray
            Window of the shot, ``(steps, n_receivers)``.

        Returns
        -------
        float
            :math:`\\|d_H - d_h\\|_W / \\|d_h\\|_W`.
        """
        d_coarse = coarse.forward_solution_receivers
        d_fine = fine.forward_solution_receivers
        difference = np.where(mask, d_coarse - d_fine, 0.0)
        reference = np.where(mask, d_fine, 0.0)
        return float(np.linalg.norm(difference) / np.linalg.norm(reference))

    def _refined_with_adjoint(self, shot: int, coarse: AcousticWave,
                              fine: AcousticWave, mask: np.ndarray,
                              result: PilotResult, skip_below: float | None = None
                              ) -> tuple[np.ndarray | None, float]:
        """Refined forward on the tape, its adjoint and the cell indicator.

        The indicator is accumulated while pyadjoint runs the reverse sweep,
        from a callback on each time-step solve, so the adjoint of a step is
        used as soon as it exists and no adjoint trajectory is kept.

        Parameters
        ----------
        shot : int
            Source index.
        coarse : AcousticWave
            Coarse wave, already solved with its states stored.
        fine : AcousticWave
            Refined wave, not yet solved.
        mask : numpy.ndarray
            Window of the shot, ``(steps, n_receivers)``.
        result : PilotResult
            Receives the identity check and the source term.
        skip_below : float, optional
            When the trace error of the shot is at most this, the reverse
            sweep is skipped. ``None`` (the default) never skips.

        Returns
        -------
        tuple
            Indicator of the locally owned refined cells (``None`` when the
            sweep was skipped) and the wall time of the reverse sweep (s).

        Raises
        ------
        RuntimeError
            If the tape does not hold one solve per time step, or the reverse
            sweep did not visit all of them.
        """
        V_f = fine.function_space
        states = coarse.forward_solution
        steps = len(states)
        w_np1, w_n, w_nm1 = (fire.Function(V_f) for _ in range(3))

        def prolong(index: int, target: fire.Function) -> None:
            """Prolong a stored coarse state onto the refined space.

            Parameters
            ----------
            index : int
                Time step of the coarse state; negative steps are the zero
                initial condition.
            target : firedrake.Function
                Refined function that receives the state.

            Returns
            -------
            None
                ``target`` is overwritten.
            """
            if index < 0:
                target.zero()
            else:
                fire.prolong(states[index], target)

        sampler = _ReceiverSampler(V_f, self.receivers)
        record = np.zeros((steps, len(self.receivers)))
        for n in range(steps):
            prolong(n, w_np1)
            record[n] = sampler(w_np1)

        fine.real_shot_record = record
        fine.misfit_window = mask.astype(float)
        fine.enable_automated_adjoint(checkpointing=self.checkpointing,
                                      snapshots=self.snapshots)
        fine.forward_solve()
        if skip_below is not None and self._trace_error(coarse, fine, mask) <= skip_below:
            # The mesh may be accepted: the adjoint is deferred (see :meth:`run`).
            return None, 0.0
        t_start = time.perf_counter()

        blocks = [b for b in get_working_tape().get_blocks()
                  if isinstance(b, GenericSolveBlock)]
        if len(blocks) != steps:
            raise RuntimeError(f"Expected {steps} time-step solves on the tape, "
                               f"found {len(blocks)}.")

        Q = fire.FunctionSpace(fine.mesh, "DG", 0)
        psi = fire.TestFunction(Q)
        lam = fire.Function(V_f)
        test, _ = sorted(fine.lhs.arguments(), key=lambda a: a.number())
        residual = fire.action(fine.lhs, w_np1) - fine.rhs
        residual = ufl.replace(residual, {fine.u_n: w_n, fine.u_nm1: w_nm1,
                                          test: lam * psi})
        buffer = fire.assemble(residual)
        source = fine.sources.source_cofunction()
        wavelet = fine.sources.wavelet
        comm = V_f.mesh().comm
        every = self.accumulate_every
        eta = np.zeros(buffer.dat.data_ro.shape[0])
        sweep = {"next": steps - 1, "source": 0.0}

        def accumulate(n: int, adjoint: fire.Function) -> None:
            """Add the contribution of one time step to the indicator.

            Called by pyadjoint right after it solves the adjoint of step
            ``n``, in reverse order of the steps.

            Parameters
            ----------
            n : int
                Time step.
            adjoint : firedrake.Function
                Adjoint of the step.

            Returns
            -------
            None
                The indicator and the source term are updated in place.

            Raises
            ------
            RuntimeError
                If a step is visited out of order.
            """
            if n != sweep["next"]:
                raise RuntimeError("The reverse sweep skipped a time step.")
            sweep["next"] = n - 1
            if wavelet[n] != 0.0:
                sweep["source"] += wavelet[n] * comm.allreduce(
                    float(np.dot(source.dat.data_ro, adjoint.dat.data_ro)))
            if every == 1:
                # Going backwards, consecutive steps share two states.
                if n == steps - 1:
                    prolong(n, w_np1)
                    prolong(n - 1, w_n)
                else:
                    w_np1.assign(w_n)
                    w_n.assign(w_nm1)
                prolong(n - 2, w_nm1)
            elif n % every == 0:
                prolong(n, w_np1)
                prolong(n - 1, w_n)
                prolong(n - 2, w_nm1)
            if n % every == 0:
                lam.assign(adjoint)
                fire.assemble(residual, tensor=buffer)
                eta[:] += every * buffer.dat.data_ro
            # The block of this step still reads its adjoint after this
            # callback, but the block of the next step is done with its own:
            # drop it, so that the adjoints of all steps are not kept. When
            # the blocks share one adjoint function (as the blocks of one
            # solver may), there is nothing to drop, and dropping it would
            # take this step's adjoint away from its own block.
            if n + 1 < steps and blocks[n + 1].adj_state is not blocks[n].adj_state:
                blocks[n + 1].adj_state = None

        for n, block in enumerate(blocks):
            block.adj_cb = lambda adjoint, n=n: accumulate(n, adjoint)
        functional = fine.functional_value
        # Only the adjoint states are needed: no velocity derivative.
        fine.automated_adjoint.compute_adjoint(functional)
        if sweep["next"] != -1:
            raise RuntimeError("The reverse sweep did not reach the first step.")
        if steps > 1 and blocks[0].adj_state is not blocks[1].adj_state:
            blocks[0].adj_state = None

        two_j = 2.0 * float(functional)
        total = comm.allreduce(float(eta.sum())) - sweep["source"]
        result.identity_ratios[shot] = total / two_j if two_j else float("nan")
        result.source_terms[shot] = sweep["source"]
        return eta, time.perf_counter() - t_start
