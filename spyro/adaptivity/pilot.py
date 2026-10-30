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
weights. Differentiating :math:`J` gives, on each time-step solve of the
tape, the discrete adjoint :math:`\\lambda^n` of the refined scheme. The
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
check. No adjoint equation is written by hand.
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
from pyadjoint import continue_annotation, get_working_tape, pause_annotation

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
        (:math:`\\pm 1` up to round-off when accumulating every step).
    source_terms : dict of int to float
        :math:`\\sum_n f^n(\\lambda^n)` of each shot.
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


def critical_time_step(wave: AcousticWave, iterations: int = 80) -> float:
    """Stability limit of the explicit scheme, by power iteration.

    Estimates :math:`2/\\sqrt{\\lambda_{\\max}(M^{-1}K)}` with the lumped mass
    and the stiffness of the wave's quadrature rule.

    Parameters
    ----------
    wave : AcousticWave
        Wave whose mesh, space and initial velocity are set.
    iterations : int, optional
        Number of power iterations. Default ``80``.

    Returns
    -------
    float
        Critical time step (s).
    """
    V = wave.function_space
    c = wave.initial_velocity_model
    v = fire.TestFunction(V)
    u = fire.TrialFunction(V)
    measure = fire.dx(**wave.quadrature_rule)
    mass = fire.assemble((1.0 / (c * c)) * v * measure).dat.data_ro.copy()
    stiffness = fire.inner(fire.grad(u), fire.grad(v)) * measure
    vector = fire.Function(V)
    comm = V.mesh().comm
    rng = np.random.default_rng(1234 + comm.rank)
    vector.dat.data[:] = rng.standard_normal(vector.dat.data.shape)
    eigenvalue = 0.0
    for _ in range(iterations):
        product = fire.assemble(fire.action(stiffness, vector)).dat.data_ro
        current = vector.dat.data_ro
        eigenvalue = (comm.allreduce(float(np.dot(current, product)))
                      / comm.allreduce(float(np.dot(current, mass * current))))
        nxt = product / mass
        vector.dat.data[:] = nxt / np.sqrt(comm.allreduce(float(np.dot(nxt, nxt))))
    return 2.0 / np.sqrt(eigenvalue)


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

    Raises
    ------
    NotImplementedError
        If the dictionary selects a PML, whose residual is not localized here.
    """

    def __init__(self, dictionary: dict,
                 velocity: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 window: Window | None = None, dt_fraction: float = 0.5,
                 accumulate_every: int = 1) -> None:
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

    def run(self, mesh_file: str, compute_indicator: bool = False) -> PilotResult:
        """Run the pilot of a mesh.

        Parameters
        ----------
        mesh_file : str
            Coarse mesh (gmsh file in spyro's ``(z, x)`` convention).
        compute_indicator : bool, optional
            Also compute the adjoint indicator. Default ``False``.

        Returns
        -------
        PilotResult
            Trace errors, costs and, if requested, the indicator.
        """
        # Every spyro wave installs a new ensemble and drops the previous one,
        # so the meshes live on an ensemble held here for the whole run.
        ensemble = self._wave(0, None).comm
        comm = ensemble.comm
        hierarchy = fire.MeshHierarchy(fire.Mesh(str(mesh_file), comm=comm), 1)
        coarse_mesh, fine_mesh = hierarchy[0], hierarchy[1]

        result = None
        cells = []
        for shot in range(self.number_of_shots):
            coarse = self._wave(shot, coarse_mesh)
            fine = self._wave(shot, fine_mesh)
            if result is None:
                dt = self.dt_fraction * critical_time_step(fine)
                result = PilotResult(
                    mesh_file=str(mesh_file),
                    dofs_coarse=coarse.function_space.dim(),
                    dofs_fine=fine.function_space.dim(),
                    steps=int(self.final_time / dt) + 1, dt=dt)
            coarse.dt = result.dt
            fine.dt = result.dt
            coarse.store_forward_time_steps = compute_indicator
            mask = self._mask(shot, result.steps, result.dt)

            t0 = time.perf_counter()
            coarse.forward_solve()
            t1 = time.perf_counter()
            if compute_indicator:
                cell_values = self._refined_with_adjoint(shot, coarse, fine, mask, result)
                cells.append(cell_values)
            else:
                fine.forward_solve()
            t2 = time.perf_counter()

            d_coarse = coarse.forward_solution_receivers
            d_fine = fine.forward_solution_receivers
            difference = np.where(mask, d_coarse - d_fine, 0.0)
            reference = np.where(mask, d_fine, 0.0)
            result.trace_errors[shot] = float(np.linalg.norm(difference)
                                              / np.linalg.norm(reference))
            t_adjoint = result.seconds.pop(shot, [0.0])[0]
            result.seconds[shot] = [t2 - t1 - t_adjoint, t1 - t0, t_adjoint]
            if fine.automated_adjoint is not None:
                fine.automated_adjoint.clear_tape()
            del coarse, fine
            gc.collect()

        if compute_indicator:
            space = fire.VectorFunctionSpace(fine_mesh, "DG", 0)
            centroids = fire.Function(space).interpolate(
                fire.SpatialCoordinate(fine_mesh)).dat.data_ro.copy()
            gathered = comm.gather((centroids, np.array(cells)), root=0)
            if comm.rank == 0:
                result.centroids = np.concatenate([g[0] for g in gathered])
                result.indicators = np.concatenate([g[1] for g in gathered], axis=1)
        return result

    def _refined_with_adjoint(self, shot: int, coarse: AcousticWave,
                              fine: AcousticWave, mask: np.ndarray,
                              result: PilotResult) -> np.ndarray:
        """Refined forward on the tape, its adjoint and the cell indicator.

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
            Receives the identity check, the source term and the adjoint time
            (as the provisional ``seconds[shot]``).

        Returns
        -------
        numpy.ndarray
            Indicator of the locally owned refined cells.

        Raises
        ------
        RuntimeError
            If the tape does not hold one solve per time step.
        """
        V_f = fine.function_space
        states = coarse.forward_solution
        steps = len(states)
        prolonged = [fire.Function(V_f) for _ in range(3)]

        def prolong(index: int, target: fire.Function) -> None:
            if index < 0:
                target.zero()
            else:
                fire.prolong(states[index], target)

        sampler = _ReceiverSampler(V_f, self.receivers)
        record = np.zeros((steps, len(self.receivers)))
        for n in range(steps):
            prolong(n, prolonged[0])
            record[n] = sampler(prolonged[0])

        fine.real_shot_record = record
        fine.enable_automated_adjoint()
        fine.forward_solve()
        t_start = time.perf_counter()

        # The misfits are UFL expressions on the receivers' vertex-only mesh.
        misfit_space = fire.FunctionSpace(
            ufl.domain.extract_unique_domain(fine.misfit[0]), "DG", 0)
        local_indices = fine.receivers.vom_input_indices
        continue_annotation()
        functional = 0.0
        for n, misfit in enumerate(fine.misfit):
            weight = 0.5 if n in (0, steps - 1) else 1.0
            window = fire.Function(misfit_space)
            window.dat.data_wo[:] = mask[n][local_indices]
            functional += fire.assemble(0.5 * result.dt * weight * window
                                        * misfit * misfit * fire.dx)
        pause_annotation()
        fine.automated_adjoint.create_reduced_functional(functional)
        fine.automated_adjoint.compute_derivative()

        blocks = [b for b in get_working_tape().get_blocks()
                  if isinstance(b, GenericSolveBlock)]
        if len(blocks) != steps:
            raise RuntimeError(f"Expected {steps} time-step solves on the tape, "
                               f"found {len(blocks)}.")

        Q = fire.FunctionSpace(fine.mesh, "DG", 0)
        psi = fire.TestFunction(Q)
        lam = fire.Function(V_f)
        w_np1, w_n, w_nm1 = prolonged
        test, _ = sorted(fine.lhs.arguments(), key=lambda a: a.number())
        residual = fire.action(fine.lhs, w_np1) - fine.rhs
        residual = ufl.replace(residual, {fine.u_n: w_n, fine.u_nm1: w_nm1,
                                          test: lam * psi})
        buffer = fire.assemble(residual)
        source = fine.sources.source_cofunction()
        wavelet = fine.sources.wavelet
        comm = V_f.mesh().comm

        eta = np.zeros(buffer.dat.data_ro.shape[0])
        source_sum = 0.0
        every = self.accumulate_every
        for state in prolonged:
            state.zero()
        for n, block in enumerate(blocks):
            if n % every:
                continue
            lam.assign(block.adj_sol)
            if every == 1:
                # Consecutive steps share two states: shift them, prolong one.
                w_nm1.assign(w_n)
                w_n.assign(w_np1)
                prolong(n, w_np1)
            else:
                prolong(n, w_np1)
                prolong(n - 1, w_n)
                prolong(n - 2, w_nm1)
            fire.assemble(residual, tensor=buffer)
            eta += every * buffer.dat.data_ro
        for n, block in enumerate(blocks):
            if wavelet[n] != 0.0:
                source_sum += wavelet[n] * comm.allreduce(
                    float(np.dot(source.dat.data_ro, block.adj_sol.dat.data_ro)))

        two_j = 2.0 * float(functional)
        total = comm.allreduce(float(eta.sum())) - source_sum
        result.identity_ratios[shot] = total / two_j if two_j else float("nan")
        result.source_terms[shot] = source_sum
        result.seconds[shot] = [time.perf_counter() - t_start]
        return eta
