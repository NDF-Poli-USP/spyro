"""Automatic mesh loop to a trace-error tolerance.

The user gives a tolerance on the relative trace error. Three strategies use
the same pilot, the same estimate and the same stopping rule and differ only
in how the next mesh is chosen:

``cpw``
    raises the cells per wavelength uniformly, :math:`C_{k+1} = C_k
    \\sqrt{N_{k+1}/N_k}`;
``gradient``
    redistributes the budget by the smoothed squared velocity gradient,
    zero in the absorbing pad;
``adjoint``
    redistributes the budget by the adjoint indicator of the pilot
    (:mod:`spyro.adaptivity.pilot`), each shot normalised and weighted by its
    squared trace error.

Iteration :math:`k`:

1. pilot on :math:`M_k`: :math:`\\eta = \\max_s \\eta_s`, the worst-shot
   relative trace error of :math:`M_k` against its uniform refinement;
2. stop when :math:`\\eta \\le \\theta\\,\\mathrm{tol}`;
3. grow the budget by :math:`\\operatorname{clip}(g_s
   (\\eta/(\\theta\\,\\mathrm{tol}))^{2/p}, g_{\\min}, g_{\\max})`, with the
   rate :math:`p` measured between iterations (``p0`` at the start);
4. build :math:`M_{k+1}` with the strategy, conforming to the given
   interfaces.

With ``initial_cpw="auto"`` an initial CPW phase raises :math:`C` from
``cpw_start`` by ``cpw_factor`` until the pilot is reliable
(:math:`\\eta \\le \\eta_{conf}`) or ``cpw_max`` is reached; its last mesh is
the first mesh of the loop, common to the three strategies.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import firedrake as fire

from .mesh_generation import (RectangularDomain, count_kmv_dofs,
                              generate_mesh, local_mesh_size,
                              read_triangulation)
from .pilot import PilotResult, TwoLevelPilot
from .size_fields import (AdaptiveSizeField, VelocityGrid, adjoint_density,
                          gradient_density)


class AdaptationStrategy(StrEnum):
    """How the next mesh of the loop is chosen."""

    CPW = "cpw"
    GRADIENT = "gradient"
    ADJOINT = "adjoint"


@dataclass
class LoopParameters:
    """Controller parameters of the automatic loop.

    Parameters
    ----------
    tolerance : float
        Target relative trace error.
    theta : float, optional
        Safety factor of the stopping rule. Default ``0.8``.
    initial_cpw : float or str, optional
        Cells per wavelength of the first mesh, or ``"auto"`` for the initial
        CPW phase. Default ``"auto"``.
    cpw_start : float, optional
        First :math:`C` of the initial phase. Default ``2``.
    cpw_factor : float, optional
        Factor of :math:`C` between meshes of the initial phase. Default ``1.5``.
    cpw_max : float, optional
        Largest :math:`C` of the initial phase. Default ``6``.
    eta_reliable : float, optional
        Pilot considered reliable when :math:`\\eta` is at most this.
        Default ``0.3``.
    p0 : float, optional
        Initial rate, error :math:`\\sim N^{-p/2}`. Default ``1.8``.
    max_iterations : int, optional
        Maximum number of refinements. Default ``6``.
    growth_safety : float, optional
        Factor on the predicted budget growth. Default ``1.0``.
    growth_min : float, optional
        Smallest budget growth per iteration. Default ``1.15``.
    growth_max : float, optional
        Largest budget growth per iteration. Default ``4``.
    order : float, optional
        Convergence order :math:`q` of the size law. Default ``2.7``.
    smoothing : float or None, optional
        Gaussian smoothing length (km); ``None`` uses
        :math:`60\\,\\mathrm{m} \\times 7/f`. Default ``None``.
    grid_spacing : float, optional
        Spacing of the design grid (km). Default ``0.005``.
    cmax_ratio : float, optional
        Largest local refinement relative to the uniform-CPW mesh with the
        same budget. Default ``1.25``.
    dof_tolerance : float, optional
        Relative tolerance of the budget of an adaptive mesh. Default ``0.02``.
    seed : int, optional
        gmsh seed of the meshes. Default ``0``.
    """

    tolerance: float
    theta: float = 0.8
    initial_cpw: float | str = "auto"
    cpw_start: float = 2.0
    cpw_factor: float = 1.5
    cpw_max: float = 6.0
    eta_reliable: float = 0.3
    p0: float = 1.8
    max_iterations: int = 6
    growth_safety: float = 1.0
    growth_min: float = 1.15
    growth_max: float = 4.0
    order: float = 2.7
    smoothing: float | None = None
    grid_spacing: float = 0.005
    cmax_ratio: float = 1.25
    dof_tolerance: float = 0.02
    seed: int = 0


class AutomaticMeshAdaptation:
    """Mesh loop to a trace-error tolerance with one of three strategies.

    Parameters
    ----------
    strategy : AdaptationStrategy or str
        ``"cpw"``, ``"gradient"`` or ``"adjoint"``.
    pilot : TwoLevelPilot
        Pilot used to estimate the error (and, for ``adjoint``, the indicator).
    velocity : VelocityGrid
        Velocity model used to design the meshes.
    domain : RectangularDomain
        Padded domain.
    frequency : float
        Peak frequency of the source (Hz).
    degree : int
        Degree of the KMV element (for the budgets).
    output_dir : str or pathlib.Path
        Directory of the meshes and of the report.
    parameters : LoopParameters
        Controller parameters.
    conforming_depths : sequence of float, optional
        Depths of horizontal interfaces imposed on every mesh.
    comm : mpi4py.MPI.Comm, optional
        Communicator of the pilots; meshes are built and the report written on
        its rank 0. Default ``firedrake.COMM_WORLD``.
    """

    def __init__(self, strategy: AdaptationStrategy | str, pilot: TwoLevelPilot,
                 velocity: VelocityGrid, domain: RectangularDomain,
                 frequency: float, degree: int, output_dir: str | Path,
                 parameters: LoopParameters,
                 conforming_depths: Sequence[float] = (),
                 comm: object = None) -> None:
        self.strategy = AdaptationStrategy(strategy)
        self.pilot = pilot
        self.velocity = velocity
        self.domain = domain
        self.frequency = float(frequency)
        self.degree = int(degree)
        self.output_dir = Path(output_dir)
        self.parameters = parameters
        self.conforming_depths = tuple(conforming_depths)
        self.comm = fire.COMM_WORLD if comm is None else comm
        self.smoothing = (parameters.smoothing if parameters.smoothing is not None
                          else 0.06 * 7.0 / self.frequency)
        self.z, self.x = domain.grid(parameters.grid_spacing)
        Z, X = np.meshgrid(self.z, self.x, indexing="ij")
        self.grid_velocity = velocity(Z, X)
        self.interior = domain.in_interior(Z, X)
        self.report = {
            "strategy": self.strategy.value,
            "parameters": asdict(parameters),
            "frequency_hz": self.frequency,
            "degree": self.degree,
            "smoothing_km": self.smoothing,
            "conforming_depths": list(self.conforming_depths),
            "initial_phase": [],
            "iterations": [],
        }

    # ------------------------------------------------------------ meshes
    def _on_root(self, build: callable) -> object:
        """Run ``build`` on rank 0 and broadcast its result.

        Parameters
        ----------
        build : callable
            Function without arguments.

        Returns
        -------
        object
            The broadcast result.
        """
        value = build() if self.comm.rank == 0 else None
        return self.comm.bcast(value, root=0)

    def _dofs(self, path: Path) -> int:
        """KMV degrees of freedom of a mesh file.

        Parameters
        ----------
        path : pathlib.Path
            Mesh file.

        Returns
        -------
        int
            Dimension of the scalar KMV space.
        """
        _, triangles = read_triangulation(path)
        return count_kmv_dofs(triangles, self.degree)

    def cpw_mesh(self, cpw: float) -> tuple[Path, int]:
        """Build (or reuse) a uniform-CPW mesh.

        Parameters
        ----------
        cpw : float
            Cells per wavelength, :math:`h = v_p/(C f)`.

        Returns
        -------
        tuple
            Mesh path and its number of degrees of freedom.
        """
        name = f"cpw_C{cpw:.4g}_s{self.parameters.seed}".replace(".", "p")
        path = self.output_dir / "meshes" / f"{name}.msh"

        def build() -> tuple[Path, int]:
            if not path.exists():
                generate_mesh(path, self.domain,
                              lambda z, x: float(self.velocity(z, x)) / (cpw * self.frequency),
                              self.conforming_depths, seed=self.parameters.seed)
            return path, self._dofs(path)

        return self._on_root(build)

    def adaptive_mesh(self, name: str, density: np.ndarray, current: Path,
                      target: int, equivalent_cpw: float) -> tuple[Path, int]:
        """Build a mesh by the adaptive law with a given budget.

        Parameters
        ----------
        name : str
            Mesh name.
        density : numpy.ndarray
            Refinement density on the design grid.
        current : pathlib.Path
            Current mesh, whose local size caps the new one.
        target : int
            Target number of degrees of freedom.
        equivalent_cpw : float
            Cells per wavelength of a uniform mesh with that budget.

        Returns
        -------
        tuple
            Mesh path and its number of degrees of freedom.
        """
        path = self.output_dir / "meshes" / f"{name}.msh"
        p = self.parameters

        def build() -> tuple[Path, int]:
            points, triangles = read_triangulation(current)
            current_size = local_mesh_size(points, triangles, self.z, self.x)
            field = AdaptiveSizeField(density, current_size, self.z, self.x, p.order)
            field.set_floor(self.grid_velocity, self.frequency, equivalent_cpw,
                            p.cmax_ratio)
            # DOFs per unit of sum(h^-2) area, measured on the current mesh.
            calibration = (count_kmv_dofs(triangles, self.degree)
                           / float((current_size ** -2).sum() * p.grid_spacing**2))
            scale = field.scale_for(target / calibration)
            dofs = 0
            for _ in range(6):
                field.set_scale(scale)
                generate_mesh(path, self.domain, field, self.conforming_depths,
                              seed=p.seed)
                dofs = self._dofs(path)
                if abs(dofs / target - 1.0) <= p.dof_tolerance:
                    break
                scale *= math.sqrt(dofs / target)
            return path, dofs

        return self._on_root(build)

    # ------------------------------------------------------------ pilots
    @staticmethod
    def _work(result: PilotResult, adjoint: bool) -> dict:
        """Cost of a pilot in DOF-steps and wall time.

        Parameters
        ----------
        result : PilotResult
            The pilot.
        adjoint : bool
            Whether the adjoint was computed.

        Returns
        -------
        dict
            Work and wall time of the forwards and of the adjoint.
        """
        shots = len(result.trace_errors)
        seconds = np.array(list(result.seconds.values()))
        return {
            "work_forwards": (result.dofs_coarse + result.dofs_fine) * result.steps * shots,
            "work_adjoint": result.dofs_fine * result.steps * shots if adjoint else 0,
            "wall_forwards_s": float(seconds[:, :2].sum()),
            "wall_adjoint_s": float(seconds[:, 2].sum()),
        }

    def _write(self) -> None:
        """Write the report on rank 0.

        Returns
        -------
        None
            ``report.json`` is written in the output directory.
        """
        if self.comm.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"report_{self.strategy.value}.json"
            path.write_text(json.dumps(self.report, indent=1, default=str))

    # ------------------------------------------------------------ loop
    def initial_mesh(self) -> tuple[Path, int, float]:
        """First mesh of the loop, from a fixed CPW or the initial CPW phase.

        Returns
        -------
        tuple
            Mesh path, its degrees of freedom and its cells per wavelength.
        """
        p = self.parameters
        if str(p.initial_cpw).lower() != "auto":
            cpw = float(p.initial_cpw)
            path, dofs = self.cpw_mesh(cpw)
            return path, dofs, cpw
        cpw = p.cpw_start
        while True:
            path, dofs = self.cpw_mesh(cpw)
            result = self.pilot.run(str(path))
            eta = result.worst_trace_error
            record = {"cpw": cpw, "mesh": path.name, "dofs": dofs,
                      "eta_per_shot": result.trace_errors, "eta_worst": eta,
                      **self._work(result, False)}
            if eta <= p.theta * p.tolerance:
                record["decision"] = "tolerance already met"
            elif eta > p.eta_reliable and cpw * p.cpw_factor <= p.cpw_max + 1e-9:
                record["decision"] = "pilot not reliable: raise C"
            elif eta > p.eta_reliable:
                record["decision"] = "cpw_max reached"
            else:
                record["decision"] = "pilot reliable: start refining"
            self.report["initial_phase"].append(record)
            self._write()
            if record["decision"] != "pilot not reliable: raise C":
                return path, dofs, cpw
            cpw *= p.cpw_factor

    def run(self) -> dict:
        """Run the loop until the tolerance or the iteration limit.

        Returns
        -------
        dict
            The report, also written to ``report_<strategy>.json``.
        """
        p = self.parameters
        start = time.perf_counter()
        mesh, dofs, cpw0 = self.initial_mesh()
        self.report["initial_cpw_effective"] = cpw0
        dofs0, cpw = dofs, cpw0
        history = []
        adjoint = self.strategy is AdaptationStrategy.ADJOINT
        for k in range(p.max_iterations + 1):
            result = self.pilot.run(str(mesh), compute_indicator=adjoint)
            eta = result.worst_trace_error
            history.append((dofs, eta))
            rate = p.p0
            if len(history) >= 2 and history[-1][1] < history[-2][1]:
                observed = (2.0 * math.log(history[-2][1] / history[-1][1])
                            / math.log(history[-1][0] / history[-2][0]))
                if 0.5 <= observed <= 6.0:
                    rate = observed
            accept = eta <= p.theta * p.tolerance
            iteration = {
                "k": k, "mesh": mesh.name, "dofs": dofs,
                "cpw_equivalent": cpw0 * math.sqrt(dofs / dofs0),
                "eta_per_shot": result.trace_errors, "eta_worst": eta,
                "steps": result.steps, "dt": result.dt, "rate_used": rate,
                "identity_ratios": result.identity_ratios,
                **self._work(result, adjoint),
                "decision": ("accept" if accept else
                             "iteration limit" if k == p.max_iterations else "refine"),
            }
            self.report["iterations"].append(iteration)
            self._write()
            if accept or k == p.max_iterations:
                break

            growth = min(max(p.growth_safety
                             * (eta / (p.theta * p.tolerance)) ** (2.0 / rate),
                             p.growth_min), p.growth_max)
            target = int(dofs * growth)
            if self.strategy is AdaptationStrategy.CPW:
                cpw *= math.sqrt(growth)
                mesh, dofs = self.cpw_mesh(cpw)
            else:
                if adjoint:
                    density = self._on_root(lambda: adjoint_density(
                        result.centroids, result.indicators,
                        np.array([result.trace_errors[s] ** 2
                                  for s in sorted(result.trace_errors)]),
                        self.z, self.x, self.smoothing))
                else:
                    density = gradient_density(self.grid_velocity, p.grid_spacing,
                                               self.smoothing, self.interior)
                name = f"{self.strategy.value}_it{k + 1}_s{p.seed}"
                mesh, dofs = self.adaptive_mesh(
                    name, density, mesh, target, cpw0 * math.sqrt(target / dofs0))
            iteration["next"] = {"growth": growth, "target_dofs": target,
                                 "obtained_dofs": dofs, "mesh": mesh.name}
            self._write()

        final = self.report["iterations"][-1]
        self.report["accepted"] = final["decision"] == "accept"
        self.report["accepted_mesh"] = str(mesh)
        phase = self.report["initial_phase"]
        # The last mesh of the initial phase is the first of the loop: its
        # forward-only pilot is extra work only for the adjoint strategy.
        counted = phase if adjoint else phase[:-1]
        iterations = self.report["iterations"]
        self.report["search_cost"] = {
            "work": sum(i["work_forwards"] + i["work_adjoint"] for i in iterations)
            + sum(r["work_forwards"] for r in counted),
            "wall_s": sum(i["wall_forwards_s"] + i["wall_adjoint_s"] for i in iterations)
            + sum(r["wall_forwards_s"] for r in counted),
            "iterations": len(iterations),
            "initial_phase_pilots": len(counted),
        }
        self.report["elapsed_s"] = time.perf_counter() - start
        self._write()
        return self.report
