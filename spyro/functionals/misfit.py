"""Receiver-space data misfits."""

import firedrake as fire
import numpy as np

from ..utils.typing import FunctionalEvaluationMode


class L2DataMisfit:
    """Half the squared receiver residual integrated by the trapezoidal rule.

    Notes
    -----
    Evaluation is local to an ensemble member. Spatial reduction for VOM
    receivers is performed by Firedrake assembly, not a second MPI sum.
    """

    def __call__(
        self, wave: object, misfit: object,
        evaluation_mode: FunctionalEvaluationMode = FunctionalEvaluationMode.AFTER_SOLVE,
        step: int | None = None, nsteps: int | None = None,
    ) -> object:
        """Evaluate the local data objective or one temporal contribution.

        Parameters
        ----------
        wave : object
            Solver providing dt and use_vertex_only_mesh.
        misfit : object
            VOM residual Function or NumPy receiver residual array.
        evaluation_mode : FunctionalEvaluationMode, optional
            Complete trace or individual time-step evaluation.
        step : int, optional
            Time-step index for per-step evaluation.
        nsteps : int, optional
            Number of recorded temporal samples.

        Returns
        -------
        float or pyadjoint.AdjFloat
            Local squared data residual integral.
        """
        if evaluation_mode == FunctionalEvaluationMode.PER_TIMESTEP:
            if nsteps is None or nsteps < 2 or step is None or not 0 <= step < nsteps:
                raise ValueError("Per-step L2 evaluation needs valid step and nsteps >= 2.")
            weight = 0.5 if step in (0, nsteps - 1) else 1.0
            if wave.use_vertex_only_mesh:
                return fire.assemble(
                    0.5 * wave.dt * weight * fire.inner(misfit, misfit) * fire.dx,
                )
            if not isinstance(misfit, np.ndarray):
                raise ValueError("Non-VOM residuals must be NumPy arrays.")
            return 0.5 * wave.dt * weight * np.sum(misfit ** 2)
        if evaluation_mode != FunctionalEvaluationMode.AFTER_SOLVE:
            raise ValueError("Unsupported L2 evaluation mode.")
        return 0.5 * np.sum(np.trapezoid(np.asarray(misfit) ** 2, dx=wave.dt, axis=0))
