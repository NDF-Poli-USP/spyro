from abc import ABCMeta
from firedrake import Constant

from ..wave import Wave
from ...utils.typing import override, WaveType


class ElasticWave(Wave, metaclass=ABCMeta):
    """Base class for elastic wave propagators."""

    def __init__(self, dictionary, anisotropy=WaveType.ISOTROPIC_ELASTIC, comm=None):
        """Wave Elastic object solver.

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the Wave class.
            Default is `None`.
        anisotropy : `WaveType`, optional
            The type of anisotropy in the medium. Options:
            - ISOTROPIC_ELASTIC: Isotropic elastic wave equation for Isotropic media.
            - ANISOTROPIC_VTI_ELASTIC: Anisotropic elastic wave equation for VTI media.
            - ANISOTROPIC_TTI_ELASTIC: Anisotropic elastic wave equation for TTI media.
        comm : `object`, optional
            MPI communicator for parallel execution. Default is `None`.

        Returns
        -------
        None
        """

        super().__init__(dictionary, wave_type=anisotropy, comm=comm)
        self.time = Constant(0)  # Time variable

    @override
    def gradient_solve(self, guess=None, misfit=None, forward_solution=None):
        raise NotImplementedError(
            "Elastic adjoint gradients are not implemented yet.",
        )

    @override
    def update_source_expression(self, t):
        self.time.assign(t)
