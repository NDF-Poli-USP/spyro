from abc import abstractmethod, ABCMeta
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
    def _initialize_model_parameters(self):
        """Declare and materialize the material parameters.

        The source of each parameter (constant, function, expression or file)
        is resolved by ``set_material_property`` during materialization, so no
        per-source dispatch is needed here.
        """
        d = self.input_dictionary.get("synthetic_data", False)
        if not bool(d):
            raise Exception("Input dictionary must contain ['synthetic_data']")

        self.declare_model_parameters(d)
        self.materialize_model_parameters()

    @abstractmethod
    def declare_model_parameters(self, synthetic_data_dict):
        """Phase A: read and validate the material declaration."""

    @abstractmethod
    def materialize_model_parameters(self):
        """Phase B: build every declared material parameter as a Function."""

    @override
    def gradient_solve(self, guess=None, misfit=None, forward_solution=None):
        raise NotImplementedError(
            "Elastic adjoint gradients are not implemented yet.",
        )

    @override
    def update_source_expression(self, t):
        self.time.assign(t)
