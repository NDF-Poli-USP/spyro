from abc import abstractmethod, ABCMeta
from firedrake import Constant

from ..wave import Wave
from ...utils.typing import AdjointType, RieszMapType, override, WaveType


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
        d = self.input_dictionary.get("synthetic_data", False)
        if bool(d) and "type" in d:
            if d["type"] == "object":
                self.initialize_model_parameters_from_object(d)
            elif d["type"] == "file":
                self.initialize_model_parameters_from_file(d)
            else:
                raise Exception(f"Invalid synthetic data type: {d['type']}")
        else:
            raise Exception("Input dictionary must contain ['synthetic_data']['type']")

    @abstractmethod
    def initialize_model_parameters_from_object(self, synthetic_data_dict):
        pass

    @abstractmethod
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        pass

    @override
    def gradient_solve(
        self,
        misfit=None,
        forward_solution=None,
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        riesz_map=RieszMapType.L2,
    ):
        """Compute the adjoint gradient of the elastic misfit functional.

        Only the automated adjoint is available for elastic media: the
        gradient is obtained by replaying the pyadjoint tape recorded during an
        annotated ``forward_solve()``. It is taken with respect to the three
        parameters the material was declared with, whichever parameterization
        that is, because those are the independent fields of the variational
        form; the remaining two are UFL expressions of them.

        Parameters
        ----------
        misfit : array_like, optional
            Accepted for signature compatibility with
            :meth:`AcousticWave.gradient_solve`; the automated adjoint reads
            the misfit from the recorded tape instead.
        forward_solution : firedrake.Function, optional
            Accepted for signature compatibility; unused for the same reason.
        adjoint_type : AdjointType, optional
            Must be :attr:`AdjointType.AUTOMATED_ADJOINT`.
        riesz_map : RieszMapType, optional
            ``L2`` returns gradients (``Function``), ``l2`` returns raw
            derivatives (``Cofunction``). See :class:`RieszMapType`.

        Returns
        -------
        dict
            Derivative of the functional with respect to each material
            control, keyed by :class:`ElasticMaterialParameter` in the order
            reported by ``get_control_parameters()``.

        Raises
        ------
        NotImplementedError
            If a hand-implemented adjoint is requested.

        Examples
        --------
        For a model declared with density and the two Lame parameters, this
        returns ``{DENSITY: dJ_drho, LAMBDA: dJ_dlambda, MU: dJ_dmu}``.
        """
        if adjoint_type is not AdjointType.AUTOMATED_ADJOINT:
            raise NotImplementedError(
                "Elastic media only support the automated adjoint; "
                f"got {adjoint_type}.",
            )
        derivatives = self._automated_adjoint_derivatives(riesz_map=riesz_map)
        return dict(zip(self.get_control_parameters(), derivatives))

    @override
    def update_source_expression(self, t):
        self.time.assign(t)
