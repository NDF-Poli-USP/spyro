from abc import abstractmethod, ABCMeta
from firedrake import Constant
from warnings import warn

from ..backward_time_integration import backward_wave_propagator
from ..wave import Wave
from ...utils.typing import (
    AbsorbingBCsType, AdjointType, ImplementedAdjointDerivation, override,
    RieszMapType, WaveType,
)


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
            warn("Input dictionary does not contain ['synthetic_data']['type']")

    @abstractmethod
    def initialize_model_parameters_from_object(self, synthetic_data_dict):
        pass

    @abstractmethod
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        pass

    def set_physical_parameterization(self, parameterization) -> None:
        """Set which elastic parameters carry the material data.

        An elastic medium is described by more than one set of physical
        parameters -- an isotropic one by density with the Lame parameters,
        or by density with the two wave speeds. The solver reads all of them
        whatever this is set to; what it chooses is which ones hold the data
        as fields, the rest being expressions computed from those.

        Only a field has degrees of freedom to perturb, so this decides what
        can be differentiated with respect to, not what the equation is
        solved for. It is a decision about the equation, made before any
        parameter is selected as a control.

        Initializing the material properties already picks a set, by reading
        whichever one the model input declares. This method is public because
        that first choice is worth revising: an inversion is not obliged to
        invert in the parameters its model happens to be stored in, and which
        set it uses changes the conditioning of the problem and the cross-talk
        between parameters. Exposing the change of variables keeps it in the
        solver, where it is the same one initialization performs, rather than
        leaving callers to convert their model input by hand and get a factor
        or a sign wrong.

        Parameters
        ----------
        parameterization : enum.Enum
            Set of elastic parameters to carry the data.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always, unless a subclass implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a change of physical "
            "parameterization.",
        )

    @override
    def gradient_solve(
        self,
        guess=None,
        misfit=None,
        forward_solution=None,
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
        riesz_map=RieszMapType.L2,
    ):
        """Compute UFL-derived implemented-adjoint elastic gradients.

        The current elastic backend supports isotropic, non-PML forward
        residuals exposed through ``forward_residual_form``.

        Parameters
        ----------
        guess : optional
            Unused; kept for the signature of :meth:`Wave.gradient_solve`.
        misfit : array_like, optional
            Receiver misfit that drives the adjoint.
        forward_solution : list of firedrake.Function, optional
            Stored forward states; a forward solve runs when absent.
        adjoint_type : AdjointType, optional
            Must be :attr:`AdjointType.UFL_DERIVED_ADJOINT`.
        riesz_map : RieszMapType, optional
            Only ``L2`` is supported.

        Returns
        -------
        PhysicalParameters
            Gradients keyed by the selected material parameters.
        """
        if (
            adjoint_type.implemented_derivation
            is not ImplementedAdjointDerivation.UFL_DIFFERENTIATION
        ):
            raise NotImplementedError(
                "Elastic gradients currently support only "
                "UFL_DERIVED_ADJOINT.",
            )
        if riesz_map != RieszMapType.L2:
            raise NotImplementedError(
                f"Riesz map {riesz_map} not implemented for elastic gradients.",
            )
        if self.abc_type == AbsorbingBCsType.PML:
            raise NotImplementedError(
                "Elastic implemented adjoint does not support PML yet.",
            )

        self._prepare_implemented_adjoint(
            misfit=misfit, forward_solution=forward_solution,
            adjoint_type=adjoint_type,
        )
        return backward_wave_propagator(
            self,
            adjoint_type=adjoint_type,
        )

    @override
    def update_source_expression(self, t):
        self.time.assign(t)
