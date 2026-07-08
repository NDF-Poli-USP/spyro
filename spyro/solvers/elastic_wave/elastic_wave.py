from abc import abstractmethod, ABCMeta
from firedrake import Constant

from ..backward_time_integration import backward_wave_propagator
from ..wave import Wave
from ...utils.typing import (
    AdjointType, ImplementedAdjointDerivation, override, RieszMapType,
)


class ElasticWave(Wave, metaclass=ABCMeta):
    '''Base class for elastic wave propagators'''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
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
        guess=None,
        misfit=None,
        forward_solution=None,
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
        riesz_map=RieszMapType.L2,
        implemented_adjoint_derivation=ImplementedAdjointDerivation.UFL_DIFFERENTIATION,
    ):
        """Compute UFL-derived implemented-adjoint elastic gradients.

        The current elastic backend supports isotropic, non-PML forward
        residuals exposed through ``forward_residual_form``.  The returned
        gradient has the same structure as ``get_control_parameters()``: for
        isotropic elastic waves it is a dictionary keyed by material parameter.
        """
        if adjoint_type != AdjointType.IMPLEMENTED_ADJOINT:
            raise NotImplementedError(
                "Elastic gradients currently support only IMPLEMENTED_ADJOINT.",
            )
        if riesz_map != RieszMapType.L2:
            raise NotImplementedError(
                f"Riesz map {riesz_map} not implemented for elastic gradients.",
            )
        if (
            implemented_adjoint_derivation
            is not ImplementedAdjointDerivation.UFL_DIFFERENTIATION
        ):
            raise NotImplementedError(
                "Elastic gradients currently support only UFL differentiation.",
            )
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError(
                "Elastic implemented adjoint does not support PML yet.",
            )

        self._prepare_implemented_adjoint(
            misfit=misfit, forward_solution=forward_solution,
            implemented_adjoint_derivation=implemented_adjoint_derivation,
        )
        return backward_wave_propagator(
            self,
            implemented_adjoint_derivation=implemented_adjoint_derivation,
        )

    @override
    def update_source_expression(self, t):
        self.time.assign(t)
