"""Composition of data discrepancy and physical regularization."""

from collections.abc import Mapping

from .misfit import L2DataMisfit
from .regularization import H1Regularization


class InversionObjective:
    """An inverse objective, independent of optimization coordinates.

    Parameters
    ----------
    misfit : L2DataMisfit, optional
        Local receiver-space discrepancy.
    regularization : H1Regularization, optional
        Physical spatial penalty, disabled by default.
    """

    def __init__(self, misfit: L2DataMisfit | None = None,
                 regularization: H1Regularization | None = None) -> None:
        self.misfit = L2DataMisfit() if misfit is None else misfit
        self.regularization = regularization

    def local_value(self, data_value: object, parameters: Mapping,
                    ensemble_size: int = 1) -> object:
        """Add one ensemble member's share of the physical penalty.

        Parameters
        ----------
        data_value : float or pyadjoint.AdjFloat
            Recorded data discrepancy for this ensemble member.
        parameters : mapping or PhysicalParameters
            Physical material fields.
        ensemble_size : int, optional
            Number of ensemble members, not spatial ranks or physical shots.

        Returns
        -------
        float or pyadjoint.AdjFloat
            Local objective whose ensemble sum contains the penalty once.
        """
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive.")
        if self.regularization is None:
            return data_value
        return data_value + self.regularization(parameters) / ensemble_size
