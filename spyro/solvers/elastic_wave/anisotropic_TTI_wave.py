from .anisotropic_VTI_wave import AnisotropicVTIWave
from ...utils.typing import AnisotropicMaterialParameter, override, WaveType


class AnisotropicTTIWave(AnisotropicVTIWave):
    """Tilted transversely isotropic (TTI) elastic wave propagator."""

    def __init__(self, dictionary: dict, comm=None) -> None:
        super().__init__(dictionary, comm=comm,
                         wave_type=WaveType.ANISOTROPIC_TTI_ELASTIC)
        self.theta = None
        self.phi = None

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict) -> None:
        """Initialize VTI parameters plus the TTI tilt angles.

        Parameters
        ----------
        synthetic_data_dict : dict
            Model dictionary; in addition to the VTI keys, may define
            ``theta`` and ``phi`` (degrees).

        Returns
        -------
        None
        """
        super().initialize_model_parameters_from_object(synthetic_data_dict)
        self.theta = self._declared(synthetic_data_dict, AnisotropicMaterialParameter.THETA)
        self.phi = self._declared(synthetic_data_dict, AnisotropicMaterialParameter.PHI)

    @override
    def _anisotropic_synthetic_data(self) -> dict:
        """Return the VTI anisotropy entries plus ``theta`` and ``phi``.

        Returns
        -------
        dict
            Anisotropy entries of the ``synthetic_data`` dictionary.
        """
        return {**super()._anisotropic_synthetic_data(),
                "theta": self.theta, "phi": self.phi}
