from abc import abstractmethod, ABCMeta
from firedrake import Constant

from ..wave import Wave
from ...utils.typing import override, WaveType


class ElasticWave(Wave, metaclass=ABCMeta):
    """Base class for elastic wave propagators."""

    def __init__(self, dictionary, anisotropy="ISO", comm=None):
        """Wave Elastic object solver.

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the Wave class.
            Default is None
        anisotropy : `str`, optional
            The type of anisotropy in the medium. Oprions: "ISO, "VTI" or "TTI"
        comm : `object`, optional
            MPI communicator for parallel execution. Default is None

        Returns
        -------
        None
        """

        # Type of wave equation to solve
        if anisotropy == "ISO":
            wave_type = WaveType.ISOTROPIC_ELASTIC
        elif anisotropy == "VTI":
            wave_type = WaveType.ANISOTROPIC_VTI_ELASTIC
        elif anisotropy == "TTI":
            wave_type = WaveType.ANISOTROPIC_TTI_ELASTIC

        super().__init__(dictionary, wave_type=wave_type, comm=comm)
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
    def update_source_expression(self, t):
        self.time.assign(t)
