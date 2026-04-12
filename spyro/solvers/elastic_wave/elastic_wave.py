"""Base Elastic wave abstract class.

Add everything wave-related and common to all elastic wave solvers here.
"""

from abc import abstractmethod, ABCMeta
from firedrake import Constant

from ..wave import Wave
from ...utils.typing import override


class ElasticWave(Wave, metaclass=ABCMeta):
    """Base class for elastic wave propagators."""

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
        """Initilize model parameters from object, based on this abstract method."""
        pass

    @abstractmethod
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        """Initilize model parameters from file, based on this abstract method."""
        pass

    @override
    def update_source_expression(self, t):
        """Update source expression based on the time."""
        self.time.assign(t)
