import numpy as np
from firedrake import Constant
from .elastic_wave import ElasticWave
from ...utils.typing import override
from warnings import warn


class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''

    def __init__(self, dictionary, comm=None):

        # Add deprecation warning
        warn("IsotropicWave class is deprecated and will be removed in a future version. "
             "Please use the updated ElasticWave class instead.",
             DeprecationWarning, stacklevel=2)

        super().__init__(dictionary, comm=comm)
        self.rho = None   # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity

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

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        def constant_wrapper(value):
            if np.isscalar(value):
                return Constant(value)
            else:
                return value

        def get_value(key, default=None):
            return constant_wrapper(synthetic_data_dict.get(key, default))

        self.rho = get_value("density")
        self.lmbda = get_value("lambda", default=get_value("lame_first"))
        self.mu = get_value("mu", get_value("lame_second"))
        self.c = get_value("p_wave_velocity")
        self.c_s = get_value("s_wave_velocity")

        # Check if {rho, lambda, mu} is set and {c, c_s} are not
        option_1 = bool(self.rho) and \
            bool(self.lmbda) and \
            bool(self.mu) and \
            not bool(self.c) and \
            not bool(self.c_s)
        # Check if {rho, c, c_s} is set and {lambda, mu} are not
        option_2 = bool(self.rho) and \
            bool(self.c) and \
            bool(self.c_s) and \
            not bool(self.lmbda) and \
            not bool(self.mu)

        if option_1:
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        elif option_2:
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
        else:
            raise Exception(f"Inconsistent selection of isotropic elastic wave parameters:\n"
                            f"    Density        : {bool(self.rho)}\n"
                            f"    Lame first     : {bool(self.lmbda)}\n"
                            f"    Lame second    : {bool(self.mu)}\n"
                            f"    P-wave velocity: {bool(self.c)}\n"
                            f"    S-wave velocity: {bool(self.c_s)}\n"
                            "The valid options are {Density, Lame first, Lame second} "
                            "or (exclusive) {Density, P-wave velocity, S-wave velocity}")

    @override
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError
