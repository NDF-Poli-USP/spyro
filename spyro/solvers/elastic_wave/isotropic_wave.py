from .elastic_wave import ElasticWave

class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''
    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)

        self.rho = None   # Density
        self.lmbda = None # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity
    
    #@override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        self.rho = synthetic_data_dict.get("density", None)
        self.lmbda = synthetic_data_dict.get("lambda", 
                                            synthetic_data_dict.get("lame_first", None))
        self.mu = synthetic_data_dict.get("mu", 
                                          synthetic_data_dict.get("lame_second", None))
        self.c = synthetic_data_dict.get("p_wave_velocity", None)
        self.c_s = synthetic_data_dict.get("s_wave_velocity", None)
        
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

        if not option_1 and not option_2:
            raise Exception(f"Inconsistent selection of isotropic elastic wave parameters:\n" \
                            f"    Density        : {bool(self.rho)}\n"\
                            f"    Lame first     : {bool(self.lmbda)}\n"\
                            f"    Lame second    : {bool(self.mu)}\n"\
                            f"    P-wave velocity: {bool(self.c)}\n"\
                            f"    S-wave velocity: {bool(self.c_s)}\n"\
                            "The valid options are \{Density, Lame first, Lame second\} "\
                            "or \{Density, P-wave velocity, S-wave velocity\}")
    
    #@override
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError
