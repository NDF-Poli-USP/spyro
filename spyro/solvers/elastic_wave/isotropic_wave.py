from firedrake import Function

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from ...domains.space import FE_method
from ...utils.typing import override

class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''
    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)

        self.rho = None   # Density
        self.lmbda = None # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity

        self.u_n = None   # Current displacement field
        self.u_nm1 = None # Displacement field in previous iteration
        self.u_np1 = None # Displacement field in next iteration
    
    @override
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
    
    @override
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError
    
    @override
    def _create_function_space(self):
        return FE_method(self.mesh, self.method, self.degree,
                         dim=self.mesh.ufl_cell().geometric_dimension())

    @override
    def _set_vstate(self, vstate):
        self.u_n.assign(vstate)

    @override
    def _get_vstate(self):
        return self.u_n

    @override
    def _set_prev_vstate(self, vstate):
        self.u_nm1.assign(vstate)

    @override
    def _get_prev_vstate(self):
        return self.u_nm1

    @override
    def _set_next_vstate(self, vstate):
        self.u_np1.assign(vstate)

    @override
    def _get_next_vstate(self):
        return self.u_np1
    
    @override
    def get_receivers_output(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self):
        return self.u_n

    @override
    def get_function_name(self):
        return "Displacement"

    @override
    def matrix_building(self):
        self.current_time = 0.0

        self.u_n = Function(self.function_space,
                            name=self.get_function_name())
        self.u_nm1 = Function(self.function_space,
                              name=self.get_function_name())
        self.u_np1 = Function(self.function_space,
                              name=self.get_function_name())

        if self.abc_boundary_layer_type is None:
            isotropic_elastic_without_pml(self)
        elif self.abc_boundary_layer_type == "PML":
            isotropic_elastic_with_pml(self)
    
    @override
    def rhs_no_pml(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.B