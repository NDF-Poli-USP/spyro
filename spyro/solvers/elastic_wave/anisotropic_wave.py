import numpy as np
import firedrake as fire
from .elastic_wave import ElasticWave

from .forms import (isotropic_elastic_without_pml, isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...domains.space import FE_method
from ...utils.typing import override


# Work from Ruben Andres Salas and Alexandre Olender


class AnisotropicWave(ElasticWave):
    '''
    Class for the anisotropic elastic wave propagator.

    Attributes
    ----------
    p_wave: `firedrake.Function`
        P-wave field
    s_wave: `firedrake.Function`
        S-wave field

    Methods
    -------
    update_p_wave()
        Build the P-wave field
    update_s_wave()
        Build the S-wave fields



    '''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)

        # Material properties
        self.vP = None   # P-wave velocity [m/s]
        self.vS = None  # S-wave velocity [m/s]
        self.rho = None  # Density [kg/m³]
        self.eps1 = None  # Thomsen parameter epsilon
        self.gamma = None  # Thomsen parameter gamma
        self.delta = None  # Thomsen parameter delta
        self.theta = None  # Tilt angle in degrees
        self.phi = None  # Azimuth angle in degrees (phi = 0: 2D case)

        # State variables
        self.u_n = None   # Current displacement field
        self.u_nm1 = None  # Displacement field in previous iteration
        self.u_nm2 = None  # Displacement field at iteration n-2
        self.u_np1 = None  # Displacement field in next iteration

        # Volumetric sources (defined through UFL)
        self.body_forces = None

        # Boundary conditions
        self.bcs = []

        # Variables for logging the P-wave
        self.p_wave = None
        self.field_logger.add_field("p-wave", "P-wave",
                                    lambda: self.update_p_wave())

        # Variables for logging the S-wave
        self.s_wave = None
        self.field_logger.add_field("s-wave", "S-wave",
                                    lambda: self.update_s_wave())

        # Variables for logging the mechanical energy functional
        self.mechanical_energy = None
        self.field_logger.add_functional(
            "mechanical_energy", lambda: assemble(self.mechanical_energy))

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        def constant_wrapper(value):
            if np.isscalar(value):
                return fire.Constant(value)
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

    @override
    def _create_function_space(self):
        return FE_method(self.mesh, self.method, self.degree,
                         dim=self.dimension)

    @override
    def _set_vstate(self, vstate):
        self.u_n.assign(vstate)

    @override
    def _get_vstate(self):
        return self.u_n

    @override
    def _set_prev_vstate(self, vstate):
        if self.u_nm2 is not None:
            self.u_nm2.assign(self.u_nm1)
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

        abc_dict = self.input_dictionary.get("absorving_boundary_conditions", None)
        if abc_dict is not None:
            abc_active = abc_dict.get("status", False)
            if abc_active:
                dt_scheme = abc_dict.get("local", {}).get("dt_scheme", None)
                if dt_scheme == "backward_2nd":
                    self.u_nm2 = Function(self.function_space,
                                          name=self.get_function_name())

        self.mechanical_energy = mechanical_energy_form(self)

        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()

        if self.abc_boundary_layer_type is None or \
                self.abc_boundary_layer_type == "local":
            isotropic_elastic_without_pml(self)
        elif self.abc_boundary_layer_type == "PML":
            isotropic_elastic_with_pml(self)

    @override
    def rhs_no_pml(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.B

    def parse_initial_conditions(self):
        time_dict = self.input_dictionary["time_axis"]
        initial_condition = time_dict.get("initial_condition", None)
        if initial_condition is not None:
            x_vec = self.get_spatial_coordinates()
            self.u_n.interpolate(initial_condition(x_vec, 0 - self.dt))
            self.u_nm1.interpolate(initial_condition(x_vec, 0 - 2*self.dt))

    def parse_boundary_conditions(self):
        bc_list = self.input_dictionary.get("boundary_conditions", [])
        for tag, idbc, value in bc_list:
            if tag == "u":
                subspace = self.function_space
            elif tag == "uz":
                subspace = self.function_space.sub(0)
            elif tag == "ux":
                subspace = self.function_space.sub(1)
            elif tag == "uy":
                subspace = self.function_space.sub(2)
            else:
                raise Exception(f"Unsupported boundary condition with tag: {tag}")
            self.bcs.append(DirichletBC(subspace, value, idbc))

    def parse_volumetric_forces(self):
        acquisition_dict = self.input_dictionary["acquisition"]
        body_forces_data = acquisition_dict.get("body_forces", None)
        if body_forces_data is not None:
            x_vec = self.get_spatial_coordinates()
            self.body_forces = body_forces_data(x_vec, self.time)

    def update_p_wave(self, ele_degree=0):
        '''
        Build the P-wave field

        Parameters
        ----------
        ele_degree: `int`, optional
            Finite element degree for P-wave field

        Returns
        -------
        p_wave: `firedrake.Function`
            P-wave field
        '''
        if self.p_wave is None:
            ele_type = 'DG' if ele_degree == 0 else 'CG'
            P = fire.FunctionSpace(self.mesh, ele_type, ele_degree)
            self.p_wave = fire.Function(P, name='p_wave')

        return self.p_wave.interpolate(fire.div(self.get_function()))

    def update_s_wave(self, ele_degree=0):
        '''
        Build the S-wave fields

        Parameters
        ----------
        ele_degree: `int`, optional
            Finite element degree for P-wave field

        Returns
        -------
        s_wave: `firedrake.Function`
            S-wave field
        '''
        if self.s_wave is None:
            ele_type = 'DG' if ele_degree == 0 else 'CG'
            S = fire.VectorFunctionSpace(self.mesh, ele_type, ele_degree)
            self.s_wave = fire.Function(S, name='s_wave')

        return self.s_wave.interpolate(fire.curl(self.get_function()))
