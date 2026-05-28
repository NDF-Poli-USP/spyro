import numpy as np

from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       FunctionSpace, project, VectorFunctionSpace)

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.typing import ElasticMaterialParameter, override, WaveType
from ...domains.space import create_function_space


LAME_CONTROL_PARAMETERS = (
    ElasticMaterialParameter.DENSITY,
    ElasticMaterialParameter.LAMBDA,
    ElasticMaterialParameter.MU,
)
VELOCITY_CONTROL_PARAMETERS = (
    ElasticMaterialParameter.DENSITY,
    ElasticMaterialParameter.P_WAVE_VELOCITY,
    ElasticMaterialParameter.S_WAVE_VELOCITY,
)
ELASTIC_PARAMETER_ALIASES = {
    "lmbda": ElasticMaterialParameter.LAMBDA,
    "lame_first": ElasticMaterialParameter.LAMBDA,
    "lame_second": ElasticMaterialParameter.MU,
}


class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.ISOTROPIC_ELASTIC
        self.rho = None   # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity
        self._control_parameter_names = None
        self._material_parameter_function_space = None

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
        self.D_h = None
        self.field_logger.add_field("p-wave", "P-wave",
                                    lambda: self.update_p_wave())

        # Variables for logging the S-wave
        self.s_wave = None
        self.C_h = None
        self.field_logger.add_field("s-wave", "S-wave",
                                    lambda: self.update_s_wave())

        self.mechanical_energy = None
        self.field_logger.add_functional("mechanical_energy",
                                         lambda: assemble(self.mechanical_energy))

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        def material_parameter(value):
            if np.isscalar(value) or isinstance(value, Constant):
                if self.mesh is None:
                    return Constant(value) if np.isscalar(value) else value
                V = create_function_space(
                    self.mesh,
                    self.method,
                    self.degree,
                    dim=1,
                )
                return Function(V).interpolate(value)
            return value

        def get_value(parameter, *aliases):
            for key in (parameter.value, *aliases):
                if key in synthetic_data_dict:
                    return material_parameter(synthetic_data_dict[key])
            return None

        self.rho = get_value(ElasticMaterialParameter.DENSITY)
        self.lmbda = get_value(
            ElasticMaterialParameter.LAMBDA,
            "lame_first",
        )
        self.mu = get_value(
            ElasticMaterialParameter.MU,
            "lame_second",
        )
        self.c = get_value(ElasticMaterialParameter.P_WAVE_VELOCITY)
        self.c_s = get_value(ElasticMaterialParameter.S_WAVE_VELOCITY)

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
            self._control_parameter_names = LAME_CONTROL_PARAMETERS
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        elif option_2:
            self._control_parameter_names = VELOCITY_CONTROL_PARAMETERS
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
        else:
            raise ValueError(
                "Inconsistent selection of isotropic elastic wave parameters:\n"
                f"    Density        : {bool(self.rho)}\n"
                f"    Lame first     : {bool(self.lmbda)}\n"
                f"    Lame second    : {bool(self.mu)}\n"
                f"    P-wave velocity: {bool(self.c)}\n"
                f"    S-wave velocity: {bool(self.c_s)}\n"
                "The valid options are {Density, Lame first, Lame second} "
                "or (exclusive) {Density, P-wave velocity, S-wave velocity}",
            )

    @override
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError

    @override
    def _create_function_space(self):
        return create_function_space(self.mesh, self.method, self.degree,
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
    def get_forward_solution_receivers(self):
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

    def get_control_parameter_function_space(self):
        """Return the scalar space used for elastic material controls."""
        if self.mesh is None:
            raise ValueError(
                "Mesh must be set before creating elastic control parameter spaces.",
            )
        self._material_parameter_function_space = create_function_space(
            self.mesh,
            self.method,
            self.degree,
        )
        return self._material_parameter_function_space

    def _coerce_material_parameter(self, value, name):
        """Return a material parameter as a scalar Firedrake Function.

        Elastic material parameters are scalar fields, while the elastic
        displacement solution lives in a vector function space. This helper
        keeps the inversion controls in the scalar space returned by
        ``get_control_parameter_function_space()`` so density, Lame
        parameters, and velocity controls can be flattened, rebuilt, written,
        and reassigned consistently during FWI.

        Accepted values are Firedrake ``Function`` objects, constants, scalar
        values, or UFL expressions. Functions already in the target space are
        copied with ``assign``; all other values are interpolated into a new
        named scalar ``Function``.
        """
        if value is None:
            return None

        V = self.get_control_parameter_function_space()
        field = Function(V, name=name)
        if isinstance(value, Function):
            if value.function_space() == V:
                field.assign(value)
            else:
                field.interpolate(value)
        else:
            field.interpolate(value)
        return field

    def _normalize_control_parameter(self, key):
        if isinstance(key, ElasticMaterialParameter):
            return key
        if isinstance(key, str):
            if key in ELASTIC_PARAMETER_ALIASES:
                return ELASTIC_PARAMETER_ALIASES[key]
            try:
                return ElasticMaterialParameter(key)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported elastic control parameter '{key}'.",
                ) from exc
        raise TypeError(
            "Elastic control parameter keys must be ElasticMaterialParameter "
            "instances or strings.",
        )

    def _normalize_control_parameters(self, controls):
        normalized = {}
        for key, value in controls.items():
            parameter = self._normalize_control_parameter(key)
            if parameter in normalized:
                raise ValueError(
                    f"Duplicated elastic control parameter '{parameter.value}'.",
                )
            normalized[parameter] = value
        return normalized

    def get_control_parameters(self):
        """Return the active isotropic elastic material controls."""
        names = self._control_parameter_names
        if names is None:
            if self.rho is None:
                return None
            names = LAME_CONTROL_PARAMETERS

        parameters = {}
        for name in names:
            if name is ElasticMaterialParameter.DENSITY:
                parameters[name] = self.rho
            elif name is ElasticMaterialParameter.LAMBDA:
                parameters[name] = self.lmbda
            elif name is ElasticMaterialParameter.MU:
                parameters[name] = self.mu
            elif name is ElasticMaterialParameter.P_WAVE_VELOCITY:
                parameters[name] = self.c
            elif name is ElasticMaterialParameter.S_WAVE_VELOCITY:
                parameters[name] = self.c_s
            else:
                raise ValueError(
                    f"Unsupported elastic control parameter '{name.value}'.",
                )
        return parameters

    def set_control_parameters(self, controls):
        """Assign isotropic elastic material controls.

        The preferred keys are :class:`ElasticMaterialParameter` values. Legacy
        strings from model dictionaries are still accepted for compatibility.
        """
        if not isinstance(controls, dict):
            raise TypeError(
                "IsotropicWave controls must be provided as a dictionary.",
            )

        normalized = self._normalize_control_parameters(controls)
        option_1 = set(normalized) == set(LAME_CONTROL_PARAMETERS)
        option_2 = set(normalized) == set(VELOCITY_CONTROL_PARAMETERS)
        if not (option_1 or option_2):
            raise ValueError(
                "Elastic controls must define either "
                "{density, lambda, mu} or "
                "{density, p_wave_velocity, s_wave_velocity}.",
            )

        self.rho = self._coerce_material_parameter(
            normalized[ElasticMaterialParameter.DENSITY],
            ElasticMaterialParameter.DENSITY.value,
        )

        synthetic_data = {
            "type": "object",
            "density": self.rho,
            "real_velocity_file": None,
        }
        if option_1:
            self.lmbda = self._coerce_material_parameter(
                normalized[ElasticMaterialParameter.LAMBDA],
                ElasticMaterialParameter.LAMBDA.value,
            )
            self.mu = self._coerce_material_parameter(
                normalized[ElasticMaterialParameter.MU],
                ElasticMaterialParameter.MU.value,
            )
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
            self._control_parameter_names = LAME_CONTROL_PARAMETERS
            synthetic_data["lambda"] = self.lmbda
            synthetic_data["mu"] = self.mu
        else:
            self.c = self._coerce_material_parameter(
                normalized[ElasticMaterialParameter.P_WAVE_VELOCITY],
                ElasticMaterialParameter.P_WAVE_VELOCITY.value,
            )
            self.c_s = self._coerce_material_parameter(
                normalized[ElasticMaterialParameter.S_WAVE_VELOCITY],
                ElasticMaterialParameter.S_WAVE_VELOCITY.value,
            )
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
            self._control_parameter_names = VELOCITY_CONTROL_PARAMETERS
            synthetic_data["p_wave_velocity"] = self.c
            synthetic_data["s_wave_velocity"] = self.c_s

        self.input_dictionary["synthetic_data"] = synthetic_data

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

    def rhs_no_pml_source(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.source_function

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
                raise Exception(
                    f"Unsupported boundary condition with tag: {tag}")
            self.bcs.append(DirichletBC(subspace, value, idbc))

    def parse_volumetric_forces(self):
        acquisition_dict = self.input_dictionary["acquisition"]
        body_forces_data = acquisition_dict.get("body_forces", None)
        if body_forces_data is not None:
            x_vec = self.get_spatial_coordinates()
            self.body_forces = body_forces_data(x_vec, self.time)

    def update_p_wave(self):
        if self.p_wave is None:
            self.D_h = FunctionSpace(self.mesh, "DG", 0)
            self.p_wave = Function(self.D_h)

        self.p_wave.assign(project(div(self.get_function()), self.D_h))

        return self.p_wave

    def update_s_wave(self):
        if self.s_wave is None:
            if self.dimension == 2:
                self.C_h = FunctionSpace(self.mesh, "DG", 0)
            else:
                self.C_h = VectorFunctionSpace(self.mesh, "DG", 0)
            self.s_wave = Function(self.C_h)

        self.s_wave.assign(project(curl(self.get_function()), self.C_h))

        return self.s_wave
