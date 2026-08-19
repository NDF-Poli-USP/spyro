import numpy as np

from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       FunctionSpace, project)

from .elastic_wave import IsotropicElasticWave
from .forms import (elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.typing import (ElasticMaterialParameter, ElasticMaterialParameterization,
                             AbsorbingBCsType, override, WaveType)
from ...domains.space import create_function_space
from .tensor_computation import C_computation

CONTROL_PARAMETERS_BY_PARAMETERIZATION = {
    ElasticMaterialParameterization.LAME: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.LAMBDA,
        ElasticMaterialParameter.MU,
        ElasticMaterialParameter.DELTA,
        ElasticMaterialParameter.EPSILON,
        ElasticMaterialParameter.GAMMA,
        ElasticMaterialParameter.THETA,
        ElasticMaterialParameter.PHI,
    ),
    ElasticMaterialParameterization.VELOCITY: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
        ElasticMaterialParameter.DELTA,
        ElasticMaterialParameter.EPSILON,
        ElasticMaterialParameter.GAMMA,
        ElasticMaterialParameter.THETA,
        ElasticMaterialParameter.PHI,
    ),
}


def _format_control_parameters(parameters):
    """Format material-parameter enum values for error messages.

    Parameters
    ----------
    parameters : iterable of ElasticMaterialParameter
        Material-parameter enum values to display.

    Returns
    -------
    str
        Human-readable set-like representation using public parameter names.

    Examples
    --------
    ``(ElasticMaterialParameter.DENSITY, ElasticMaterialParameter.MU)``
    becomes ``"{density, mu}"``.
    """
    return "{" + ", ".join(parameter.value for parameter in parameters) + "}"


class AnisotropicTTIWave(IsotropicElasticWave):
    '''Anisotropic elastic wave propagator'''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, anisotropy = WaveType.ANISOTROPIC_TTI_ELASTIC, comm=comm)
        self.delta = None
        self.epsilon = None
        self.gamma = None
        self.theta = None
        self.phi = None

    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        """Initialize isotropic elastic material parameters from a dictionary.

        The dictionary must define exactly one supported material
        parameterization: either density with Lame parameters, or density with
        P- and S-wave velocities. The missing derived parameters are computed
        from the provided set, and the active control parameterization is stored
        for FWI.

        Parameters
        ----------
        synthetic_data_dict : dict
            Material parameter dictionary using the public Spyro model schema.
            Valid combinations are ``density``, ``lambda`` (or ``lame_first``),
            and ``mu`` (or ``lame_second``); or ``density``,
            ``p_wave_velocity``, and ``s_wave_velocity``. Values may be
            scalars, Firedrake ``Constant`` objects, Firedrake ``Function``
            objects, or UFL expressions.

        Returns
        -------
        None
            The method assigns ``rho``, ``lmbda``, ``mu``, ``c``, ``c_s``, and
            the active control parameterization on ``self``.
        """
        def material_parameter(value):
            """Normalize model-dictionary values for elastic parameters.

            Parameters
            ----------
            value : scalar, firedrake.Constant, firedrake.Function, or UFL expression
                Material parameter read from ``synthetic_data_dict``.

            Returns
            -------
            firedrake.Constant, firedrake.Function, or object
                Scalars and ``Constant`` values are converted to scalar
                material ``Function`` objects once a mesh exists. Before mesh
                creation, scalar values remain as ``Constant`` values so the
                regular model initialization flow can continue.

            Examples
            --------
            ``density=1.0`` becomes ``Constant(1.0)`` before the mesh exists,
            and becomes a scalar material ``Function`` after the mesh has been
            created.
            """
            if np.isscalar(value) or isinstance(value, Constant):
                if self.mesh is None:
                    return Constant(value) if np.isscalar(value) else value
                V = create_function_space(
                    self.mesh, self.method, self.degree, dim=1,
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
        self.delta = get_value(ElasticMaterialParameter.DELTA)
        self.gamma = get_value(ElasticMaterialParameter.GAMMA)
        self.epsilon = get_value(ElasticMaterialParameter.EPSILON)
        self.theta = get_value(ElasticMaterialParameter.THETA)
        self.phi = get_value(ElasticMaterialParameter.PHI)
        self.anisotropy_type = synthetic_data_dict["anisotropy"]

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
            self._control_parameterization = ElasticMaterialParameterization.LAME
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        elif option_2:
            self._control_parameterization = ElasticMaterialParameterization.VELOCITY
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

    def get_control_parameters(self):
        """Return the active isotropic elastic material controls.

        The returned dictionary is keyed by
        :class:`ElasticMaterialParameter`. Its contents depend on the active
        parameterization: density/Lame parameters or density/P- and S-wave
        velocities.

        Returns
        -------
        dict or None
            Dictionary mapping material-parameter enum values to scalar
            Firedrake ``Function`` controls. Returns ``None`` if material
            parameters have not been initialized.

        Examples
        --------
        Lame parameterization returns ``{DENSITY: rho, LAMBDA: lmbda, MU: mu}``.
        Velocity parameterization returns
        ``{DENSITY: rho, P_WAVE_VELOCITY: c, S_WAVE_VELOCITY: c_s}``.
        """
        parameterization = self._control_parameterization
        if parameterization is None:
            if self.rho is None:
                return None
            parameterization = ElasticMaterialParameterization.LAME

        parameters = {}
        for parameter in CONTROL_PARAMETERS_BY_PARAMETERIZATION[parameterization]:
            if parameter is ElasticMaterialParameter.DENSITY:
                parameters[parameter] = self.rho
            elif parameter is ElasticMaterialParameter.LAMBDA:
                parameters[parameter] = self.lmbda
            elif parameter is ElasticMaterialParameter.MU:
                parameters[parameter] = self.mu
            elif parameter is ElasticMaterialParameter.P_WAVE_VELOCITY:
                parameters[parameter] = self.c
            elif parameter is ElasticMaterialParameter.S_WAVE_VELOCITY:
                parameters[parameter] = self.c_s
            elif parameter is ElasticMaterialParameter.DELTA:
                parameters[parameter] = self.delta
            elif parameter is ElasticMaterialParameter.EPSILON:
                parameters[parameter] = self.epsilon
            elif parameter is ElasticMaterialParameter.GAMMA:
                parameters[parameter] = self.gamma
            elif parameter is ElasticMaterialParameter.THETA:
                parameters[parameter] = self.theta
            elif parameter is ElasticMaterialParameter.PHI:
                parameters[parameter] = self.phi
            elif parameter is ElasticMaterialParameter.ANISOTROPY_TYPE:
                parameters[parameter] = self.anisotropy_type
            else:
                raise ValueError(
                    f"Unsupported elastic control parameter '{parameter.value}'.",
                )
        return parameters

    def set_control_parameters(self, controls):
        """Assign isotropic elastic material controls.

        Control dictionaries must use :class:`ElasticMaterialParameter` keys.
        Model input dictionaries still use the public Spyro string schema, but
        the FWI control API is intentionally enum-only.

        Parameters
        ----------
        controls : dict
            Dictionary containing either density/Lame controls or density/P-
            and S-wave velocity controls. Values may be Firedrake ``Function``
            objects, Firedrake ``Constant`` objects, scalars, or UFL
            expressions; all stored controls are scalar ``Function`` objects.

        Returns
        -------
        None
            The method updates ``rho``, ``lmbda``, ``mu``, ``c``, ``c_s`` and
            the active material parameterization.

        Raises
        ------
        TypeError
            If ``controls`` is not a dictionary or if any key is not an
            ``ElasticMaterialParameter``.
        ValueError
            If the dictionary does not define one complete supported
            parameterization.

        Examples
        --------
        Lame controls are passed as::

            {
                ElasticMaterialParameter.DENSITY: rho,
                ElasticMaterialParameter.LAMBDA: lmbda,
                ElasticMaterialParameter.MU: mu,
            }

        Velocity controls are passed as::

            {
                ElasticMaterialParameter.DENSITY: rho,
                ElasticMaterialParameter.P_WAVE_VELOCITY: c,
                ElasticMaterialParameter.S_WAVE_VELOCITY: c_s,
            }
        """
        if not isinstance(controls, dict):
            raise TypeError(
                "IsotropicWave controls must be provided as a dictionary.",
            )

        if not all(isinstance(key, ElasticMaterialParameter) for key in controls):
            raise TypeError(
                "IsotropicWave control keys must be ElasticMaterialParameter "
                "enum values.",
            )

        lame_controls = CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            ElasticMaterialParameterization.LAME
        ]
        velocity_controls = CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            ElasticMaterialParameterization.VELOCITY
        ]
        option_1 = set(controls) == set(lame_controls)
        option_2 = set(controls) == set(velocity_controls)
        if not (option_1 or option_2):
            lame_names = _format_control_parameters(lame_controls)
            velocity_names = _format_control_parameters(velocity_controls)
            raise ValueError(
                "Elastic controls must define either "
                f"{lame_names} or {velocity_names}.",
            )

        self.rho = self._as_control_field(
            controls[ElasticMaterialParameter.DENSITY],
            ElasticMaterialParameter.DENSITY.value,
        )

        synthetic_data = {
            "type": "object",
            "density": self.rho,
            "real_velocity_file": None,
        }
        if option_1:
            self.lmbda = self._as_control_field(
                controls[ElasticMaterialParameter.LAMBDA],
                ElasticMaterialParameter.LAMBDA.value,
            )
            self.mu = self._as_control_field(
                controls[ElasticMaterialParameter.MU],
                ElasticMaterialParameter.MU.value,
            )
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
            self._control_parameterization = ElasticMaterialParameterization.LAME
            synthetic_data["lambda"] = self.lmbda
            synthetic_data["mu"] = self.mu
            synthetic_data["delta"] = self.delta
            synthetic_data["epsilon"] = self.epsilon
            synthetic_data["gamma"] = self.gamma
            synthetic_data["theta"] = self.theta
            synthetic_data["phi"] = self.phi
            synthetic_data["anisotropy"] = self.anisotropy_type
        else:
            self.c = self._as_control_field(
                controls[ElasticMaterialParameter.P_WAVE_VELOCITY],
                ElasticMaterialParameter.P_WAVE_VELOCITY.value,
            )
            self.c_s = self._as_control_field(
                controls[ElasticMaterialParameter.S_WAVE_VELOCITY],
                ElasticMaterialParameter.S_WAVE_VELOCITY.value,
            )
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
            self._control_parameterization = ElasticMaterialParameterization.VELOCITY
            synthetic_data["p_wave_velocity"] = self.c
            synthetic_data["s_wave_velocity"] = self.c_s
            synthetic_data["delta"] = self.delta
            synthetic_data["epsilon"] = self.epsilon
            synthetic_data["gamma"] = self.gamma
            synthetic_data["theta"] = self.theta
            synthetic_data["phi"] = self.phi
            synthetic_data["anisotropy"] = self.anisotropy_type

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
                dt_scheme = abc_dict.get("nrbc", {}).get("dt_scheme", None)
                if dt_scheme == "backward_2nd":
                    self.u_nm2 = Function(self.function_space,
                                          name=self.get_function_name())

        self.mechanical_energy = mechanical_energy_form(self)

        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()

        self.Elastic_C = C_computation(self)

        if self.abc_type in [AbsorbingBCsType.NRBC, AbsorbingBCsType.NOABCS]:
            elastic_without_pml(self)
        elif self.abc_type == AbsorbingBCsType.PML:
            isotropic_elastic_with_pml(self)
