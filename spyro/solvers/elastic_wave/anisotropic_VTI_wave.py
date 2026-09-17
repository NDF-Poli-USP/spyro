import numpy as np

from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       TensorFunctionSpace, project)

from .isotropic_wave import IsotropicWave
from .forms import (elastic_without_pml, viscoelastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.typing import (ElasticMaterialParameter, ElasticMaterialParameterization,
                             AbsorbingBCsType, override, WaveType, ViscoelasticMaterialParameter,
                             AnisotropicMaterialParameter)
from ...domains.space import create_function_space
from .tensor_computation import C_computation, build_Gamma

CONTROL_PARAMETERS_BY_PARAMETERIZATION = {
    ElasticMaterialParameterization.LAME: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.LAMBDA,
        ElasticMaterialParameter.MU
    ),
    ElasticMaterialParameterization.VELOCITY: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY
    ),
}
VISCOELASTIC_PARAMETERS = (ViscoelasticMaterialParameter.Q_VP, ViscoelasticMaterialParameter.Q_VS,
                           ViscoelasticMaterialParameter.Q_GAMMA, ViscoelasticMaterialParameter.Q_DELTA,
                           ViscoelasticMaterialParameter.Q_EPSILON)

ANISOTROPIC_PARAMETERS = (AnisotropicMaterialParameter.DELTA, AnisotropicMaterialParameter.EPSILON,)

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


class AnisotropicVTIWave(IsotropicWave):
    '''Anisotropic elastic wave propagator'''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, anisotropy = WaveType.ANISOTROPIC_VTI_ELASTIC, comm=comm)
        self.wave_type = WaveType.ANISOTROPIC_VTI_ELASTIC
        self.viscoelastic = dictionary.get("viscoelastic", False)
        self.delta = None
        self.epsilon = None
        self.gamma = None

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

        def declared(parameter, *aliases):
                    """Return the model value of ``parameter``, or ``None``."""
                    for key in (parameter.value, *aliases):
                        if key in synthetic_data_dict:
                            value = synthetic_data_dict[key]
                            return Constant(value) if np.isscalar(value) else value
                    return None
        
        self.rho = declared(ElasticMaterialParameter.DENSITY)
        self.lmbda = declared(ElasticMaterialParameter.LAMBDA, "lame_first")
        self.mu = declared(ElasticMaterialParameter.MU, "lame_second")
        self.c = declared(ElasticMaterialParameter.P_WAVE_VELOCITY)
        self.c_s = declared(ElasticMaterialParameter.S_WAVE_VELOCITY)
        self.gamma = declared(AnisotropicMaterialParameter.GAMMA)
        self.epsilon = declared(AnisotropicMaterialParameter.EPSILON)
        self.delta = declared(AnisotropicMaterialParameter.DELTA)
        self.Q_lambda = declared(ViscoelasticMaterialParameter.Q_LAMBDA)
        self.Q_mu = declared(ViscoelasticMaterialParameter.Q_MU)
        self.Q_vp = declared(ViscoelasticMaterialParameter.Q_VP)
        self.Q_vs = declared(ViscoelasticMaterialParameter.Q_VS)
        self.Q_epsilon = declared(ViscoelasticMaterialParameter.Q_EPSILON)
        self.Q_delta = declared(ViscoelasticMaterialParameter.Q_DELTA)
        self.Q_gamma = declared(ViscoelasticMaterialParameter.Q_GAMMA)
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
    def set_physical_parameterization(
        self, parameterization: ElasticMaterialParameterization,
    ) -> None:
        """Set which elastic parameters carry the material data.

        All five are read whatever this is set to: the variational form is
        written in density and the Lame parameters, while the absorbing
        boundary conditions and the stable timestep estimate are written in
        the two wave speeds. The chosen three become scalar ``Function``
        objects and the other two become UFL expressions of them, recomputed
        wherever they appear, so updating one of the chosen parameters
        carries through to the computed ones and to the assembled forms.

        This is a change of variables on the solver, not an edit of the
        model: the input dictionary is left as the user wrote it, and the
        set chosen here survives because initialization does not read the
        model a second time.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Set of elastic parameters to carry the data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the mesh has not been created, or the set of parameters is
            not one this solver supports.
        """
        space = None if self.mesh is None else create_function_space(
            self.mesh, self.method, self.degree, dim=1,
        )

        def as_function(value, parameter):
            """Return ``value`` as the independent field of ``parameter``.

            Before a mesh exists there is no space to build a ``Function``
            in, so the value is left as the scalar or ``Constant`` it came
            in as, and this set still carries the data.
            """
            if space is None or isinstance(value, Function):
                return value
            return Function(space, name=parameter.value).interpolate(value)

        self.viscoelastic = self.input_dictionary.get("viscoelastic", False)


        if parameterization is ElasticMaterialParameterization.LAME:
            self.rho = as_function(self.rho, ElasticMaterialParameter.DENSITY)
            self.lmbda = as_function(self.lmbda, ElasticMaterialParameter.LAMBDA)
            self.mu = as_function(self.mu, ElasticMaterialParameter.MU)
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
            if self.viscoelastic:
                self.Q_lambda = as_function(self.Q_lambda, ViscoelasticMaterialParameter.Q_LAMBDA)
                self.Q_mu = as_function(self.Q_mu, ViscoelasticMaterialParameter.Q_MU)
                self.Q_delta = as_function(self.Q_delta, ViscoelasticMaterialParameter.Q_DELTA)
                self.Q_gamma = as_function(self.Q_gamma, ViscoelasticMaterialParameter.Q_GAMMA)
                self.Q_epsilon = as_function(self.Q_epsilon, ViscoelasticMaterialParameter.Q_EPSILON)
        elif parameterization is ElasticMaterialParameterization.VELOCITY:
            self.rho = as_function(self.rho, ElasticMaterialParameter.DENSITY)
            self.c = as_function(
                self.c, ElasticMaterialParameter.P_WAVE_VELOCITY,
            )
            self.c_s = as_function(
                self.c_s, ElasticMaterialParameter.S_WAVE_VELOCITY,
            )
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
            if self.viscoelastic:
                self.Q_vp = as_function(self.Q_vp, ViscoelasticMaterialParameter.Q_VP)
                self.Q_vs = as_function(self.Q_vs, ViscoelasticMaterialParameter.Q_VS)
                self.Q_delta = as_function(self.Q_delta, ViscoelasticMaterialParameter.Q_DELTA)
                self.Q_gamma = as_function(self.Q_gamma, ViscoelasticMaterialParameter.Q_GAMMA)
                self.Q_epsilon = as_function(self.Q_epsilon, ViscoelasticMaterialParameter.Q_EPSILON)

        else:
            raise ValueError(
                "Unsupported elastic material parameterization: "
                f"{parameterization}.",
            )

        add = self._physical_parameters.add
        add(ElasticMaterialParameter.DENSITY, self.rho)
        add(ElasticMaterialParameter.LAMBDA, self.lmbda)
        add(ElasticMaterialParameter.MU, self.mu)
        add(ElasticMaterialParameter.P_WAVE_VELOCITY, self.c)
        add(ElasticMaterialParameter.S_WAVE_VELOCITY, self.c_s)
        
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
            elif parameter is ElasticMaterialParameter.ANISOTROPY_TYPE:
                parameters[parameter] = self.anisotropy
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

        if self.viscoelastic:
            
            d = self.input_dictionary.get("viscoelasticity", False)
            self.visco_type = d["visco_type"]
            W = TensorFunctionSpace(self.function_space.mesh(), "DG", 0)
            self.strain_space = W
            
            # GSLS parameters
            self.y_list     = d["y_gsls"]        # list of y_l
            self.omega_list = d["omega_gsls"]    # list of omega_l
            dim = self.function_space.mesh().topological_dimension()

            num_branches = d["branches"] 
            
            # Memory variables
            self.zeta_list = [Function(self.strain_space, name=f"Memory variable zeta_{i}")
                    for i in range(num_branches)]

            for zeta in self.zeta_list:
                zeta.assign(0.0)

            self.eps_np1 = Function(self.strain_space, name="eps_np1")
            self.eps_n   = Function(self.strain_space, name="eps_n")

            self.eps_n.assign(0.0)

            self.sigma_np1 = Function(self.strain_space, name="eps_np1")
            self.sigma_n   = Function(self.strain_space, name="eps_n")

            self.sigma_n.assign(0.0)

            self.Gamma = build_Gamma(self)

            if self.abc_type in [AbsorbingBCsType.NRBC, AbsorbingBCsType.NOABCS]:
                viscoelastic_without_pml(self)
            elif self.abc_type == AbsorbingBCsType.PML:
                viscoelastic_with_pml(self)
        else:
            if self.abc_type in [AbsorbingBCsType.NRBC, AbsorbingBCsType.NOABCS]:
                elastic_without_pml(self)
            elif self.abc_type == AbsorbingBCsType.PML:
                elastic_with_pml(self)
