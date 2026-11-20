import numpy as np

from firedrake import (Constant, Function, grad, interpolate, sym)

from .isotropic_wave import IsotropicWave
from .forms import (elastic_without_pml, viscoelastic_without_pml,
                    elastic_with_pml, viscoelastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.typing import (ElasticMaterialParameter, ElasticMaterialParameterization,
                             AbsorbingBCsType, override, WaveType, ViscoelasticMaterialParameter,
                             AnisotropicMaterialParameter)
from ...domains.space import create_function_space
from .tensor_computation import C_computation, build_Gamma, create_strain_space

CONTROL_PARAMETERS_BY_PARAMETERIZATION = {
    ElasticMaterialParameterization.LAME: (
        ElasticMaterialParameter.DENSITY,
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
    """
    return "{" + ", ".join(parameter.value for parameter in parameters) + "}"


class AnisotropicVTIWave(IsotropicWave):
    '''Anisotropic elastic wave propagator'''

    def _declared(self, synthetic_data_dict: dict, parameter, *aliases):
        """Return the model value of ``parameter``, or ``None``.

        Parameters
        ----------
        synthetic_data_dict : dict
            Model dictionary to read from.
        parameter : enum
            Material parameter whose ``.value`` is the primary key.
        *aliases : str
            Alternative keys accepted for the same parameter.

        Returns
        -------
        firedrake.Constant, firedrake.Function, UFL expression or None
            The declared value, with scalars wrapped in ``Constant``.
        """
        for key in (parameter.value, *aliases):
            if key in synthetic_data_dict:
                value = synthetic_data_dict[key]
                return Constant(value) if np.isscalar(value) else value
        return None

    def __init__(self, dictionary: dict, comm=None,
                 wave_type: WaveType = WaveType.ANISOTROPIC_VTI_ELASTIC) -> None:
        super().__init__(dictionary, anisotropy=wave_type, comm=comm)
        self.wave_type = wave_type
        self.viscoelastic = dictionary.get("viscoelastic", False)
        self.delta = None
        self.epsilon = None
        self.gamma = None
        self.anisotropy_type = None
        self.Q_vp = None
        self.Q_vs = None
        self.Q_delta = None
        self.Q_gamma = None
        self.Q_epsilon = None

    def _anisotropic_synthetic_data(self) -> dict:
        """Return the anisotropy entries of the ``synthetic_data`` dictionary.

        Returns
        -------
        dict
            Thomsen parameters and anisotropy type keyed by their model-schema names.
        """
        return {
            "delta": self.delta,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "anisotropy": self.anisotropy_type,
        }

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict) -> None:
        """Initialize VTI elastic material parameters from a dictionary.

        The dictionary must define density with P- and S-wave velocities;
        the Lame parameters are computed from them. The Thomsen parameters,
        the anisotropy type and, for viscoelastic runs, the quality factors
        are read as well.

        Parameters
        ----------
        synthetic_data_dict : dict
            Material parameter dictionary using the public Spyro model schema.
            Must contain ``density``, ``p_wave_velocity``,
            ``s_wave_velocity`` and ``anisotropy``, and must not contain
            ``lambda``/``lame_first`` or ``mu``/``lame_second``. Values may
            be scalars, Firedrake ``Constant`` objects, Firedrake ``Function``
            objects, or UFL expressions.

        Returns
        -------
        None
            The method assigns ``rho``, ``c``, ``c_s``,
            the anisotropic and viscoelastic parameters, and the active
            control parameterization on ``self``.

        Raises
        ------
        ValueError
            If density or either velocity is missing, or if a Lame parameter
            is declared.
        """
        d = synthetic_data_dict
        self.rho = self._declared(d, ElasticMaterialParameter.DENSITY)
        self.c = self._declared(d, ElasticMaterialParameter.P_WAVE_VELOCITY)
        self.c_s = self._declared(d, ElasticMaterialParameter.S_WAVE_VELOCITY)
        self.gamma = self._declared(d, AnisotropicMaterialParameter.GAMMA)
        self.epsilon = self._declared(d, AnisotropicMaterialParameter.EPSILON)
        self.delta = self._declared(d, AnisotropicMaterialParameter.DELTA)
        self.Q_vp = self._declared(d, ViscoelasticMaterialParameter.Q_VP)
        self.Q_vs = self._declared(d, ViscoelasticMaterialParameter.Q_VS)
        self.Q_epsilon = self._declared(d, ViscoelasticMaterialParameter.Q_EPSILON)
        self.Q_delta = self._declared(d, ViscoelasticMaterialParameter.Q_DELTA)
        self.Q_gamma = self._declared(d, ViscoelasticMaterialParameter.Q_GAMMA)
        self.anisotropy_type = d["anisotropy"]

        lame_declared = (
            self._declared(d, ElasticMaterialParameter.LAMBDA, "lame_first") is not None
            or self._declared(d, ElasticMaterialParameter.MU, "lame_second") is not None
        )
        if self.rho is None or self.c is None or self.c_s is None or lame_declared:
            raise ValueError(
                "Inconsistent selection of anisotropic elastic wave parameters:\n"
                f"    Density        : {self.rho is not None}\n"
                f"    P-wave velocity: {self.c is not None}\n"
                f"    S-wave velocity: {self.c_s is not None}\n"
                f"    Lame parameter : {lame_declared}\n"
                "Anisotropic elastic models must define exactly "
                "{Density, P-wave velocity, S-wave velocity}.",
            )

        self._control_parameterization = ElasticMaterialParameterization.VELOCITY
        self.set_physical_parameterization(ElasticMaterialParameterization.VELOCITY)

    @override
    def set_physical_parameterization(
        self, parameterization: ElasticMaterialParameterization,
    ) -> None:
        """Set which elastic parameters carry the material data.

        Anisotropic solvers support only the velocity parameterization:
        density and the two wave speeds become scalar ``Function`` objects,
        and the Lame parameters become UFL expressions of them, recomputed
        wherever they appear, so updating a velocity carries through to the
        assembled forms. For viscoelastic runs, the quality factors are
        converted to ``Function`` objects as well.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Set of elastic parameters to carry the data. Must be
            ``ElasticMaterialParameterization.VELOCITY``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``parameterization`` is not the velocity parameterization.
        """
        if parameterization is not ElasticMaterialParameterization.VELOCITY:
            raise ValueError(
                "Unsupported elastic material parameterization for anisotropic "
                f"solvers: {parameterization}.",
            )

        space = None if self.mesh is None else create_function_space(
            self.mesh, self.method, self.degree, dim=1,
        )

        def as_function(value, parameter):
            """Return ``value`` as the independent field of ``parameter``.

            Before a mesh exists there is no space to build a ``Function``
            in, so the value is left as the scalar or ``Constant`` it came
            in as.

            Parameters
            ----------
            value : scalar, firedrake.Constant, firedrake.Function, or UFL expression
                Value of the material parameter.
            parameter : enum
                Material parameter whose ``.value`` names the ``Function``.

            Returns
            -------
            firedrake.Function or object
                ``value`` as a scalar ``Function``, or unchanged if it
                already is one or no mesh exists yet.
            """
            if space is None or isinstance(value, Function):
                return value
            return Function(space, name=parameter.value).interpolate(value)

        self.rho = as_function(self.rho, ElasticMaterialParameter.DENSITY)
        self.c = as_function(self.c, ElasticMaterialParameter.P_WAVE_VELOCITY)
        self.c_s = as_function(self.c_s, ElasticMaterialParameter.S_WAVE_VELOCITY)
        if self.viscoelastic:
            self.Q_vp = as_function(self.Q_vp, ViscoelasticMaterialParameter.Q_VP)
            self.Q_vs = as_function(self.Q_vs, ViscoelasticMaterialParameter.Q_VS)
            self.Q_delta = as_function(self.Q_delta, ViscoelasticMaterialParameter.Q_DELTA)
            self.Q_gamma = as_function(self.Q_gamma, ViscoelasticMaterialParameter.Q_GAMMA)
            self.Q_epsilon = as_function(self.Q_epsilon, ViscoelasticMaterialParameter.Q_EPSILON)

        add = self._physical_parameters.add
        add(ElasticMaterialParameter.DENSITY, self.rho)
        add(ElasticMaterialParameter.P_WAVE_VELOCITY, self.c)
        add(ElasticMaterialParameter.S_WAVE_VELOCITY, self.c_s)

    @override
    def get_control_parameters(self) -> dict | None:
        """Return the active anisotropic elastic material controls.

        Returns
        -------
        dict or None
            Dictionary mapping ``DENSITY``, ``P_WAVE_VELOCITY`` and
            ``S_WAVE_VELOCITY`` to their scalar Firedrake ``Function``
            controls. Returns ``None`` if material parameters have not been
            initialized.

        Examples
        --------
        ``{DENSITY: rho, P_WAVE_VELOCITY: c, S_WAVE_VELOCITY: c_s}``.
        """
        if self.rho is None:
            return None
        return {
            ElasticMaterialParameter.DENSITY: self.rho,
            ElasticMaterialParameter.P_WAVE_VELOCITY: self.c,
            ElasticMaterialParameter.S_WAVE_VELOCITY: self.c_s,
        }

    @override
    def set_control_parameters(self, controls: dict) -> None:
        """Assign anisotropic elastic material controls.

        Control dictionaries must use :class:`ElasticMaterialParameter` keys.
        Model input dictionaries still use the public Spyro string schema, but
        the FWI control API is intentionally enum-only.

        Parameters
        ----------
        controls : dict
            Dictionary containing density, P-wave velocity and S-wave
            velocity controls. Values may be Firedrake ``Function`` objects,
            Firedrake ``Constant`` objects, scalars, or UFL expressions; all
            stored controls are scalar ``Function`` objects.

        Returns
        -------
        None
            The method updates ``rho``, ``c``, ``c_s``, the Lame parameters
            computed from them, and the stored ``synthetic_data``.

        Raises
        ------
        TypeError
            If ``controls`` is not a dictionary or if any key is not an
            ``ElasticMaterialParameter``.
        ValueError
            If the dictionary does not define exactly density and the two
            wave velocities.

        Examples
        --------
        Velocity controls are passed as::

            {
                ElasticMaterialParameter.DENSITY: rho,
                ElasticMaterialParameter.P_WAVE_VELOCITY: c,
                ElasticMaterialParameter.S_WAVE_VELOCITY: c_s,
            }
        """
        if not isinstance(controls, dict):
            raise TypeError(
                "Anisotropic wave controls must be provided as a dictionary.",
            )

        if not all(isinstance(key, ElasticMaterialParameter) for key in controls):
            raise TypeError(
                "Anisotropic wave control keys must be ElasticMaterialParameter "
                "enum values.",
            )

        velocity_controls = CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            ElasticMaterialParameterization.VELOCITY
        ]
        if set(controls) != set(velocity_controls):
            raise ValueError(
                "Anisotropic elastic controls must define "
                f"{_format_control_parameters(velocity_controls)}.",
            )

        self.rho = self._as_control_field(
            controls[ElasticMaterialParameter.DENSITY],
            ElasticMaterialParameter.DENSITY.value,
        )
        self.c = self._as_control_field(
            controls[ElasticMaterialParameter.P_WAVE_VELOCITY],
            ElasticMaterialParameter.P_WAVE_VELOCITY.value,
        )
        self.c_s = self._as_control_field(
            controls[ElasticMaterialParameter.S_WAVE_VELOCITY],
            ElasticMaterialParameter.S_WAVE_VELOCITY.value,
        )
        self._control_parameterization = ElasticMaterialParameterization.VELOCITY

        self.input_dictionary["synthetic_data"] = {
            "type": "object",
            "density": self.rho,
            "real_velocity_file": None,
            "p_wave_velocity": self.c,
            "s_wave_velocity": self.c_s,
            **self._anisotropic_synthetic_data(),
        }

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

        # self.mechanical_energy = mechanical_energy_form(self) 

        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()

        self.Elastic_C = C_computation(self)

        if self.viscoelastic:
            d = self.input_dictionary["viscoelasticity"]
            self.visco_type = d["visco_type"]
            self.strain_space = create_strain_space(self)

            # GSLS parameters
            self.y_list = [float(y) for y in d["y_gsls"]]        # list of y_l
            self.omega_list = [float(w) for w in d["omega_gsls"]]    # list of omega_l

            num_branches = d["branches"]

            # Memory variables
            self.zeta_list = [Function(self.strain_space, name=f"Memory variable zeta_{i}")
                              for i in range(num_branches)]

            for zeta in self.zeta_list:
                zeta.assign(0.0)

            # Strain on the memory space, reused in each step
            self.eps_n = Function(self.strain_space, name="eps_n")

            # Symbolic interpolation, built once. After state rotation,
            # u_nm1 (= prev_vstate) contains u^n, as in the original.
            self.eps_interp = interpolate(sym(grad(self.u_nm1)),
                                          self.strain_space)

            dt = float(self.dt)
            self.zeta_a = [1.0 - dt * w for w in self.omega_list]
            self.zeta_b = [dt * w for w in self.omega_list]

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
