import numpy as np

from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       project)

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.typing import (ElasticMaterialParameter, ElasticMaterialParameterization,
                             AbsorbingBCsType, override)
from ...domains.space import create_function_space


CONTROL_PARAMETERS_BY_PARAMETERIZATION = {
    ElasticMaterialParameterization.LAME: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.LAMBDA,
        ElasticMaterialParameter.MU,
    ),
    ElasticMaterialParameterization.VELOCITY: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    ),
}


_CONTROL_PARAMETER_ALIASES = {
    parameter.value: parameter for parameter in ElasticMaterialParameter
}
_CONTROL_PARAMETER_ALIASES.update({
    "rho": ElasticMaterialParameter.DENSITY,
    "lmbda": ElasticMaterialParameter.LAMBDA,
    "c": ElasticMaterialParameter.P_WAVE_VELOCITY,
    "c_s": ElasticMaterialParameter.S_WAVE_VELOCITY,
})


class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.rho = None   # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity
        self._control_parameterization = None
        self._control_parameters = None
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
        """Initialize isotropic elastic material parameters from a dictionary.

        The material declaration must define one complete parameterization,
        while ``control_parameters`` may independently select any non-empty
        subset from either the Lame or velocity family. Without an explicit
        selection, all parameters from the material declaration are controls.

        Parameters
        ----------
        synthetic_data_dict : dict
            Material parameter dictionary using the public Spyro model schema.
            Valid material combinations are ``density``, ``lambda`` (or
            ``lame_first``), and ``mu`` (or ``lame_second``); or ``density``,
            ``p_wave_velocity``, and ``s_wave_velocity``. The optional
            ``control_parameters`` entry must be a list or tuple.

        Returns
        -------
        None
            The method assigns all material fields and records the selected
            control subset and its parameterization.
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

        rho = get_value(ElasticMaterialParameter.DENSITY)
        lmbda = get_value(
            ElasticMaterialParameter.LAMBDA,
            "lame_first",
        )
        mu = get_value(
            ElasticMaterialParameter.MU,
            "lame_second",
        )
        c = get_value(ElasticMaterialParameter.P_WAVE_VELOCITY)
        c_s = get_value(ElasticMaterialParameter.S_WAVE_VELOCITY)

        # Check if {rho, lambda, mu} is set and {c, c_s} are not
        option_1 = bool(rho) and bool(lmbda) and bool(mu) \
            and not bool(c) and not bool(c_s)
        # Check if {rho, c, c_s} is set and {lambda, mu} are not
        option_2 = bool(rho) and bool(c) and bool(c_s) \
            and not bool(lmbda) and not bool(mu)

        if option_1:
            equation_parameterization = ElasticMaterialParameterization.LAME
            self.rho, self.lmbda, self.mu = rho, lmbda, mu
            self._update_derived_material_parameters(equation_parameterization)
        elif option_2:
            equation_parameterization = ElasticMaterialParameterization.VELOCITY
            self.rho, self.c, self.c_s = rho, c, c_s
            self._update_derived_material_parameters(equation_parameterization)
        else:
            raise ValueError(
                "Inconsistent selection of isotropic elastic wave parameters:\n"
                f"    Density        : {bool(rho)}\n"
                f"    Lame first     : {bool(lmbda)}\n"
                f"    Lame second    : {bool(mu)}\n"
                f"    P-wave velocity: {bool(c)}\n"
                f"    S-wave velocity: {bool(c_s)}\n"
                "The valid options are {Density, Lame first, Lame second} "
                "or (exclusive) {Density, P-wave velocity, S-wave velocity}",
            )

        control_parameterization, controls = (
            self._normalize_control_parameter_selection(
                synthetic_data_dict.get("control_parameters"),
                equation_parameterization,
            )
        )
        if control_parameterization is not equation_parameterization:
            self._materialize_parameterization(control_parameterization)
            self._update_derived_material_parameters(control_parameterization)

        self._control_parameterization = control_parameterization
        self._control_parameters = controls

    def _normalize_control_parameter_selection(
        self,
        parameters,
        default_parameterization,
    ):
        """Normalize a public elastic-control selection to enum values."""
        if parameters is None:
            return (
                default_parameterization,
                CONTROL_PARAMETERS_BY_PARAMETERIZATION[default_parameterization],
            )
        if not isinstance(parameters, (list, tuple)):
            raise TypeError(
                "Elastic control_parameters must be a list or tuple.",
            )

        normalized = []
        for parameter in parameters:
            if isinstance(parameter, ElasticMaterialParameter):
                normalized.append(parameter)
            elif isinstance(parameter, str):
                try:
                    normalized.append(_CONTROL_PARAMETER_ALIASES[parameter])
                except KeyError as exc:
                    supported = ", ".join(_CONTROL_PARAMETER_ALIASES)
                    raise ValueError(
                        f"Unknown elastic control parameter '{parameter}'. "
                        f"Supported names are: {supported}.",
                    ) from exc
            else:
                raise TypeError(
                    "Elastic control parameters must be strings or "
                    "ElasticMaterialParameter values.",
                )

        if not normalized:
            raise ValueError("At least one elastic control parameter is required.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "Elastic control parameters must not contain duplicates."
            )

        selected = set(normalized)
        lame_parameters = set(CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            ElasticMaterialParameterization.LAME
        ])
        velocity_parameters = set(CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            ElasticMaterialParameterization.VELOCITY
        ])
        is_lame = selected <= lame_parameters
        is_velocity = selected <= velocity_parameters

        if is_lame and is_velocity:
            parameterization = default_parameterization
        elif is_lame:
            parameterization = ElasticMaterialParameterization.LAME
        elif is_velocity:
            parameterization = ElasticMaterialParameterization.VELOCITY
        else:
            raise ValueError(
                "Elastic controls cannot mix Lame parameters with wave "
                "velocities."
            )
        return parameterization, tuple(normalized)

    @staticmethod
    def _material_parameter_attribute(parameter):
        """Return the wave attribute associated with a material parameter."""
        return {
            ElasticMaterialParameter.DENSITY: "rho",
            ElasticMaterialParameter.LAMBDA: "lmbda",
            ElasticMaterialParameter.MU: "mu",
            ElasticMaterialParameter.P_WAVE_VELOCITY: "c",
            ElasticMaterialParameter.S_WAVE_VELOCITY: "c_s",
        }[parameter]

    def _get_material_parameter(self, parameter):
        """Return one material field or expression."""
        return getattr(self, self._material_parameter_attribute(parameter))

    def _set_material_parameter(self, parameter, value):
        """Set one material field or expression."""
        setattr(self, self._material_parameter_attribute(parameter), value)

    def _materialize_parameterization(self, parameterization):
        """Represent one complete primary parameter family as Functions."""
        if self.mesh is None:
            return
        for parameter in CONTROL_PARAMETERS_BY_PARAMETERIZATION[parameterization]:
            self._set_material_parameter(
                parameter,
                self._as_control_field(
                    self._get_material_parameter(parameter),
                    parameter.value,
                ),
            )

    def _update_derived_material_parameters(self, parameterization):
        """Update parameters derived from the active primary family."""
        if parameterization is ElasticMaterialParameterization.LAME:
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        else:
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu

    def get_equation_material_parameters(self):
        """Return density and elastic moduli from the active control family.

        Building the velocity-to-Lame conversion directly in the variational
        form keeps the dependency on ``c`` and ``c_s`` visible to pyadjoint.
        """
        if (
            self._control_parameterization
            is ElasticMaterialParameterization.VELOCITY
        ):
            mu = self.rho*self.c_s**2
            lmbda = self.rho*self.c**2 - 2*mu
            return self.rho, lmbda, mu
        return self.rho, self.lmbda, self.mu

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
        if self.abc_type == AbsorbingBCsType.PML:
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
        """Return the scalar space used for elastic material controls.

        Elastic displacement is vector-valued, but density, Lame parameters,
        and wave speeds are scalar material fields. This method creates and
        returns the scalar space used for those controls.

        Returns
        -------
        firedrake.FunctionSpace
            Scalar material-parameter function space.

        Raises
        ------
        ValueError
            If the mesh has not been created yet.

        Examples
        --------
        ``Function(wave.get_control_parameter_function_space())`` creates a
        scalar density or Lame-parameter control compatible with
        ``set_control_parameters``.
        """
        if self.mesh is None:
            raise ValueError(
                "Mesh must be set before creating elastic control parameter spaces.",
            )
        self._material_parameter_function_space = create_function_space(
            self.mesh, self.method, self.degree,
        )
        return self._material_parameter_function_space

    def _as_control_field(self, value, name):
        """Return a material control as a scalar Firedrake Function.

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

        Parameters
        ----------
        value : firedrake.Function, firedrake.Constant, scalar, or UFL expression
            Material control value to represent in the scalar control space.
        name : str
            Name assigned to the returned Firedrake ``Function``.

        Returns
        -------
        firedrake.Function or None
            Scalar control field in the material-parameter function space. If
            ``value`` is ``None``, returns ``None``.
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

    def get_control_parameters(self):
        """Return the active isotropic elastic material controls.

        The returned dictionary is keyed by
        :class:`ElasticMaterialParameter` and contains only the user-selected
        controls. Without an explicit selection, it contains every parameter
        from the material-input family.

        Returns
        -------
        dict or None
            Dictionary mapping material-parameter enum values to scalar
            Firedrake ``Function`` controls. Returns ``None`` if material
            parameters have not been initialized.

        Examples
        --------
        With ``control_parameters=["mu"]``, only ``{MU: mu}`` is returned.
        """
        parameterization = self._control_parameterization
        if parameterization is None:
            if self.rho is None:
                return None
            parameterization = ElasticMaterialParameterization.LAME

        selected = self._control_parameters
        if selected is None:
            selected = CONTROL_PARAMETERS_BY_PARAMETERIZATION[parameterization]
        return {
            parameter: self._get_material_parameter(parameter)
            for parameter in selected
        }

    def set_control_parameters(self, controls):
        """Assign isotropic elastic material controls.

        Control dictionaries must use :class:`ElasticMaterialParameter` keys
        and may contain a non-empty subset from either material family.
        Parameters omitted from the dictionary retain their current values.

        Parameters
        ----------
        controls : dict
            Dictionary containing a non-empty subset of density/Lame controls
            or density/P- and S-wave velocity controls.

        Returns
        -------
        None
            The method updates the selected controls, complementary material
            expressions and active parameterization.

        Raises
        ------
        TypeError
            If ``controls`` is not a dictionary or if any key is not an
            ``ElasticMaterialParameter``.
        ValueError
            If the dictionary is empty or mixes both parameter families.

        Examples
        --------
        A subset of Lame controls is passed as::

            {
                ElasticMaterialParameter.LAMBDA: lmbda,
                ElasticMaterialParameter.MU: mu,
            }

        A single velocity control is passed as::

            {
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

        lame_family = set(CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            ElasticMaterialParameterization.LAME
        ])
        velocity_family = set(CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            ElasticMaterialParameterization.VELOCITY
        ])
        selected_set = set(controls)
        current_parameterization = self._control_parameterization
        if current_parameterization is None:
            if selected_set == lame_family:
                current_parameterization = ElasticMaterialParameterization.LAME
            elif selected_set == velocity_family:
                current_parameterization = (
                    ElasticMaterialParameterization.VELOCITY
                )
            else:
                self._initialize_model_parameters()
                current_parameterization = self._control_parameterization

        parameterization, selected = self._normalize_control_parameter_selection(
            list(controls),
            current_parameterization,
        )
        if parameterization is not current_parameterization:
            self._materialize_parameterization(parameterization)

        for parameter, value in controls.items():
            self._set_material_parameter(
                parameter,
                self._as_control_field(value, parameter.value),
            )

        self._control_parameterization = parameterization
        self._control_parameters = selected
        self._update_derived_material_parameters(parameterization)

        primary_parameters = CONTROL_PARAMETERS_BY_PARAMETERIZATION[
            parameterization
        ]
        self.input_dictionary["synthetic_data"] = {
            "type": "object",
            **{
                parameter.value: self._get_material_parameter(parameter)
                for parameter in primary_parameters
            },
            "real_velocity_file": None,
            "control_parameters": [
                parameter.value for parameter in selected
            ],
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

        self.mechanical_energy = mechanical_energy_form(self)

        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()

        if self.abc_type in [AbsorbingBCsType.NRBC, AbsorbingBCsType.NOABCS]:
            isotropic_elastic_without_pml(self)
        elif self.abc_type == AbsorbingBCsType.PML:
            isotropic_elastic_with_pml(self)

    @override
    def rhs_no_pml(self):
        if self.abc_type == AbsorbingBCsType.PML:
            raise NotImplementedError
        else:
            return self.B

    def rhs_no_pml_source(self):
        if self.abc_type == AbsorbingBCsType.PML:
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
            self.D_h = create_function_space(self.mesh, "DG0", 0)
            self.p_wave = Function(self.D_h)

        self.p_wave.assign(project(div(self.get_function()), self.D_h))

        return self.p_wave

    def update_s_wave(self):
        if self.s_wave is None:
            if self.dimension == 2:
                self.C_h = create_function_space(self.mesh, "DG0", 0)
            else:
                self.C_h = create_function_space(self.mesh, "DG0", 0,
                                                 dim=self.dimension)
            self.s_wave = Function(self.C_h)

        self.s_wave.assign(project(curl(self.get_function()), self.C_h))

        return self.s_wave
