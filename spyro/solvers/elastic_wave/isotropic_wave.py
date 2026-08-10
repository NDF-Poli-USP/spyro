from firedrake import (assemble, curl, DirichletBC, div, Function,
                       project)

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.typing import (
    AbsorbingBCsType,
    ElasticMaterialParameter,
    ElasticMaterialParameterization,
    override,
)
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

_MATERIAL_PARAMETER_ALIASES = {
    ElasticMaterialParameter.LAMBDA: ("lmbda", "lame_first"),
    ElasticMaterialParameter.MU: ("lame_second",),
}


class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''

    def __init__(self, dictionary, comm=None):
        self.rho = None   # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity
        self._control_parameterization = None
        self._material_parameter_function_space = None
        super().__init__(dictionary, comm=comm)

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

    def initialize_model_parameters(self, synthetic_data=None):
        """Initialize isotropic-elastic material parameters."""
        parameterization = self._control_parameterization
        if synthetic_data is not None or parameterization is None:
            data = synthetic_data
            if data is None:
                data = self.input_dictionary.get("synthetic_data")
            if not isinstance(data, dict) or "type" not in data:
                raise ValueError(
                    "Input dictionary must contain ['synthetic_data']['type']."
                )
            if data["type"] == "file":
                raise NotImplementedError(
                    "File-based isotropic-elastic material initialization is "
                    "not implemented."
                )
            if data["type"] != "object":
                raise ValueError(f"Invalid synthetic data type: {data['type']}")

            values = {}
            for parameter in ElasticMaterialParameter:
                names = (
                    parameter.value,
                    *_MATERIAL_PARAMETER_ALIASES.get(parameter, ()),
                )
                for name in names:
                    if name in data:
                        values[parameter] = data[name]
                        break

            provided_parameters = set(values)
            lame_parameters = set(CONTROL_PARAMETERS_BY_PARAMETERIZATION[
                ElasticMaterialParameterization.LAME
            ])
            velocity_parameters = set(CONTROL_PARAMETERS_BY_PARAMETERIZATION[
                ElasticMaterialParameterization.VELOCITY
            ])
            if provided_parameters == lame_parameters:
                parameterization = ElasticMaterialParameterization.LAME
            elif provided_parameters == velocity_parameters:
                parameterization = ElasticMaterialParameterization.VELOCITY
            else:
                raise ValueError(
                    "Inconsistent selection of isotropic elastic wave "
                    f"parameters: {provided_parameters}. The valid options are "
                    "{density, lambda, mu} or "
                    "{density, p_wave_velocity, s_wave_velocity}."
                )

            if self.mesh is None:
                self.set_mesh()
            elif self.function_space is None:
                self.force_rebuild_function_space()

            fields = {
                ElasticMaterialParameter.DENSITY: self.rho,
                ElasticMaterialParameter.LAMBDA: self.lmbda,
                ElasticMaterialParameter.MU: self.mu,
                ElasticMaterialParameter.P_WAVE_VELOCITY: self.c,
                ElasticMaterialParameter.S_WAVE_VELOCITY: self.c_s,
            }
            for parameter, value in values.items():
                material_field = self.set_material_property(
                    parameter.value,
                    "scalar",
                    value=value,
                )
                fields[parameter].assign(material_field)

            self._control_parameterization = parameterization

        if parameterization is ElasticMaterialParameterization.LAME:
            self.c.interpolate(((self.lmbda + 2*self.mu)/self.rho)**0.5)
            self.c_s.interpolate((self.mu/self.rho)**0.5)
        else:
            self.mu.interpolate(self.rho*self.c_s**2)
            self.lmbda.interpolate(self.rho*self.c**2 - 2*self.mu)
        self._model_parameters_initialized = True

    @override
    def _create_function_space(self):
        return create_function_space(self.mesh, self.method, self.degree,
                                     dim=self.dimension)

    def _build_function_space(self):
        super()._build_function_space()
        V = create_function_space(self.mesh, self.method, self.degree)
        self.scalar_function_space = V
        self._material_parameter_function_space = V
        self.rho = Function(V, name=ElasticMaterialParameter.DENSITY.value)
        self.lmbda = Function(V, name=ElasticMaterialParameter.LAMBDA.value)
        self.mu = Function(V, name=ElasticMaterialParameter.MU.value)
        self.c = Function(
            V,
            name=ElasticMaterialParameter.P_WAVE_VELOCITY.value,
        )
        self.c_s = Function(
            V,
            name=ElasticMaterialParameter.S_WAVE_VELOCITY.value,
        )
        self._control_parameterization = None
        self._model_parameters_initialized = False

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
        if self._material_parameter_function_space is None:
            self._material_parameter_function_space = create_function_space(
                self.mesh, self.method, self.degree,
            )
        return self._material_parameter_function_space

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
            return None
        if parameterization is ElasticMaterialParameterization.LAME:
            return {
                ElasticMaterialParameter.DENSITY: self.rho,
                ElasticMaterialParameter.LAMBDA: self.lmbda,
                ElasticMaterialParameter.MU: self.mu,
            }
        return {
            ElasticMaterialParameter.DENSITY: self.rho,
            ElasticMaterialParameter.P_WAVE_VELOCITY: self.c,
            ElasticMaterialParameter.S_WAVE_VELOCITY: self.c_s,
        }

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

        synthetic_data = {
            "type": "object",
            **{parameter.value: value for parameter, value in controls.items()},
            "real_velocity_file": None,
        }
        self.initialize_model_parameters(synthetic_data=synthetic_data)
        self.input_dictionary["synthetic_data"] = {
            "type": "object",
            **{
                parameter.value: value
                for parameter, value in self.get_control_parameters().items()
            },
            "real_velocity_file": None,
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
