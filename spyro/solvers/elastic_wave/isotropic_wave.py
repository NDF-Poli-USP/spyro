import numpy as np

from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       project)
from pyadjoint import AdjFloat, Tape

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.physical_parameters import (ELASTIC_PARAMETERIZATIONS,
                                          PhysicalParameters)
from ...utils.typing import (AdjointType, ElasticMaterialParameter,
                             ElasticMaterialParameterization, AbsorbingBCsType,
                             RieszMapType)
from ...domains.space import create_function_space


class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''

    #: An isotropic elastic medium is described by density plus either the
    #: two Lame parameters or the two wave speeds; whichever pair is not
    #: declared is computed from the other.
    _physical_parameter_names = frozenset(ElasticMaterialParameter)

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.rho = None   # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity
        self._physical_parameterization = None
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

    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        """Initialize isotropic elastic material parameters from a dictionary.

        The dictionary must define exactly one supported material
        parameterization: either density with Lame parameters, or density with
        P- and S-wave velocities. The missing dependent parameters are computed
        from the provided set, and the active physical parameterization is
        stored.

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
            the active physical parameterization on ``self``.
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
            self._physical_parameterization = ElasticMaterialParameterization.LAME
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        elif option_2:
            self._physical_parameterization = ElasticMaterialParameterization.VELOCITY
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
        add = self._physical_parameters.add
        add(ElasticMaterialParameter.DENSITY, self.rho)
        add(ElasticMaterialParameter.LAMBDA, self.lmbda)
        add(ElasticMaterialParameter.MU, self.mu)
        add(ElasticMaterialParameter.P_WAVE_VELOCITY, self.c)
        add(ElasticMaterialParameter.S_WAVE_VELOCITY, self.c_s)

    def _material_parameter_space(self) -> object:
        """Return the scalar space used for material parameters.

        Returns
        -------
        firedrake.FunctionSpace
            Scalar finite-element space on the elastic mesh.

        Raises
        ------
        ValueError
            If the mesh has not been created.
        """
        if self.mesh is None:
            raise ValueError(
                "Mesh must be set before creating elastic material fields.",
            )
        space = self._material_parameter_function_space
        if space is None or space.mesh() is not self.mesh:
            space = create_function_space(
                self.mesh, self.method, self.degree, dim=1,
            )
            self._material_parameter_function_space = space
        return space

    def _get_material_parameter(
        self, parameter: ElasticMaterialParameter,
    ) -> object:
        """Return one elastic physical field or expression.

        Parameters
        ----------
        parameter : ElasticMaterialParameter
            Physical parameter to retrieve.

        Returns
        -------
        object
            Firedrake field or dependent UFL expression.
        """
        if parameter is ElasticMaterialParameter.DENSITY:
            return self.rho
        if parameter is ElasticMaterialParameter.LAMBDA:
            return self.lmbda
        if parameter is ElasticMaterialParameter.MU:
            return self.mu
        if parameter is ElasticMaterialParameter.P_WAVE_VELOCITY:
            return self.c
        if parameter is ElasticMaterialParameter.S_WAVE_VELOCITY:
            return self.c_s
        raise ValueError(f"Unsupported elastic material parameter: {parameter}.")

    def _set_material_parameter(
        self, parameter: ElasticMaterialParameter, value: object,
    ) -> None:
        """Assign one elastic physical field or expression.

        Parameters
        ----------
        parameter : ElasticMaterialParameter
            Physical parameter to assign.
        value : object
            Firedrake field or dependent UFL expression.

        Returns
        -------
        None
        """
        if parameter is ElasticMaterialParameter.DENSITY:
            self.rho = value
        elif parameter is ElasticMaterialParameter.LAMBDA:
            self.lmbda = value
        elif parameter is ElasticMaterialParameter.MU:
            self.mu = value
        elif parameter is ElasticMaterialParameter.P_WAVE_VELOCITY:
            self.c = value
        elif parameter is ElasticMaterialParameter.S_WAVE_VELOCITY:
            self.c_s = value
        else:
            raise ValueError(
                f"Unsupported elastic material parameter: {parameter}.",
            )

    def _derive_complementary_parameters(
        self, parameterization: ElasticMaterialParameterization,
    ) -> None:
        """Express dependent parameters using one independent family.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Family whose three parameters are independent fields.

        Returns
        -------
        None
        """
        if parameterization is ElasticMaterialParameterization.LAME:
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        else:
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu

    def _register_physical_parameters(self) -> None:
        """Register the current elastic fields and expressions.

        Returns
        -------
        None
        """
        for parameter in ElasticMaterialParameter:
            self._physical_parameters.add(
                parameter, self._get_material_parameter(parameter),
            )

    def _record_parameterization(
        self, parameterization: ElasticMaterialParameterization,
    ) -> None:
        """Persist independent fields for the next forward initialization.

        ``forward_solve`` reads the material dictionary again. Storing the
        actual independent ``Function`` objects preserves their identity across
        model initialization.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Independent family to write to ``synthetic_data``.

        Returns
        -------
        None
        """
        synthetic_data = self.input_dictionary["synthetic_data"]
        for parameter in ElasticMaterialParameter:
            synthetic_data.pop(parameter.value, None)
        synthetic_data.pop("lame_first", None)
        synthetic_data.pop("lame_second", None)
        for parameter in ELASTIC_PARAMETERIZATIONS[parameterization]:
            synthetic_data[parameter.value] = self._get_material_parameter(
                parameter,
            )

    def _set_physical_parameterization(
        self, parameterization: ElasticMaterialParameterization,
    ) -> None:
        """Materialize one independent elastic parameter family.

        This change of variables occurs before tape recording. The target
        family becomes three scalar ``Function`` objects and the complementary
        family remains algebraically linked through UFL expressions.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Independent physical parameter family to materialize.

        Returns
        -------
        None
        """
        if parameterization is self._physical_parameterization:
            # Model dictionaries often contain scalars. Replace them with the
            # initialized fields so forward_solve() does not create new
            # Functions and invalidate external references to these fields.
            self._record_parameterization(parameterization)
            return

        space = self._material_parameter_space()
        for parameter in ELASTIC_PARAMETERIZATIONS[parameterization]:
            value = self._get_material_parameter(parameter)
            if isinstance(value, Function):
                field = value
            else:
                field = Function(space, name=parameter.value).interpolate(value)
            self._set_material_parameter(parameter, field)

        self._physical_parameterization = parameterization
        self._derive_complementary_parameters(parameterization)
        self._register_physical_parameters()
        self._record_parameterization(parameterization)

    def gradient_solve(
        self,
        misfit=None,
        forward_solution=None,
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        riesz_map=RieszMapType.L2,
    ) -> PhysicalParameters:
        """Compute automated-adjoint elastic material derivatives.

        Parameters
        ----------
        misfit : array_like, optional
            Accepted for compatibility; the recorded functional owns the
            elastic misfit.
        forward_solution : firedrake.Function, optional
            Accepted for compatibility; the automated adjoint replays its tape.
        adjoint_type : AdjointType, optional
            Must be :attr:`AdjointType.AUTOMATED_ADJOINT`.
        riesz_map : RieszMapType, optional
            ``L2`` returns primal gradients and ``l2`` raw derivatives.

        Returns
        -------
        PhysicalParameters
            Derivatives keyed by the selected elastic parameter enums.

        Raises
        ------
        NotImplementedError
            If a hand-implemented adjoint or unsupported Riesz map is requested.
        ValueError
            If no valid annotated functional is available.
        """
        if adjoint_type is not AdjointType.AUTOMATED_ADJOINT:
            raise NotImplementedError(
                "Elastic gradients only support the automated adjoint.",
            )
        if not isinstance(self.functional_value, AdjFloat):
            raise ValueError(
                "Functional value must be an AdjFloat for automated adjoint "
                "gradient computation.",
            )
        if self.automated_adjoint is None:
            raise ValueError(
                "Enable the automated adjoint before the elastic forward solve.",
            )
        if (
            self.automated_adjoint.reduced_functional is None
            and isinstance(self.automated_adjoint._tape, Tape)
        ):
            self.automated_adjoint.create_reduced_functional(
                self.functional_value,
            )

        if riesz_map is RieszMapType.L2:
            derivatives = self.automated_adjoint.compute_gradient()
        elif riesz_map is RieszMapType.l2:
            derivatives = self.automated_adjoint.compute_derivative()
        else:
            raise NotImplementedError(
                f"Riesz map {riesz_map} not implemented for automated adjoint.",
            )
        return self.automated_adjoint.label_derivatives(derivatives)

    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError

    def _create_function_space(self):
        return create_function_space(self.mesh, self.method, self.degree,
                                     dim=self.dimension)

    def _set_vstate(self, vstate):
        self.u_n.assign(vstate)

    def _get_vstate(self):
        return self.u_n

    def _set_prev_vstate(self, vstate):
        if self.u_nm2 is not None:
            self.u_nm2.assign(self.u_nm1)
        self.u_nm1.assign(vstate)

    def _get_prev_vstate(self):
        return self.u_nm1

    def _set_next_vstate(self, vstate):
        self.u_np1.assign(vstate)

    def _get_next_vstate(self):
        return self.u_np1

    def get_forward_solution_receivers(self):
        if self.abc_type == AbsorbingBCsType.PML:
            raise NotImplementedError
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    def get_function(self):
        return self.u_n

    def get_function_name(self):
        return "Displacement"

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
