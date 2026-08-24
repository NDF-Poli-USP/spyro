import numpy as np

from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       project)
from pyadjoint import AdjFloat, Tape

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.physical_parameters import PhysicalParameters
from ...utils.typing import (AdjointType, ElasticMaterialParameter,
                             ElasticMaterialParameterization, AbsorbingBCsType,
                             RieszMapType)
from ...domains.space import create_function_space


PHYSICAL_PARAMETERIZATION = {
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


def _format_physical_parameters(parameters):
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
        #: Family whose independent fields already exist as ``Function``
        #: objects owned by this solver rather than by the model dictionary.
        self._materialized_parameterization = None
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

        Every forward solve calls this method again. Once a family has been
        materialized by :meth:`_set_physical_parameterization` the independent
        fields are owned by this solver, so they are kept as they are rather
        than rebuilt from the dictionary: they hold the current values, and
        rebuilding them would break the identity that pyadjoint controls and
        already assembled forms rely on.

        Parameters
        ----------
        synthetic_data_dict : dict
            Material parameter dictionary using the public Spyro model schema.
            Valid combinations are ``density``, ``lambda`` (or ``lame_first``),
            and ``mu`` (or ``lame_second``); or ``density``,
            ``p_wave_velocity``, and ``s_wave_velocity``. Values may be
            scalars, Firedrake ``Constant`` objects, Firedrake ``Function``
            objects, or UFL expressions. The dictionary is only read from.

        Returns
        -------
        None
            The method assigns ``rho``, ``lmbda``, ``mu``, ``c``, ``c_s``, and
            the active physical parameterization on ``self``.
        """
        if self._materialized_parameterization is not None:
            self._register_physical_parameters()
            return

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
        self._register_physical_parameters()

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

    def _register_physical_parameters(self) -> None:
        """Publish the material fields under their parameter names.

        Both independent fields and the expressions computed from them are
        registered, so callers can read any elastic parameter regardless of
        which family the equation is currently written in.

        Returns
        -------
        None
        """
        add = self._physical_parameters.add
        add(ElasticMaterialParameter.DENSITY, self.rho)
        add(ElasticMaterialParameter.LAMBDA, self.lmbda)
        add(ElasticMaterialParameter.MU, self.mu)
        add(ElasticMaterialParameter.P_WAVE_VELOCITY, self.c)
        add(ElasticMaterialParameter.S_WAVE_VELOCITY, self.c_s)

    def _set_physical_parameterization(
        self, parameterization: ElasticMaterialParameterization,
    ) -> None:
        """Materialize one independent elastic parameter family.

        The target family becomes three scalar ``Function`` objects and the
        complementary family is relinked to them through UFL expressions, so
        updating an independent field carries through to the dependent ones
        and to the assembled variational forms.

        Once a family has been materialized the fields belong to this object
        rather than to the model dictionary: they may be registered as
        pyadjoint controls, and rebuilding them from the dictionary would
        both discard the current values and orphan those controls. This is
        why ``_materialized_parameterization`` makes
        :meth:`initialize_model_parameters_from_object` keep them.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Independent physical parameter family to materialize.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the family is not one this solver supports.
        """
        space = self._material_parameter_space()

        def as_field(value, parameter):
            """Return ``value`` as an independent field of ``parameter``."""
            if isinstance(value, Function):
                return value
            return Function(space, name=parameter.value).interpolate(value)

        if parameterization is ElasticMaterialParameterization.LAME:
            self.rho = as_field(self.rho, ElasticMaterialParameter.DENSITY)
            self.lmbda = as_field(self.lmbda, ElasticMaterialParameter.LAMBDA)
            self.mu = as_field(self.mu, ElasticMaterialParameter.MU)
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        elif parameterization is ElasticMaterialParameterization.VELOCITY:
            self.rho = as_field(self.rho, ElasticMaterialParameter.DENSITY)
            self.c = as_field(
                self.c, ElasticMaterialParameter.P_WAVE_VELOCITY,
            )
            self.c_s = as_field(
                self.c_s, ElasticMaterialParameter.S_WAVE_VELOCITY,
            )
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
        else:
            raise ValueError(
                "Unsupported elastic material parameterization: "
                f"{parameterization}.",
            )

        self._physical_parameterization = parameterization
        self._materialized_parameterization = parameterization
        self._register_physical_parameters()

    def select_physical_parameters(
        self, names: object = None,
    ) -> PhysicalParameters:
        """Resolve an independent isotropic-elastic selection.

        An isotropic elastic medium is written either in terms of
        ``{density, lambda, mu}`` or of
        ``{density, p_wave_velocity, s_wave_velocity}``, and carries the family
        it is not written in as expressions of the other. A selection is
        therefore valid when it fits inside one family, and asking for names
        from the family currently held as expressions changes the equation over
        to it.

        Parameters
        ----------
        names : ElasticMaterialParameter or iterable, optional
            Elastic parameters to resolve. ``None`` selects the whole family
            the equation is currently written in.

        Returns
        -------
        PhysicalParameters
            Selected names, mapped to the independent fields carrying them.

        Raises
        ------
        ValueError
            If the selection is empty, repeats a name, or spans both families.
        TypeError
            If a name is not an :class:`ElasticMaterialParameter` member.
        """
        current = self._physical_parameterization
        if current is None:
            raise ValueError(
                "Elastic material parameters have not been initialized. "
                "Call initialize_physical_parameters() first.",
            )
        if names is None:
            selected_names = list(PHYSICAL_PARAMETERIZATION[current])
        elif isinstance(names, ElasticMaterialParameter):
            selected_names = [names]
        else:
            selected_names = list(names)

        if not selected_names:
            raise ValueError("At least one elastic control parameter is required.")
        if not all(
            isinstance(name, ElasticMaterialParameter)
            for name in selected_names
        ):
            raise TypeError(
                "Elastic controls must be ElasticMaterialParameter enum "
                "members.",
            )
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("Elastic control parameters must be unique.")

        wanted = set(selected_names)
        candidates = [
            parameterization
            for parameterization, family in PHYSICAL_PARAMETERIZATION.items()
            if wanted <= set(family)
        ]
        if not candidates:
            families = " or ".join(
                _format_physical_parameters(family)
                for family in PHYSICAL_PARAMETERIZATION.values()
            )
            raise ValueError(
                f"Elastic controls must be a subset of either {families}; "
                f"got {_format_physical_parameters(selected_names)}.",
            )
        target = current if current in candidates else candidates[0]
        self._set_physical_parameterization(target)

        selected = PhysicalParameters()
        for parameter in PHYSICAL_PARAMETERIZATION[target]:
            if parameter in wanted:
                selected.add(parameter, self.physical_parameters[parameter])
        return selected

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
