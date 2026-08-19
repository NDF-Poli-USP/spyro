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
from ...utils.physical_parameters import (DENSITY, LAMBDA, MU,
                                          P_WAVE_VELOCITY, S_WAVE_VELOCITY)


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
    _physical_parameter_names = frozenset({
        DENSITY, LAMBDA, MU, P_WAVE_VELOCITY, S_WAVE_VELOCITY,
    })

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

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        """Initialize isotropic elastic material parameters from a dictionary.

        The dictionary must define exactly one supported material
        parameterization: either density with Lame parameters, or density with
        P- and S-wave velocities. The missing dependent parameters are computed
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

    def physical_parameter_function_space(self):
        """Return the scalar space the elastic material fields live in.

        Elastic displacement is vector-valued, but density, the Lame
        parameters and the wave speeds are scalar material fields, so they do
        not live in the solver function space.

        Returns
        -------
        firedrake.FunctionSpace
            Scalar material-parameter function space.

        Raises
        ------
        ValueError
            If the mesh has not been created yet.
        """
        if self.mesh is None:
            raise ValueError(
                "Mesh must be set before creating elastic material parameter "
                "spaces.",
            )
        self._material_parameter_function_space = create_function_space(
            self.mesh, self.method, self.degree,
        )
        return self._material_parameter_function_space

    def _assign_physical_parameter(self, name, field):
        """Store a material field built by ``set_physical_parameter``."""
        if name == DENSITY:
            self.rho = field
        elif name == LAMBDA:
            self.lmbda = field
        elif name == MU:
            self.mu = field
        elif name == P_WAVE_VELOCITY:
            self.c = field
        else:
            self.c_s = field

    def _register_physical_parameters(self):
        """Declare the five isotropic elastic parameters of the medium."""
        self._physical_parameters.add(DENSITY, self.rho)
        self._physical_parameters.add(LAMBDA, self.lmbda)
        self._physical_parameters.add(MU, self.mu)
        self._physical_parameters.add(P_WAVE_VELOCITY, self.c)
        self._physical_parameters.add(S_WAVE_VELOCITY, self.c_s)

    def _refresh_dependent_parameters(self):
        """Recompute the parameter pair the medium was not declared with.

        The medium is described by density and either the Lame parameters or
        the wave speeds. The other pair is a UFL expression of the declared
        fields, so it follows them without being rebuilt; it is recomputed
        here only when the declared fields themselves are replaced.
        """
        declared = {
            name
            for name, value in (
                (DENSITY, self.rho),
                (LAMBDA, self.lmbda),
                (MU, self.mu),
                (P_WAVE_VELOCITY, self.c),
                (S_WAVE_VELOCITY, self.c_s),
            )
            if isinstance(value, Function)
        }
        for parameterization, parameters in PHYSICAL_PARAMETERIZATION.items():
            if declared != {parameter.value for parameter in parameters}:
                continue
            self._physical_parameterization = parameterization
            if parameterization is ElasticMaterialParameterization.LAME:
                self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
                self.c_s = (self.mu/self.rho)**0.5
            else:
                self.mu = self.rho*self.c_s**2
                self.lmbda = self.rho*self.c**2 - 2*self.mu
            self._register_physical_parameters()
            return

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
